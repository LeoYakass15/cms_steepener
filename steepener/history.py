"""ECB history of the EUR 2Y and 10Y rates (AAA par yields, 2004-2026).

Used for: the historical 2Y/10Y correlation, historical stress tests, and three back-tests
(the investor's return, the desk's hedge and a monthly P&L explain).

The back-tests reprice the note every month, so they use a small closed-form "desk model"
instead of SABR + G2++:
- curve: zero rate linear between the 2Y and the 10Y rate (flat outside), + GS spread to discount,
- each future 2s10s fixing is normal around today's 2s10s, with a constant vol (Bachelier),
- the call is priced "to worst": the value is the lowest of the values if called at each call date
  or never called, and the issuer calls when the note is worth more than par on a call date.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .market import bachelier_call


def load_history(path="data/ecb/eur_aaa_par_yields_daily.csv"):
    """Daily 2Y and 10Y rates (decimals) and the 2s10s spread."""
    h = pd.read_csv(path, index_col="date", parse_dates=True)
    out = pd.DataFrame({"r2": h["aaa_2y_pct"] / 100, "r10": h["aaa_10y_pct"] / 100})
    out["slope"] = out["r10"] - out["r2"]
    return out


def monthly(hist):
    return hist.resample("ME").last().dropna()


# ---------- correlation ----------

def weekly_changes(hist):
    return hist[["r2", "r10"]].resample("W-FRI").last().diff().dropna()


def rolling_correlation(hist, weeks=52):
    """1-year rolling correlation of weekly changes in the 2Y and 10Y rates."""
    d = weekly_changes(hist)
    return d["r2"].rolling(weeks).corr(d["r10"]).dropna()


def correlation_estimate(hist, years=5):
    """Correlation of weekly changes over the last `years` years: the base case for pricing."""
    d = weekly_changes(hist)
    d = d[d.index > d.index[-1] - pd.DateOffset(years=years)]
    return float(d["r2"].corr(d["r10"]))


# ---------- historical stress tests ----------

STRESS_PERIODS = {
    "Lehman (Sep-Dec 2008)": ("2008-09-12", "2008-12-31"),
    "Euro crisis (Jul-Nov 2011)": ("2011-07-01", "2011-11-30"),
    "Rates go negative (2014-2016)": ("2014-01-02", "2016-07-08"),
    "Covid (Feb-Mar 2020)": ("2020-02-19", "2020-03-18"),
    "ECB hikes (2022)": ("2021-12-31", "2022-10-21"),
    "Curve inversion (Oct 2022-Sep 2023)": ("2022-10-21", "2023-09-29"),
}


def stress_moves(hist):
    """Real moves of the 2Y and 10Y rates in each period, as a parallel move + a 2s10s move (bp)."""
    rows = []
    for name, (start, end) in STRESS_PERIODS.items():
        a, b = hist.loc[:start].iloc[-1], hist.loc[:end].iloc[-1]
        d2, d10 = (b["r2"] - a["r2"]) * 1e4, (b["r10"] - a["r10"]) * 1e4
        rows.append({"scenario": name, "move_2y_bp": d2, "move_10y_bp": d10,
                     "parallel_bp": (d2 + d10) / 2, "slope_bp": d10 - d2})
    return pd.DataFrame(rows)


# ---------- the desk model ----------

class DeskModel:
    """Closed-form value of the note on a curve given by its 2Y and 10Y rates."""

    def __init__(self, note, spread, slope_vol):
        self.note, self.spread, self.vol = note, spread, slope_vol
        self.k_low, self.k_high = note.strikes

    @staticmethod
    def zero(r2, r10, tau):
        return r2 + (r10 - r2) * np.clip((tau - 2) / 8, 0, 1)

    def expected_coupon(self, slope, theta):
        n = self.note
        call_spread = bachelier_call(slope, self.k_low, self.vol, theta) - bachelier_call(slope, self.k_high, self.vol, theta)
        return n.floor + n.participation * call_spread

    def value(self, r2, r10, t, fixed=None):
        """Value at t (years since launch) of the coupons paid after t and of the principal.

        fixed = {coupon number: rate} for the floating coupons already fixed."""
        n = self.note
        k = np.arange(int(np.floor(t + 1e-9)) + 1, n.maturity + 1)
        tau = k - t
        df = np.exp(-(self.zero(r2, r10, tau) + self.spread) * tau)
        coupons = self.expected_coupon(r10 - r2, np.maximum(k - 1 - t, 1e-6))
        for kk, rate in list(enumerate(n.fixed_rates, start=1)) + list((fixed or {}).items()):
            coupons[k == kk] = rate
        # "price to worst": the issuer picks the call date that is cheapest for it
        coupons_pv = np.cumsum(df * coupons)
        values = [coupons_pv[-1] + df[-1]] + [coupons_pv[i] + df[i] for i, kk in enumerate(k) if kk in n.call_dates]
        return float(min(values))

    def greeks(self, r2, r10, t, fixed=None, h=1e-4):
        """DV01 and Slope01 (per 1bp) and the second-order terms, by bump and revalue."""
        def v(level, slope):
            return self.value(r2 + level - slope / 2, r10 + level + slope / 2, t, fixed)
        v0 = v(0, 0)
        g = {"dv01": (v(h, 0) - v(-h, 0)) / 2, "slope01": (v(0, h) - v(0, -h)) / 2}
        H = 10 * h  # 10bp for the gammas
        g["gamma_level"] = (v(H, 0) - 2 * v0 + v(-H, 0)) / 100  # per bp^2
        g["gamma_slope"] = (v(0, H) - 2 * v0 + v(0, -H)) / 100
        g["cross"] = (v(H, H) - v(H, -H) - v(-H, H) + v(-H, -H)) / 400
        return v0, g


def swap_annuity(r2, r10, n):
    tau = np.arange(1, n + 1)
    return float(np.sum(np.exp(-DeskModel.zero(r2, r10, tau) * tau)))


def swap_greeks(r2, r10, n):
    """Receiver swap at par: value change per +1bp parallel and per +1bp 2s10s."""
    par = r2 if n == 2 else r10
    move = {"dv01": 1e-4, "slope01": -0.5e-4 if n == 2 else 0.5e-4}
    return {k: -m * swap_annuity(r2, r10, n) for k, m in move.items()} | {"par": par}


# ---------- one note through history ----------

def run_note(note, model, m, start):
    """Launch the note at month `start` of the monthly rates m and follow it to maturity or call.

    Returns the investor's cash flows, the call year, the monthly P&L explain of the note and the
    desk's monthly P&L (short the note, funded at its own rate) with three hedges rebalanced every
    month: none, DV01 only (10Y swap) and DV01 + slope (2Y and 10Y swaps).
    """
    r2, r10 = m["r2"].to_numpy(), m["r10"].to_numpy()
    fixed, flows, rows = {}, {}, []
    called = None
    for i in range(note.maturity * 12):
        t, a, b = i / 12, start + i, start + i + 1
        v0, g = model.greeks(r2[a], r10[a], t, fixed)

        # receiver swaps entered at par this month: notionals for each hedge
        s2, s10 = swap_greeks(r2[a], r10[a], 2), swap_greeks(r2[a], r10[a], 10)
        A = np.array([[s2["dv01"], s10["dv01"]], [s2["slope01"], s10["slope01"]]])
        hedges = {"none": (0.0, 0.0), "dv01": (0.0, g["dv01"] / s10["dv01"]),
                  "dv01_slope": tuple(np.linalg.solve(A, [g["dv01"], g["slope01"]]))}

        # one month later: coupon on the anniversary, fixing of the next coupon, issuer's call
        t1, k = (i + 1) / 12, (i + 1) // 12
        anniversary = (i + 1) % 12 == 0
        cash, theta_fixed = 0.0, dict(fixed)
        if anniversary:
            cash = note.fixed_rates[k - 1] if k <= len(note.fixed_rates) else fixed[k]
            cash += 1.0 if k == note.maturity else 0.0
            if len(note.fixed_rates) <= k < note.maturity:
                fixed[k + 1] = float(note.coupon(r10[b] - r2[b]))
                theta_fixed[k + 1] = float(note.coupon(r10[a] - r2[a]))  # same fixing, old rates
        last = anniversary and k == note.maturity
        alive = 0.0 if last else model.value(r2[b], r10[b], t1, fixed)
        if anniversary and k in note.call_dates and alive > 1.0:
            called = k
        redemption = 1.0 if called else 0.0
        v1 = 0.0 if called else alive

        # P&L explain of the note (investor view): carry + delta + slope + gamma + call + residual
        dl = ((r2[b] + r10[b]) - (r2[a] + r10[a])) / 2 * 1e4  # parallel move (bp)
        ds = ((r10[b] - r2[b]) - (r10[a] - r2[a])) * 1e4  # 2s10s move (bp)
        same_rates = 0.0 if last else model.value(r2[a], r10[a], t1, theta_fixed)
        actual = v1 + cash + redemption - v0
        explain = {"carry": same_rates + cash - v0, "delta": g["dv01"] * dl, "slope": g["slope01"] * ds,
                   "gamma": 0.5 * g["gamma_level"] * dl**2 + 0.5 * g["gamma_slope"] * ds**2 + g["cross"] * dl * ds,
                   "call": redemption - alive if called else 0.0}
        explain["residual"] = actual - sum(explain.values())

        # desk: short the note, cash funded at r2 + GS spread, plus the swaps (value + carry)
        swap2 = (s2["par"] - r2[b]) * swap_annuity(r2[b], r10[b], 2) + (s2["par"] - r2[a]) / 12
        swap10 = (s10["par"] - r10[b]) * swap_annuity(r2[b], r10[b], 10) + (s10["par"] - r2[a]) / 12
        desk = v0 * (r2[a] + model.spread) / 12 - actual
        row = {"date": m.index[b], "month": i + 1, "value": v1, "cash": cash + redemption, "note_pnl": actual}
        for name, (h2, h10) in hedges.items():
            row[f"desk_{name}"] = desk + h2 * swap2 + h10 * swap10
        rows.append(row | explain)
        if anniversary:
            flows[k] = cash + redemption
        if called:
            break
    return {"flows": flows, "called": called, "value_at_launch": model.value(r2[start], r10[start], 0.0),
            "months": pd.DataFrame(rows)}


def irr(flows, price=1.0):
    """Annual IRR of yearly cash flows {year: amount} for a given price."""
    def f(y):
        return sum(c / (1 + y) ** k for k, c in flows.items()) - price
    return brentq(f, -0.5, 0.5)


def backtest(note, model, hist, gs_spread, anchor=None):
    """The note launched at the end of every month that has a full 10-year history (2004-2016).

    anchor: if given, the rates of each note's life are shifted so that its 10Y rate at launch
    equals `anchor` (today's level). The 2s10s spread and the monthly moves stay historical.

    Returns one row per note for the investor (IRR vs a fixed-rate GS bond), one row per note for
    the desk (total P&L with each hedge, % of notional) and every month of every note (P&L explain).
    """
    m = monthly(hist)
    investor, desk, months = [], [], []
    for start in range(len(m) - note.maturity * 12):
        shift = 0.0 if anchor is None else anchor - m["r10"].iloc[start]
        res = run_note(note, model, m.assign(r2=m["r2"] + shift, r10=m["r10"] + shift), start)
        launch, mo = m.index[start], res["months"]
        floating = [c for k, c in res["flows"].items() if len(note.fixed_rates) < k and c < 1]
        investor.append({"launch": launch, "irr": irr(res["flows"]), "called": res["called"] or 0,
                         "avg_floating_coupon": np.mean(floating) if floating else np.nan,
                         "fixed_bond_yield": m["r10"].iloc[start] + shift + gs_spread,
                         "slope_at_launch": m["slope"].iloc[start], "model_value": res["value_at_launch"]})
        desk.append({"launch": launch, **{h: mo[f"desk_{h}"].sum() for h in ("none", "dv01", "dv01_slope")}})
        months.append(mo.assign(launch=launch))
    return pd.DataFrame(investor), pd.DataFrame(desk), pd.concat(months, ignore_index=True)

"""Greeks (bump and revalue), leverage solver, levers and the desk's swap hedge.

All sensitivities are changes in the note's price in bp of notional. The investor holds them;
the desk that sold the note holds the opposite.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import market as mk
from .g2pp import calibrate, price_callable, price_on_paths
from .market import Market
from .pricing import price_no_call

BUMPS = {
    "DV01 (rates +1bp)": lambda q, h: mk.parallel(q, h),
    "Slope01 (2s10s +1bp)": lambda q, h: mk.steepen(q, h),
    "Vega (all vols +1bp)": lambda q, h: mk.bump_vols(q, h),
    "Correlation (+1 pt)": lambda q, h: mk.bump_corr(q, h / 100),
    "Credit (GS spread +1bp)": lambda q, h: mk.bump_spread(q, h),
}


def price(note, q, callable_=False, n_paths=30_000, start=None):
    m = Market(q)
    if callable_:
        return price_callable(note, m, n_paths, start=start)["price"]
    return price_no_call(note, m)[0]


def greeks(note, q, callable_=False, n_paths=30_000, start=None, h=1.0):
    """Central differences. Use h = 5 for the callable price (Monte Carlo, same random numbers)."""
    out = {}
    for name, bump in BUMPS.items():
        up = price(note, bump(q, h), callable_, n_paths, start)
        down = price(note, bump(q, -h), callable_, n_paths, start)
        out[name] = (up - down) / (2 * h) * 1e4
    return out


def cross_gamma(note, q, h=10.0):
    def p(level, slope):
        return price(note, mk.steepen(mk.parallel(q, level), slope))
    return (p(h, h) - p(h, -h) - p(-h, h) + p(-h, -h)) / (4 * h * h) * 1e4


def key_rate_deltas(note, q):
    rows = [(int(T), (price(note, mk.key_rate(q, T, 1)) - price(note, mk.key_rate(q, T, -1))) / 2 * 1e4)
            for T in q["tenors"]]
    return pd.DataFrame(rows, columns=["tenor", "bp_of_notional"])


def vega_buckets(note, q):
    rows = []
    for T in sorted(q["vols"]["expiry"].unique()):
        for n in sorted(q["vols"]["tenor"].unique()):
            up, down = price(note, mk.bump_vols(q, 1, T, n)), price(note, mk.bump_vols(q, -1, T, n))
            rows.append((T, n, (up - down) / 2 * 1e4))
    return pd.DataFrame(rows, columns=["expiry", "tenor", "bp_of_notional"])


def shift_profile(note, q, shifts, kind="parallel", n_paths=30_000, start=None):
    """Full revaluation after parallel or 2s10s moves."""
    rows = []
    for s in shifts:
        m = Market(mk.parallel(q, s) if kind == "parallel" else mk.steepen(q, s))
        res = price_callable(note, m, n_paths, start=start)
        rows.append({"shift_bp": s, "non_callable": res["copula_no_call"], "callable": res["price"],
                     "p_called": sum(res["p_called"].values()), "expected_life": res["expected_life"]})
    return pd.DataFrame(rows)


# ---------- leverage solver and levers ----------

class Pricer:
    """Calibrate and simulate once, then price any variant of the note on the same paths."""

    def __init__(self, market, n_paths=30_000, model=None, start=None, elapsed=0.0):
        self.market = market
        self.model = model or calibrate(market, start)[0]
        self.paths = self.model.simulate(np.arange(1, 11) - elapsed, n_paths)

    def price(self, note):
        copula = price_no_call(note.no_call(), self.market)[0]
        if not note.call_dates:
            return copula
        res = price_on_paths(note, self.model, self.paths)
        return copula - (res["mc_no_call"] - res["mc_callable"])

    def solve(self, note, target, term="participation", lo=0.5, hi=30.0):
        """Value of `term` that gives the target price."""
        f = lambda v: self.price(note.with_(**{term: v})) - target  # noqa: E731
        return brentq(f, lo, hi, xtol=1e-4) if f(lo) * f(hi) < 0 else np.nan


MARKET_LEVERS = {
    "Vols +10%": lambda q: mk.scale_vols(q, 1.10),
    "Vols -10%": lambda q: mk.scale_vols(q, 0.90),
    "Correlation -5 pts": lambda q: mk.bump_corr(q, -0.05),
    "Correlation +5 pts": lambda q: mk.bump_corr(q, 0.05),
    "Curve +50bp": lambda q: mk.parallel(q, 50),
    "2s10s +25bp steeper": lambda q: mk.steepen(q, 25),
    "2s10s -25bp flatter": lambda q: mk.steepen(q, -25),
    "GS spread +25bp": lambda q: mk.bump_spread(q, 25),
}
STRUCTURE_LEVERS = {
    "No issuer call": lambda n: n.no_call(),
    "First call in 2029": lambda n: n.with_(call_dates=n.call_dates[1:]),
    "Floor 1.5%": lambda n: n.with_(floor=0.015),
    "Cap 7%": lambda n: n.with_(cap=0.07),
    "Fixed coupons 5%": lambda n: n.with_(fixed_rates=(0.05, 0.05)),
}


def levers(note, q, target, n_paths=30_000, start=None):
    """Fair participation for the target price, in the base case and after each change."""
    base = Pricer(Market(q), n_paths, start=start)
    rows = [("Base", base.solve(note, target))]
    for name, change in MARKET_LEVERS.items():
        rows.append((name, Pricer(Market(change(q)), n_paths, start=base.model.params).solve(note, target)))
    for name, change in STRUCTURE_LEVERS.items():
        rows.append((name, base.solve(change(note), target)))
    out = pd.DataFrame(rows, columns=["lever", "participation"])
    out["change"] = out["participation"] - out["participation"].iloc[0]
    return out


def vol_corr_grid(note, q, vol_scales, correlations):
    """Price without the call for each vol level x flat correlation."""
    grid = [[price_no_call(note.no_call(), Market(mk.flat_corr(q, c), vol_scale=v))[0] for c in correlations]
            for v in vol_scales]
    return pd.DataFrame(grid, index=pd.Index(vol_scales, name="vol_scale"),
                        columns=pd.Index(correlations, name="correlation"))


# ---------- the desk's hedge with 2Y and 10Y swaps ----------

def swap_value(q, n, fixed_rate):
    """Value today of a receiver swap (receive fixed_rate, pay 6M Euribor) on 1 of notional."""
    m = Market(q)
    return (fixed_rate - m.swap_rate(0.0, n)) * m.annuity(0.0, n)


def swap_hedge(greeks_note, q):
    """Notionals of 2Y and 10Y receiver swaps that cancel the desk's DV01 and Slope01.

    The desk is short the note, so it holds -greeks; the swaps must bring both to zero.
    """
    k2, k10 = Market(q).swap_rate(0, 2), Market(q).swap_rate(0, 10)
    sens = np.zeros((2, 2))
    for j, (n, k) in enumerate([(2, k2), (10, k10)]):
        for i, bump in enumerate([mk.parallel, mk.steepen]):
            sens[i, j] = (swap_value(bump(q, 1), n, k) - swap_value(bump(q, -1), n, k)) / 2 * 1e4
    desk = -np.array([greeks_note["DV01 (rates +1bp)"], greeks_note["Slope01 (2s10s +1bp)"]])
    notionals = np.linalg.solve(sens, -desk)
    return {"receive_2y": notionals[0], "receive_10y": notionals[1], "k2": k2, "k10": k10}


def hedge_pnl(note, q, hedge, levels, slopes, n_paths=30_000, start=None):
    """Desk P&L (short the note, long the swaps) after instant moves of the curve, in % of notional."""
    base_note = price_callable(note, Market(q), n_paths, start=start)["price"]
    unhedged, hedged = np.zeros((len(levels), len(slopes))), np.zeros((len(levels), len(slopes)))
    for i, lv in enumerate(levels):
        for j, sl in enumerate(slopes):
            qs = mk.steepen(mk.parallel(q, lv), sl)
            note_pnl = -(price_callable(note, Market(qs), n_paths, start=start)["price"] - base_note)
            swaps = hedge["receive_2y"] * swap_value(qs, 2, hedge["k2"]) + hedge["receive_10y"] * swap_value(qs, 10, hedge["k10"])
            unhedged[i, j], hedged[i, j] = note_pnl, note_pnl + swaps
    idx = pd.Index(levels, name="parallel_bp")
    cols = pd.Index(slopes, name="slope_bp")
    return pd.DataFrame(unhedged, idx, cols), pd.DataFrame(hedged, idx, cols)


# ---------- scenarios on the real note: history, correlation, GS funding ----------

def scenario_prices(note, q, scenarios, n_paths=30_000, start=None):
    """Callable and non-callable prices after each (parallel_bp, slope_bp) curve move."""
    rows = []
    for name, (level, slope) in scenarios.items():
        res = price_callable(note, Market(mk.steepen(mk.parallel(q, level), slope)), n_paths, start=start)
        rows.append({"scenario": name, "parallel_bp": level, "slope_bp": slope,
                     "non_callable": res["copula_no_call"], "callable": res["price"]})
    return pd.DataFrame(rows)


def correlation_profile(note, q, correlations, n_paths=30_000, start=None):
    """Price for each flat 10Y/2Y correlation (G2++ is recalibrated to each correlation)."""
    rows = []
    for c in correlations:
        res = price_callable(note, Market(mk.flat_corr(q, c)), n_paths, start=start)
        rows.append({"correlation": c, "non_callable": res["copula_no_call"], "callable": res["price"],
                     "p_called": sum(res["p_called"].values())})
    return pd.DataFrame(rows)


def margin_vs_spread(note, q, model, spreads, n_paths=30_000, issue_price=1.0):
    """Value of the note and the issuer's margin for several GS funding spreads.

    The spread only changes the discounting, so G2++ is not recalibrated."""
    rows = []
    for s in spreads:
        m = Market(mk.bump_spread(q, (s - q["issuer_spread"]) * 1e4))
        res = price_callable(note, m, n_paths, model=type(model)(model.params, m))
        rows.append({"gs_spread_bp": s * 1e4, "value": res["price"], "margin": issue_price - res["price"]})
    return pd.DataFrame(rows)


def path_irr(flows, price=1.0, n_iter=50):
    """IRR of each simulated path (rows = paths, columns = yearly cash flows), by Newton steps."""
    years = np.arange(1, flows.shape[1] + 1)
    y = np.full(flows.shape[0], 0.04)
    for _ in range(n_iter):
        disc = (1 + y[:, None]) ** -years
        f = (flows * disc).sum(axis=1) - price
        df = -(flows * years * disc / (1 + y[:, None])).sum(axis=1)
        y = y - f / df
    return y


def investor_scenarios(note, res):
    """Split the simulated paths into what can happen to the investor."""
    called, flows = res["called"], res["flows"]
    coupons = flows[:, len(note.fixed_rates):].copy()
    coupons[:, -1] -= 1.0  # remove the principal (only used for the paths never called)
    at_floor = np.isclose(coupons, note.floor).all(axis=1)
    first = note.call_dates[0]
    groups = {
        f"Called at the first date ({2026 + first})": called == first,
        f"Called later ({2027 + first}-{2026 + note.call_dates[-1]})": (called > first) & (called < note.maturity),
        "Never called, all coupons at the 2% floor": (called == note.maturity) & at_floor,
        "Never called, some coupons above 2%": (called == note.maturity) & ~at_floor,
    }
    return groups

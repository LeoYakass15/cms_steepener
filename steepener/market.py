"""EUR market on 29/10/2026 (Bloomberg): ESTR and Euribor 6M curves, swaption smiles (SABR), correlation.

All quotes live in a plain dict, so a scenario (rates +1bp, vols +1bp, ...) is just a modified
copy of that dict, and a new Market is built from it.
"""

import copy

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


DATE = "2026-10-29"
TENORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30], float)


def load_quotes(folder="data_bloomberg", correlation=0.89, issuer_spread=None, max_expiry=12):
    """Bloomberg quotes of 29/10/2026 in a plain dict.

    - ESTR OIS swaps (1-30Y) for discounting, EUSA swaps vs Euribor 6M (1-10Y) for projection.
      Missing tenors are interpolated; beyond 10Y Euribor = ESTR + the 10Y basis (23bp).
    - Swaption smiles of the 2Y and 10Y swaps (normal vols, -300bp to +300bp around ATM).
    - One flat 10Y/2Y correlation (no market quote exists, it comes from the ECB history).
    - GS spread: from the GS 2035 bond if not given.
    """
    curve = pd.read_csv(f"{folder}/eur_swap_curves_{DATE}.csv")
    ois = np.interp(TENORS, curve["tenor"], curve["estr_ois_pct"]) / 100
    eusa = curve.dropna()
    basis = np.interp(TENORS, eusa["tenor"], eusa["euribor6m_swap_pct"] - eusa["estr_ois_pct"]) / 100
    vols = pd.read_csv(f"{folder}/swaption_smiles_{DATE}.csv")
    vols = vols[vols["expiry"] <= max_expiry].reset_index(drop=True)
    return {
        "tenors": TENORS,
        "ois": ois,
        "swap": ois + basis,
        "vols": vols,  # normal vols in bp
        "corr_expiries": np.array([1.0, 10.0]),
        "corr": np.array([correlation, correlation]),
        "issuer_spread": gs_spread(folder) if issuer_spread is None else issuer_spread,
    }


def gs_spread(folder="data_bloomberg"):
    """GS credit spread = mid yield of the GS 5.35% 2035 note - USD SOFR swap of the same maturity."""
    d = pd.read_csv(f"{folder}/gs_credit_{DATE}.csv").set_index("item")["value"]
    bond_yield = (d["gs_bond_bid_yield_pct"] + d["gs_bond_ask_yield_pct"]) / 2
    swap = np.interp(d["gs_bond_maturity_years"], [7, 10], [d["usd_sofr_swap_7y_pct"], d["usd_sofr_swap_10y_pct"]])
    return (bond_yield - swap) / 100


def load_atm_cube(folder="data_bloomberg"):
    """Full ATM normal vol cube (bp): rows = option expiry (years), columns = swap tenor (years)."""
    cube = pd.read_csv(f"{folder}/swaption_atm_cube_{DATE}.csv", index_col="expiry")
    cube.columns = cube.columns.astype(int)
    return cube


# ---------- scenarios: each returns a bumped copy of the quotes ----------

def shift_curves(q, shift_bp):
    """Add shift_bp (a number or an array, one value per tenor) to both curves."""
    q = copy.deepcopy(q)
    q["ois"] = q["ois"] + np.asarray(shift_bp) / 1e4
    q["swap"] = q["swap"] + np.asarray(shift_bp) / 1e4
    return q


def parallel(q, bp):
    return shift_curves(q, np.full(len(q["tenors"]), bp, dtype=float))


def steepen(q, bp):
    """10Y - 2Y moves by bp: rotation around 6Y, flat below 2Y, linear beyond 10Y
    (so the forward 2s10s spreads steepen too)."""
    return shift_curves(q, bp * (np.maximum(q["tenors"], 2) - 6.0) / 8.0)


def key_rate(q, tenor, bp):
    return shift_curves(q, np.where(q["tenors"] == tenor, bp, 0.0))


def bump_vols(q, bp, expiry=None, tenor=None):
    q = copy.deepcopy(q)
    v = q["vols"]
    mask = np.ones(len(v), bool)
    if expiry is not None:
        mask &= v["expiry"].to_numpy() == expiry
    if tenor is not None:
        mask &= v["tenor"].to_numpy() == tenor
    v.loc[mask, "normal_vol_bp"] += bp
    return q


def scale_vols(q, factor):
    q = copy.deepcopy(q)
    q["vols"]["normal_vol_bp"] *= factor
    return q


def bump_corr(q, delta):
    q = copy.deepcopy(q)
    q["corr"] = np.clip(q["corr"] + delta, -0.99, 0.99)
    return q


def flat_corr(q, value):
    q = copy.deepcopy(q)
    q["corr"] = np.full_like(q["corr"], value)
    return q


def bump_spread(q, bp):
    q = copy.deepcopy(q)
    q["issuer_spread"] += bp / 1e4
    return q


# ---------- curves ----------

def log_df(t, nodes, zeros):
    """Log discount factor, linear between nodes (flat forwards), last forward extended."""
    t = np.asarray(t, float)
    x = np.concatenate([[0.0], nodes])
    y = np.concatenate([[0.0], -zeros * nodes])
    last_fwd = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return np.where(t > x[-1], y[-1] + last_fwd * (t - x[-1]), np.interp(t, x, y))


def float_leg(disc, proj, start, end):
    """PV of a 6M Euribor leg on [start, end]: sum P(t_i) * (Pf(t_i-1) / Pf(t_i) - 1)."""
    pay = np.arange(start + 0.5, end + 1e-9, 0.5)
    return float(np.sum(disc(pay) * (proj(pay - 0.5) / proj(pay) - 1.0)))


def bootstrap(tenors, rates, disc=None):
    """Zero rates at each tenor so that every par swap reprices.

    disc=None: ESTR OIS curve (annual fixed vs compounded ESTR: rate * annuity = 1 - P(T)).
    Otherwise: Euribor 6M curve (annual fixed vs 6M Euribor), discounted on `disc`.
    """
    zeros = np.full(len(tenors), rates[0])
    for k, (T, rate) in enumerate(zip(tenors, rates)):

        def error(z):
            zeros[k] = z
            curve = lambda t: np.exp(log_df(t, tenors[: k + 1], zeros[: k + 1]))  # noqa: E731
            if disc is None:
                return rate * curve(np.arange(1, T + 1)).sum() - (1.0 - curve(T))
            return rate * disc(np.arange(1, T + 1)).sum() - float_leg(disc, curve, 0.0, T)

        zeros[k] = brentq(error, -0.05, 0.20, xtol=1e-14)
    return zeros


# ---------- swaption smiles: Bachelier and SABR (beta = 0) ----------

def bachelier_call(F, K, vol, T):
    """Undiscounted call on a normally distributed rate."""
    std = np.maximum(vol * np.sqrt(T), 1e-12)
    d = (F - K) / std
    return (F - K) * norm.cdf(d) + std * norm.pdf(d)


def sabr_vol(F, K, T, alpha, rho, nu):
    """Hagan's normal SABR vol with beta = 0. It only depends on F - K."""
    zeta = nu / alpha * (np.asarray(F, float) - np.asarray(K, float))
    small = np.abs(zeta) < 1e-7
    z = np.where(small, 1.0, zeta)
    x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    ratio = np.where(small, 1 - 0.5 * rho * zeta, z / x)
    return alpha * ratio * (1 + (2 - 3 * rho**2) / 24 * nu**2 * T)


def fit_sabr(offsets, vols, T):
    """Fit (alpha, rho, nu) to normal vols quoted at strike offsets K - F."""
    atm = float(np.interp(0.0, offsets, vols))
    res = least_squares(lambda p: (sabr_vol(0.0, offsets, T, *p) - vols) * 1e4, x0=[atm, -0.1, 0.3],
                        bounds=([1e-5, -0.99, 1e-4], [0.1, 0.99, 3.0]))
    return res.x


class Market:
    """Curves, SABR parameters and correlation built from a quotes dict."""

    def __init__(self, q, vol_scale=1.0):
        self.quotes = q
        self.tenors = q["tenors"]
        self.ois_zeros = bootstrap(self.tenors, q["ois"])
        self.swap_zeros = bootstrap(self.tenors, q["swap"], disc=self.df)
        self.issuer_spread = q["issuer_spread"]
        self.vol_scale = vol_scale
        # one SABR fit per (expiry, swap tenor): sabr[tenor] = array (n_expiries, 3)
        v = q["vols"]
        self.expiries = np.sort(v["expiry"].unique()).astype(float)
        self.sabr = {}
        for tenor in sorted(v["tenor"].unique()):
            fits = []
            for T in self.expiries:
                s = v[(v["tenor"] == tenor) & (v["expiry"] == T)].sort_values("offset_bp")
                fits.append(fit_sabr(s["offset_bp"].to_numpy() / 1e4, s["normal_vol_bp"].to_numpy() / 1e4, T))
            self.sabr[int(tenor)] = np.array(fits)

    # curves
    def df(self, t):
        return np.exp(log_df(t, self.tenors, self.ois_zeros))

    def proj_df(self, t):
        return np.exp(log_df(t, self.tenors, self.swap_zeros))

    def zero(self, t, curve="ois"):
        t = np.maximum(np.asarray(t, float), 1e-8)
        return -log_df(t, self.tenors, self.ois_zeros if curve == "ois" else self.swap_zeros) / t

    def risky_df(self, t):
        """Discount factor for GS cash flows: ESTR + issuer spread."""
        return self.df(t) * np.exp(-self.issuer_spread * np.asarray(t, float))

    def annuity(self, T, n):
        return float(self.df(T + np.arange(1, n + 1)).sum())

    def swap_rate(self, T, n):
        """Forward par swap rate starting in T years, n years long."""
        return float_leg(self.df, self.proj_df, T, T + n) / self.annuity(T, n)

    # vols and correlation
    def sabr_params(self, T, tenor):
        p = self.sabr[tenor]
        return [np.interp(T, self.expiries, p[:, i]) for i in range(3)]

    def vol(self, T, tenor, F, K):
        return self.vol_scale * sabr_vol(F, K, T, *self.sabr_params(T, tenor))

    def atm_vol(self, T, tenor):
        return float(self.vol(T, tenor, 0.0, 0.0))

    def corr(self, T):
        return float(np.interp(T, self.quotes["corr_expiries"], self.quotes["corr"]))

"""The note without the call: CMS rates from the smile, Gaussian copula for the 10Y - 2Y spread.

Each floating coupon = 2% + 5 x [(S - 0.40%)+ - (S - 1.20%)+] with S = CMS 10Y - CMS 2Y,
so the note is a zero-coupon + fixed coupons + a floor strip + leveraged CMS spread call spreads.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm

from .market import bachelier_call


@dataclass(frozen=True)
class Steepener:
    """Goldman Sachs callable fixed-to-steepener, EUR, 2026-2036 (XS2061620741). Annual coupons."""

    maturity: int = 10
    fixed_rates: tuple = (0.06, 0.06)  # years 1 and 2
    participation: float = 5.0
    floor: float = 0.02
    cap: float = 0.06
    call_dates: tuple = (2, 3, 4, 5, 6, 7, 8, 9)  # issuer can redeem at par after these coupons
    elapsed: float = 0.0  # years since issue on the valuation date (29/10/2026 = 2 months = 2/12)

    def coupon(self, spread):
        return np.clip(self.participation * np.asarray(spread, float), self.floor, self.cap)

    @property
    def strikes(self):
        """Spread levels where the floor and the cap start to bind (0.40% and 1.20%)."""
        return self.floor / self.participation, self.cap / self.participation

    @property
    def pay_times(self):
        """Coupon dates 1 Sep 2027 ... 1 Sep 2036, in years from the valuation date."""
        return np.arange(1, self.maturity + 1, dtype=float) - self.elapsed

    @property
    def fixings(self):
        """Floating coupons are fixed at the start of the period (years 2 to 9), paid one year later."""
        return np.arange(len(self.fixed_rates), self.maturity, dtype=float) - self.elapsed

    @property
    def accrued(self):
        """Accrued interest of the first 6% coupon (the clean price = value - accrued)."""
        return self.fixed_rates[0] * self.elapsed

    def with_(self, **changes):
        return replace(self, **changes)

    def no_call(self):
        return replace(self, call_dates=())


# ---------- CMS rate distribution ----------

class CmsRate:
    """Distribution of the n-year swap rate fixing at T, under the measure of its payment date.

    1. The smile gives call prices C(K); their second derivative is the density of S(T) under
       the annuity measure (Breeden-Litzenberger).
    2. A coupon paid at T_pay needs the T_pay-forward measure. With the linear terminal swap rate
       model P(T, T_pay) / A(T) = a + b S, the density is just reweighted by (a + b S).
    The mean shift is the CMS convexity adjustment.
    """

    def __init__(self, market, T, n, T_pay, smile=True, width=7.0, n_points=1401):
        self.forward = market.swap_rate(T, n)
        atm = market.atm_vol(T, n)
        half = width * atm * np.sqrt(T)
        K = np.linspace(self.forward - half, self.forward + half, n_points)
        h = K[1] - K[0]
        vols = market.vol(T, n, self.forward, K) if smile else np.full_like(K, atm)
        calls = bachelier_call(self.forward, K, vols, T)

        pdf_annuity = np.zeros_like(K)
        pdf_annuity[1:-1] = (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / h**2
        pdf_annuity = np.maximum(pdf_annuity, 0)
        pdf_annuity /= np.trapezoid(pdf_annuity, K)
        # the grid cuts the fat SABR wings a little: shift it so the swap rate is a martingale
        K = K + self.forward - np.trapezoid(K * pdf_annuity, K)

        a = 1.0 / n
        b = (market.df(T_pay) / market.annuity(T, n) - a) / self.forward
        pdf = pdf_annuity * np.maximum(a + b * K, 0)
        pdf /= np.trapezoid(pdf, K)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * h)])

        self.grid, self.pdf_annuity, self.pdf, self.cdf = K, pdf_annuity, pdf, cdf / cdf[-1]
        self.mean = float(np.trapezoid(K * pdf, K))  # the CMS rate
        self.convexity = self.mean - self.forward

    def quantile(self, u):
        return np.interp(u, self.cdf, self.grid)


# ---------- CMS spread with a Gaussian copula ----------

Z, W = hermegauss(80)  # Gauss-Hermite nodes: smooth prices, no Monte Carlo noise
W = W / np.sqrt(2 * np.pi)


def spread_points(market, T, rho=None, smile=True, long_tenor=10, short_tenor=2):
    """CMS 10Y - CMS 2Y on a 2D quadrature grid, with the weights of each point."""
    rho = market.corr(T) if rho is None else rho
    cms_long = CmsRate(market, T, long_tenor, T + 1, smile)
    cms_short = CmsRate(market, T, short_tenor, T + 1, smile)
    z1, z2 = np.meshgrid(Z, Z, indexing="ij")
    z_short = rho * z1 + np.sqrt(1 - rho**2) * z2
    spread = cms_long.quantile(norm.cdf(z1)) - cms_short.quantile(norm.cdf(z_short))
    return spread.ravel(), np.outer(W, W).ravel()


def expected_coupon(note, market, T, model="copula"):
    """E[coupon fixed at T] under the payment measure, for four models."""
    k_low, k_high = note.strikes
    if model == "intrinsic":  # forwards, no vol
        return float(note.coupon(market.swap_rate(T, 10) - market.swap_rate(T, 2)))
    if model == "bachelier":  # normal spread, ATM vols, no smile
        rho = market.corr(T)
        fwd = CmsRate(market, T, 10, T + 1, smile=False).mean - CmsRate(market, T, 2, T + 1, smile=False).mean
        vl, vs = market.atm_vol(T, 10), market.atm_vol(T, 2)
        vol = np.sqrt(vl**2 + vs**2 - 2 * rho * vl * vs)
        call_spread = bachelier_call(fwd, k_low, vol, T) - bachelier_call(fwd, k_high, vol, T)
        return note.floor + note.participation * float(call_spread)
    s, w = spread_points(market, T, smile=(model == "copula"))
    call_spread = np.sum(w * (np.maximum(s - k_low, 0) - np.maximum(s - k_high, 0)))
    return note.floor + note.participation * float(call_spread)


def price_no_call(note, market, model="copula"):
    """Price of the note without the issuer's call, and its building blocks."""
    dfs = market.risky_df(note.pay_times)
    n_fixed = len(note.fixed_rates)
    coupons = np.array([expected_coupon(note, market, T, model) for T in note.fixings])
    blocks = {
        "zero_coupon": float(dfs[-1]),
        "fixed_coupons": float(np.sum(np.array(note.fixed_rates) * dfs[:n_fixed])),
        "floor_strip": float(note.floor * dfs[n_fixed:].sum()),
        "spread_call_spreads": float(np.sum((coupons - note.floor) * dfs[n_fixed:])),
    }
    table = pd.DataFrame({
        "fixing_year": note.fixings + note.elapsed,
        "forward_spread": [market.swap_rate(T, 10) - market.swap_rate(T, 2) for T in note.fixings],
        "expected_coupon": coupons,
    })
    return sum(blocks.values()), blocks, table

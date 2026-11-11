"""The issuer's call: G2++ two-factor rate model, Monte Carlo and Longstaff-Schwartz.

r(t) = x(t) + y(t) + phi(t),  dx = -a x dt + sigma dW1,  dy = -b y dt + eta dW2,  corr(dW1, dW2) = rho

Why two factors: with one factor the 2Y and 10Y rates move together and the steepener is
worth almost nothing. phi(t) is chosen so that today's ESTR curve is matched exactly; the
Euribor curve keeps a constant basis to ESTR.
"""

import numpy as np
from scipy.optimize import least_squares

from .pricing import price_no_call


def B(z, tau):
    return (1 - np.exp(-z * tau)) / z


class G2pp:
    def __init__(self, params, market):
        self.a, self.b, self.sigma, self.eta, self.rho = params
        self.params = tuple(params)
        self.market = market

    def V(self, t, T):
        """Variance of the integral of x + y between t and T (Brigo-Mercurio)."""
        a, b, s, e, r = self.params
        tau = np.asarray(T, float) - np.asarray(t, float)

        def one(z):
            return tau + 2 / z * np.exp(-z * tau) - 0.5 / z * np.exp(-2 * z * tau) - 1.5 / z

        cross = tau + (np.exp(-a * tau) - 1) / a + (np.exp(-b * tau) - 1) / b - (np.exp(-(a + b) * tau) - 1) / (a + b)
        return s**2 / a**2 * one(a) + e**2 / b**2 * one(b) + 2 * r * s * e / (a * b) * cross

    def bond(self, t, T, x, y):
        """Zero-coupon P(t, T) for each simulated state (x, y); T can be a list of maturities."""
        T = np.atleast_1d(np.asarray(T, float))
        P0 = self.market.df
        log_a = np.log(P0(T) / P0(t)) + 0.5 * (self.V(t, T) - self.V(0, T) + self.V(0, t))
        x, y = np.asarray(x, float)[..., None], np.asarray(y, float)[..., None]
        return np.exp(log_a - B(self.a, T - t) * x - B(self.b, T - t) * y)

    def basis(self, u):
        return self.market.proj_df(u) / self.market.df(u)

    def swap_rate(self, t, n, x, y):
        """n-year par swap rate at time t in each state (annual fixed vs 6M Euribor)."""
        fixed = t + np.arange(1, n + 1)
        pay = t + 0.5 * np.arange(1, 2 * n + 1)
        prev = np.concatenate([[t], pay[:-1]])
        p_pay = self.bond(t, pay, x, y)
        p_prev = np.concatenate([np.ones_like(p_pay[..., :1]), p_pay[..., :-1]], axis=-1)
        float_leg = np.sum(p_prev * self.basis(prev) / self.basis(pay) - p_pay, axis=-1)
        return float_leg / np.sum(self.bond(t, fixed, x, y), axis=-1)

    def gradient(self, T, n):
        """dS/dx and dS/dy at x = y = 0 (closed form, the float leg is linear in bond prices)."""
        fixed = T + np.arange(1, n + 1)
        pay = T + 0.5 * np.arange(1, 2 * n + 1)
        prev = np.concatenate([[T], pay[:-1]])
        p_fixed, p_pay = self.bond(T, fixed, 0.0, 0.0), self.bond(T, pay, 0.0, 0.0)
        p_prev = np.concatenate([[1.0], p_pay[:-1]])
        ratio = self.basis(prev) / self.basis(pay)
        annuity = p_fixed.sum()
        rate = np.sum(p_prev * ratio - p_pay) / annuity
        grad = []
        for z in (self.a, self.b):
            d_float = np.sum(-B(z, prev - T) * p_prev * ratio + B(z, pay - T) * p_pay)
            d_annuity = -np.sum(B(z, fixed - T) * p_fixed)
            grad.append((d_float - rate * d_annuity) / annuity)
        return np.array(grad)

    def cov(self, T):
        a, b, s, e, r = self.params
        vx = s**2 / (2 * a) * (1 - np.exp(-2 * a * T))
        vy = e**2 / (2 * b) * (1 - np.exp(-2 * b * T))
        cxy = r * s * e / (a + b) * (1 - np.exp(-(a + b) * T))
        return np.array([[vx, cxy], [cxy, vy]])

    def normal_vol(self, T, n):
        """ATM swaption normal vol, swap rate linearised in (x, y)."""
        g = self.gradient(T, n)
        return float(np.sqrt(g @ self.cov(T) @ g / T))

    def rate_corr(self, T, n1, n2):
        """Terminal correlation of two swap rates fixing at T."""
        g1, g2, c = self.gradient(T, n1), self.gradient(T, n2), self.cov(T)
        return float(g1 @ c @ g2 / np.sqrt((g1 @ c @ g1) * (g2 @ c @ g2)))

    def simulate(self, dates, n_paths=40_000, steps_per_year=12, seed=42):
        """Exact Gaussian steps for x and y (antithetic), trapezoid for the integral of r.

        `dates` must sit on the monthly grid (coupon dates are 10, 22, 34... months away).
        Returns x, y and the discount factor D(0, t) at t = 0 and at each date.
        """
        a, b, s, e, r = self.params
        dt = 1 / steps_per_year
        ea, eb = np.exp(-a * dt), np.exp(-b * dt)
        sx = s * np.sqrt((1 - ea**2) / (2 * a))
        sy = e * np.sqrt((1 - eb**2) / (2 * b))
        c = r * s * e * (1 - np.exp(-(a + b) * dt)) / (a + b) / (sx * sy)

        steps = set(np.round(np.asarray(dates) * steps_per_year).astype(int))
        rng = np.random.default_rng(seed)
        half = n_paths // 2
        x, y, integral = np.zeros(2 * half), np.zeros(2 * half), np.zeros(2 * half)
        xs, ys, ints = [x], [y], [integral]
        for k in range(1, max(steps) + 1):
            z1, z2 = rng.standard_normal((2, half))
            z1, z2 = np.concatenate([z1, -z1]), np.concatenate([z2, -z2])
            x_new = x * ea + sx * z1
            y_new = y * eb + sy * (c * z1 + np.sqrt(1 - c**2) * z2)
            integral = integral + 0.5 * dt * (x + y + x_new + y_new)
            x, y = x_new, y_new
            if k in steps:
                xs.append(x)
                ys.append(y)
                ints.append(integral)
        t = np.concatenate([[0.0], np.sort(list(steps)) / steps_per_year])
        D = self.market.df(t) * np.exp(-0.5 * self.V(0.0, t)) * np.exp(-np.stack(ints, axis=1))
        return np.stack(xs, axis=1), np.stack(ys, axis=1), D


def calibrate(market, start=None, corr_weight=100.0):
    """Fit G2++ to the 12 ATM vols (2Y and 10Y swaps) and to the CMS correlation at 2, 5 and 9 years.

    Fitted to vols only, G2++ goes to rho = -1 (2Y and 10Y perfectly correlated), so the
    correlation has to be a target too. `start` = previous parameters (warm start for scenarios).
    """
    targets = [(T, n, market.atm_vol(T, n)) for T in (1, 2, 3, 5, 7, 10) for n in (2, 10)]
    corr_targets = [(T, market.corr(T)) for T in (2, 5, 9)]

    def errors(p):
        m = G2pp(p, market)
        e = [(m.normal_vol(T, n) - v) * 1e4 for T, n, v in targets]
        return np.array(e + [(m.rate_corr(T, 10, 2) - c) * corr_weight for T, c in corr_targets])

    lower, upper = [0.005, 0.005, 1e-4, 1e-4, -0.999], [3.0, 3.0, 0.06, 0.06, 0.999]
    starts = [(0.6, 0.04, 0.012, 0.009, -0.7), (0.3, 0.06, 0.009, 0.007, -0.4)]
    if start is not None:
        starts = [tuple(np.clip(start, np.add(lower, 1e-9), np.subtract(upper, 1e-9)))]
    fits = [least_squares(errors, s, bounds=(lower, upper), x_scale=[0.1, 0.01, 0.001, 0.001, 0.1]) for s in starts]
    best = min(fits, key=lambda f: f.cost)
    model = G2pp(best.x, market)
    vol_errors = np.array([(model.normal_vol(T, n) - v) * 1e4 for T, n, v in targets])
    corr_errors = {T: model.rate_corr(T, 10, 2) - c for T, c in corr_targets}
    return model, float(np.sqrt(np.mean(vol_errors**2))), corr_errors


def cash_flows(note, model, x, y):
    """Coupons and principal on each path at the 10 coupon dates, plus the 10Y and 2Y rates at each fixing.

    Column k of x, y is coupon date k (column 0 = today), so coupon k + 1 is fixed in column k.
    """
    flows = np.zeros((x.shape[0], note.maturity))
    flows[:, : len(note.fixed_rates)] = note.fixed_rates
    rates = {}
    for k, T in zip(range(len(note.fixed_rates), note.maturity), note.fixings):
        s10 = model.swap_rate(T, 10, x[:, k], y[:, k])
        s2 = model.swap_rate(T, 2, x[:, k], y[:, k])
        rates[k] = (s10, s2)
        flows[:, k] = note.coupon(s10 - s2)  # fixed at year k, paid at year k + 1
    flows[:, -1] += 1.0
    return flows, rates


def longstaff_schwartz(note, flows, rates, disc):
    """Backward over the call dates: GS calls when the expected value of what it still owes
    (regressed on the 10Y rate, the 2Y rate and the coupon just fixed) is above par."""
    flows = flows.copy()
    called = np.full(flows.shape[0], note.maturity)
    for t in sorted(note.call_dates, reverse=True):
        future = (flows[:, t:] * disc[:, t:] / disc[:, [t - 1]]).sum(axis=1)
        s10, s2 = rates[t]
        c = note.coupon(s10 - s2)
        X = np.column_stack([np.ones_like(s10), s10, s2, s10**2, s2**2, s10 * s2, c])
        beta = np.linalg.lstsq(X, future, rcond=None)[0]
        call = X @ beta > 1.0
        flows[call, t:] = 0.0
        flows[call, t - 1] += 1.0
        called[call] = t
    return flows, called


def price_on_paths(note, model, paths):
    """Monte Carlo value with and without the call on given paths (same random numbers)."""
    x, y, D = paths
    disc = D[:, 1:] * np.exp(-model.market.issuer_spread * note.pay_times)
    flows, rates = cash_flows(note, model, x, y)
    no_call = (flows * disc).sum(axis=1).mean()
    if note.call_dates:
        flows, called = longstaff_schwartz(note, flows, rates, disc)
    else:
        called = np.full(x.shape[0], note.maturity)
    pv = (flows * disc).sum(axis=1)
    spreads = np.column_stack([s10 - s2 for s10, s2 in rates.values()])
    return {"mc_no_call": float(no_call), "mc_callable": float(pv.mean()), "pv": pv, "called": called,
            "spreads": spreads, "flows": flows}


def price_callable(note, market, n_paths=40_000, seed=42, model=None, start=None, paths=None):
    """Final price = copula price without the call - value of the call in G2++ (the usual desk split)."""
    if model is None:
        model = calibrate(market, start)[0]
    if paths is None:
        paths = model.simulate(note.pay_times, n_paths, seed=seed)
    res = price_on_paths(note, model, paths)
    res["copula_no_call"] = price_no_call(note.no_call(), market)[0]
    res["call_value"] = res["mc_no_call"] - res["mc_callable"]
    res["price"] = res["copula_no_call"] - res["call_value"]
    half = len(res["pv"]) // 2
    res["std_error"] = float(np.std(0.5 * (res["pv"][:half] + res["pv"][half:])) / np.sqrt(half))
    res["p_called"] = {t: float(np.mean(res["called"] == t)) for t in note.call_dates}
    res["expected_life"] = float(res["called"].mean())
    res["model"] = model
    return res

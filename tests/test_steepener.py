import os

import numpy as np
import pytest

from steepener import history, risks
from steepener.g2pp import calibrate, price_callable
from steepener.market import Market, bachelier_call, fit_sabr, load_quotes, sabr_vol
from steepener.pricing import CmsRate, Steepener, price_no_call, spread_points

# The Bloomberg quotes are not in the repo (licence): these tests run only where they are.
HAS_BLOOMBERG = os.path.exists("data_bloomberg")
bloomberg = pytest.mark.skipif(not HAS_BLOOMBERG, reason="needs the Bloomberg files in data_bloomberg/")
Q = load_quotes() if HAS_BLOOMBERG else None
MARKET = Market(Q) if HAS_BLOOMBERG else None
NOTE = Steepener()


def test_coupon_matches_term_sheet_examples():
    # Final Terms examples: spread -1%, 0%, 0.5%, >= 1.2% -> 2%, 2%, 2.5%, 6%
    assert np.allclose(NOTE.coupon([-0.01, 0.0, 0.005, 0.012]), [0.02, 0.02, 0.025, 0.06])


@bloomberg
def test_curves_reprice_the_quotes():
    for T, swap in zip(Q["tenors"], Q["swap"]):
        assert abs(MARKET.swap_rate(0.0, int(T)) - swap) < 1e-10


def test_sabr_fit_recovers_parameters():
    offsets = np.array([-150, -100, -50, -25, 0, 25, 50, 100, 150]) / 1e4
    vols = sabr_vol(0.0, offsets, 5.0, 0.0075, -0.25, 0.35)
    assert np.allclose(fit_sabr(offsets, vols, 5.0), [0.0075, -0.25, 0.35], atol=[1e-6, 1e-3, 1e-3])


@bloomberg
def test_cms_convexity():
    cms = CmsRate(MARKET, 9.0, 10, 10.0)
    assert abs(np.trapezoid(cms.grid * cms.pdf_annuity, cms.grid) - cms.forward) < 1e-8  # martingale
    assert 0 < CmsRate(MARKET, 5.0, 2, 6.0).convexity < CmsRate(MARKET, 5.0, 10, 6.0).convexity
    assert abs(CmsRate(Market(Q, vol_scale=0.01), 9.0, 10, 10.0).convexity) < 1e-6  # no vol, no convexity


@bloomberg
def test_copula_without_smile_matches_bachelier():
    s, w = spread_points(MARKET, 5.0, smile=False)
    rho = MARKET.corr(5.0)
    fwd = CmsRate(MARKET, 5.0, 10, 6.0, smile=False).mean - CmsRate(MARKET, 5.0, 2, 6.0, smile=False).mean
    vl, vs = MARKET.atm_vol(5.0, 10), MARKET.atm_vol(5.0, 2)
    vol = np.sqrt(vl**2 + vs**2 - 2 * rho * vl * vs)
    assert abs(np.sum(w * np.maximum(s - 0.004, 0)) - bachelier_call(fwd, 0.004, vol, 5.0)) < 2e-5


@bloomberg
def test_g2pp_matches_the_curve_and_the_call_lowers_the_price():
    model, rms, _ = calibrate(MARKET)
    assert rms < 4.0
    x, y, D = model.simulate(np.arange(1, 11), 20_000, seed=5)
    assert np.all(np.abs(D.mean(axis=0)[1:] / MARKET.df(np.arange(1, 11)) - 1) < 2e-3)
    res = price_callable(NOTE, MARKET, 20_000, model=model)
    assert res["mc_callable"] < res["mc_no_call"]
    blocks = price_no_call(NOTE, MARKET)[1]
    assert abs(sum(blocks.values()) - res["call_value"] - res["price"]) < 1e-12


@bloomberg
def test_solver_and_hedge():
    pricer = risks.Pricer(MARKET, 10_000)
    target = pricer.price(NOTE.with_(participation=7.0))
    assert abs(pricer.solve(NOTE, target) - 7.0) < 1e-3
    g = risks.greeks(NOTE.no_call(), Q)
    hedge = risks.swap_hedge(g, Q)
    # after the hedge, a 1bp parallel move barely moves the desk's P&L
    q_up = risks.mk.parallel(Q, 1)
    desk = -(price_no_call(NOTE, Market(q_up))[0] - price_no_call(NOTE, MARKET)[0])
    swaps = hedge["receive_2y"] * risks.swap_value(q_up, 2, hedge["k2"]) + hedge["receive_10y"] * risks.swap_value(q_up, 10, hedge["k10"])
    assert abs(desk + swaps) < 2e-6


@bloomberg
def test_the_note_on_29_october_rolls_forward():
    live = NOTE.with_(elapsed=2 / 12)
    assert np.allclose(live.pay_times[:2], [10 / 12, 22 / 12])
    assert abs(live.accrued - 0.01) < 1e-12
    # two months closer to the coupons: worth a bit more on the same market
    assert price_no_call(live, MARKET)[0] > price_no_call(NOTE, MARKET)[0]


def test_irr_of_a_par_bond_is_its_coupon():
    flows = {k: 0.05 for k in range(1, 11)}
    flows[10] += 1.0
    assert abs(history.irr(flows) - 0.05) < 1e-10
    assert abs(risks.path_irr(np.array([list(flows.values())]))[0] - 0.05) < 1e-10


def test_desk_model_and_pnl_explain():
    model = history.DeskModel(NOTE.no_call(), 0.01, 1e-8)  # no vol: coupons from today's 2s10s
    r2, r10 = 0.02, 0.025
    tau = np.arange(1, 11)
    df = np.exp(-(model.zero(r2, r10, tau) + 0.01) * tau)
    coupons = np.r_[0.06, 0.06, np.full(8, NOTE.coupon(r10 - r2))]
    assert abs(model.value(r2, r10, 0.0) - (np.sum(df * coupons) + df[-1])) < 1e-12
    # the Greeks explain the monthly P&L of the real note through history
    hist = history.load_history()
    m = history.monthly(hist)
    months = history.run_note(NOTE, history.DeskModel(NOTE, 0.0168, 0.005), m, 100)["months"]
    parts = ["carry", "delta", "slope", "gamma", "call", "residual"]
    assert np.allclose(months[parts].sum(axis=1), months["note_pnl"])
    normal = months["call"] == 0  # the month of the call is a jump, not a Greek
    assert months.loc[normal, "residual"].abs().mean() < 0.2 * months.loc[normal, "note_pnl"].abs().mean()

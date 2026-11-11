"""Every study on the Goldman Sachs steepener, on Bloomberg data of 29/10/2026 and ECB history.

python run_analysis.py          (about 10 minutes)
python run_analysis.py --quick  (fewer paths, for a fast check)

Charts and tables go to results/. The raw Bloomberg quotes are not written there (licence).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from steepener import history, plots, risks
from steepener import market as mk
from steepener.g2pp import calibrate, price_callable
from steepener.market import Market, load_atm_cube, load_quotes
from steepener.pricing import CmsRate, Steepener, price_no_call

QUICK = "--quick" in sys.argv
N_MAIN = 20_000 if QUICK else 100_000  # paths for the main price
N_STUDY = 6_000 if QUICK else 30_000  # paths for Greeks, scenarios and solver
OUT = "results"
os.makedirs(OUT, exist_ok=True)
start_time = time.time()
summary = {}


def step(msg):
    print(f"[{time.time() - start_time:6.1f}s] {msg}")


def save(df, name, **kw):
    df.to_csv(f"{OUT}/{name}.csv", **kw)


# ---------- A. Market ----------
step("ECB history and correlation")
hist = history.load_history()
rho = history.correlation_estimate(hist, years=5)
rolling = history.rolling_correlation(hist)
print(f"    10Y/2Y correlation of weekly changes, last 5 years: {rho:.3f}")

q = load_quotes(correlation=rho)
market = Market(q)
gs_spread = q["issuer_spread"]
note = Steepener(elapsed=2 / 12)  # the note on 29/10/2026, two months after issue
new_note = Steepener()  # the same note if it were launched today (structuring questions)
print(f"    GS spread from the GS 5.35% 2035 bond: {gs_spread * 1e4:.0f}bp")
summary.update(correlation=rho, gs_spread_bp=gs_spread * 1e4)

step("Curves, cube, smiles and CMS convexity")
rows = []
for T in note.fixings:
    for n in (2, 10):
        cms = CmsRate(market, T, n, T + 1)
        rows.append({"expiry": T, "tenor": n, "forward": cms.forward, "cms": cms.mean,
                     "convexity_bp": cms.convexity * 1e4})
cms_rows = pd.DataFrame(rows)
save(cms_rows, "cms_convexity", index=False)
cube = load_atm_cube()
plots.curves(market, note, cms_rows, f"{OUT}/01_curves.png")
plots.atm_cube(cube, f"{OUT}/02_atm_cube.png")
plots.smiles(market, f"{OUT}/03_swaption_smiles.png")
plots.sabr_errors(market, f"{OUT}/04_sabr_fit_errors.png")
plots.surfaces_3d_html(market, f"{OUT}/vol_surfaces_3d.html")

step("G2++ calibration")
model, rms, corr_errors = calibrate(market)
params = model.params
print("    a, b, sigma, eta, rho =", np.round(params, 4), f"| RMS error on the 12 vols {rms:.2f}bp")
cube_err = plots.g2pp_cube_errors(model, cube, f"{OUT}/05_g2pp_cube_errors.png")
save(cube_err.round(2), "g2pp_cube_errors_bp")
save(pd.DataFrame([dict(zip(["a", "b", "sigma", "eta", "rho"], params), rms_vol_error_bp=rms,
                        rms_whole_cube_bp=float(np.sqrt((cube_err**2).to_numpy().mean())),
                        **{f"corr_error_{T}y": e for T, e in corr_errors.items()})]), "g2pp_calibration", index=False)
plots.convexity(cms_rows.assign(expiry=cms_rows["expiry"] + note.elapsed), f"{OUT}/06_cms_convexity.png")

# ---------- B. Pricing on 29/10/2026 ----------
step("Pricing")
res = price_callable(note, market, N_MAIN, model=model)
comparison = {
    "Intrinsic\n(forwards, no vol)": price_no_call(note, market, "intrinsic")[0],
    "Bachelier\nATM vols": price_no_call(note, market, "bachelier")[0],
    "Copula\nno smile": price_no_call(note, market, "copula_no_smile")[0],
    "Copula\nSABR smile": res["copula_no_call"],
    "G2++ MC\nno call": res["mc_no_call"],
    "G2++ MC\nwith call": res["mc_callable"],
    "Hybrid: copula\n- G2++ call": res["price"],
}
clean = res["price"] - note.accrued
print(f"    without call {res['copula_no_call']:.2%} | call {res['call_value']:.2%} | note {res['price']:.2%}"
      f" (+/- {1.96 * res['std_error']:.2%}), clean {clean:.2%} | P(called) {sum(res['p_called'].values()):.0%},"
      f" expected life {res['expected_life']:.1f}y")
save(pd.Series({k.replace("\n", " "): v for k, v in comparison.items()}, name="value"), "model_comparison")
plots.model_comparison(comparison, f"{OUT}/07_model_comparison.png")
_, blocks, coupons = price_no_call(note, market)
save(coupons, "expected_coupons", index=False)
labels = ["Zero-coupon bond\n(GS credit)", "Fixed coupons\n2 x 6%", "Floor strip\n2% x 8 years",
          "5 x CMS spread\ncall spread 0.40%/1.20%"]
blocks = dict(zip(labels, blocks.values()))
blocks["Issuer's call\n(sold to GS)"] = -res["call_value"]
save(pd.Series({k.replace("\n", " "): v for k, v in blocks.items()}, name="value"), "decomposition")
plots.waterfall(blocks, res["price"], f"{OUT}/08_decomposition_waterfall.png")
summary.update(price=res["price"], clean_price=clean, copula_no_call=res["copula_no_call"],
               call_value=res["call_value"], std_error=res["std_error"],
               p_called=sum(res["p_called"].values()), expected_life=res["expected_life"])

step("Mark-to-market since issue (1 September 2026)")
issue_date, today = "2026-09-01", "2026-10-29"
d2 = (hist.loc[today, "r2"] - hist.loc[issue_date, "r2"]) * 1e4
d10 = (hist.loc[today, "r10"] - hist.loc[issue_date, "r10"]) * 1e4
level, slope = (d2 + d10) / 2, d10 - d2
q_issue = mk.steepen(mk.parallel(q, -level), -slope)  # today's market with the moves taken back
mtm = {}
for name, qq, n in [("issue", q_issue, new_note), ("carry", q_issue, note),
                    ("level", mk.parallel(q_issue, level), note), ("today", q, note)]:
    mtm[name] = price_callable(n, Market(qq), N_STUDY, start=params)["price"]
steps = [("Paid at issue", 1.0, "total"), ("Issuer's margin\nand fees", mtm["issue"] - 1.0, "step"),
         ("Value at issue\n(model)", mtm["issue"], "total"),
         ("Time: 2 months\nof carry", mtm["carry"] - mtm["issue"], "step"),
         (f"Rates level\n{level:+.0f}bp", mtm["level"] - mtm["carry"], "step"),
         (f"2s10s\n{slope:+.0f}bp", mtm["today"] - mtm["level"], "step"),
         ("Value on 29/10\n(incl. accrued)", mtm["today"], "total"),
         ("Accrued\ncoupon", -note.accrued, "step"), ("Clean price\n29/10", mtm["today"] - note.accrued, "total")]
plots.steps_waterfall(steps, f"{OUT}/09_mark_to_market.png",
                      "Mark-to-market since issue: what the investor's note is worth two months later")
save(pd.DataFrame(steps, columns=["step", "value", "kind"]).replace("\n", " ", regex=True), "mark_to_market", index=False)
print(f"    2Y {d2:+.1f}bp, 10Y {d10:+.1f}bp | value at issue {mtm['issue']:.2%} -> today {mtm['today']:.2%}")

# ---------- C. Correlation ----------
step("Correlation")
plots.spread_distribution(market, note, f"{OUT}/11_spread_distribution.png")
grid = risks.vol_corr_grid(note, q, [0.7, 0.85, 1.0, 1.15, 1.3], [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
save(grid, "price_vol_correlation")
plots.correlation_heatmap(grid, f"{OUT}/12_price_vol_correlation.png")
corrs = [0.6, 0.7, 0.8, 0.9, 0.95] if QUICK else [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
profile = risks.correlation_profile(note, q, corrs, N_STUDY, start=params)
save(profile, "price_vs_correlation", index=False)
save(rolling.rename("rolling_1y_correlation"), "historical_correlation")
plots.historical_correlation(rolling, rho, profile, f"{OUT}/10_historical_correlation.png")

# ---------- D. Investor view ----------
step("Investor IRR and call profile (note launched today at 100%)")
new_res = price_callable(new_note, market, N_MAIN, model=model)
irr = risks.path_irr(new_res["flows"])
groups = risks.investor_scenarios(new_note, new_res)
bond_yield = market.swap_rate(0, 10) + gs_spread
plots.investor_irr(irr, groups, bond_yield, f"{OUT}/13_investor_irr.png")
save(pd.DataFrame([{"scenario": k, "probability": m.mean(), "mean_irr": irr[m].mean() if m.any() else np.nan}
                   for k, m in groups.items()]), "investor_scenarios", index=False)
summary.update(irr_mean=irr.mean(), irr_median=float(np.median(irr)), gs_fixed_bond_yield=bond_yield,
               p_irr_above_bond=float(np.mean(irr > bond_yield)), value_if_launched_today=new_res["price"])
print(f"    IRR mean {irr.mean():.2%}, median {np.median(irr):.2%} vs GS fixed bond {bond_yield:.2%}"
      f" (P(note better) {np.mean(irr > bond_yield):.0%})")
plots.call_profile(res["p_called"], res["expected_life"], res["spreads"], res["called"], note,
                   f"{OUT}/14_call_profile.png")
save(pd.Series(res["p_called"], name="probability").rename_axis("call_year"), "call_probabilities")

# ---------- E. Structuring ----------
step("Leverage solver, levers and GS margin")
# levers: keep the price of the term-sheet note (5x) and see how much leverage each change allows
pricer = risks.Pricer(market, N_STUDY, model=model)
target = pricer.price(new_note)
lev = risks.levers(new_note, q, target, N_STUDY, start=params)
save(lev, "participation_levers", index=False)
plots.levers(lev, f"{OUT}/15_participation_levers.png", target)
spreads = np.arange(0, 201, 25) / 1e4
margins = risks.margin_vs_spread(new_note, q, model, spreads, N_STUDY)
save(margins, "margin_vs_gs_spread", index=False)
plots.margin_vs_funding(margins, gs_spread * 1e4, f"{OUT}/16_margin_vs_funding.png")
# GS funding spread at which the note launched today would be worth 98% and 100% (value falls with the spread)
spread_98 = np.interp(-0.98, -margins["value"], margins["gs_spread_bp"])
spread_100 = np.interp(-1.00, -margins["value"], margins["gs_spread_bp"])
print(f"    value today {target:.2%} at {gs_spread * 1e4:.0f}bp | worth 98% at {spread_98:.0f}bp, 100% at {spread_100:.0f}bp")
summary.update(levers_target_price=target, spread_for_98=spread_98, spread_for_100=spread_100)

# ---------- F. Desk risks ----------
step("Greeks")
g_no_call = risks.greeks(note.no_call(), q)
g_callable = risks.greeks(note, q, callable_=True, n_paths=N_STUDY, start=params, h=5.0)
greeks = pd.DataFrame({"non_callable": g_no_call, "callable": g_callable}).rename_axis("risk")
greeks.loc["Cross gamma level x slope (per bp x bp)"] = [risks.cross_gamma(note.no_call(), q), np.nan]
save(greeks, "greeks")
print(greeks.round(3).to_string())
kr = risks.key_rate_deltas(note.no_call(), q)
vb = risks.vega_buckets(note.no_call(), q)
vb = vb[(vb["expiry"] >= 1) & (vb["expiry"] <= 10)]
save(kr, "key_rate_deltas", index=False)
save(vb, "vega_buckets", index=False)
plots.key_rates_and_vega(kr, vb, f"{OUT}/17_key_rates_and_vega.png")

step("Scenario profiles and historical stress tests")
par = risks.shift_profile(note, q, np.arange(-150, 151, 50 if QUICK else 25), "parallel", N_STUDY, params)
slo = risks.shift_profile(note, q, np.arange(-100, 101, 50 if QUICK else 20), "slope", N_STUDY, params)
save(par, "profile_parallel", index=False)
save(slo, "profile_slope", index=False)
plots.shift_profiles(par, slo, f"{OUT}/18_level_slope_profiles.png")
moves = history.stress_moves(hist)
stress = risks.scenario_prices(note, q, {r.scenario: (r.parallel_bp, r.slope_bp) for r in moves.itertuples()},
                               N_STUDY, start=params)
stress = moves.merge(stress[["scenario", "non_callable", "callable"]])
base = risks.scenario_prices(note, q, {"base": (0.0, 0.0)}, N_STUDY, start=params).iloc[0]
save(stress.assign(pnl_callable=stress["callable"] - base["callable"],
                   pnl_non_callable=stress["non_callable"] - base["non_callable"]), "historical_stress", index=False)
plots.historical_stress(stress, base, f"{OUT}/19_historical_stress.png")

step("Desk hedge with 2Y and 10Y swaps")
hedge = risks.swap_hedge(g_callable, q)
print(f"    receive {hedge['receive_2y']:+.3f} x 2Y swap and {hedge['receive_10y']:+.3f} x 10Y swap per 1 of notional")
shocks = [-50, -25, 0, 25, 50]
unhedged, hedged = risks.hedge_pnl(note, q, hedge, shocks, shocks, N_STUDY, start=params)
save(unhedged, "hedge_pnl_unhedged")
save(hedged, "hedge_pnl_hedged")
plots.hedge_pnl(unhedged, hedged, hedge, f"{OUT}/20_hedge_pnl.png")
summary.update(hedge_receive_2y=hedge["receive_2y"], hedge_receive_10y=hedge["receive_10y"])

# ---------- G. Back-tests on the ECB history ----------
step("Back-tests 2004-2026 (desk model, rates re-based to today's 10Y level)")
anchor = hist.loc[today, "r10"]
slope_vol = market.atm_vol(5, 10) ** 2 + market.atm_vol(5, 2) ** 2 - 2 * rho * market.atm_vol(5, 10) * market.atm_vol(5, 2)
desk_model = history.DeskModel(new_note, gs_spread, np.sqrt(slope_vol))
print(f"    2s10s normal vol {np.sqrt(slope_vol) * 1e4:.0f}bp (5Y swaptions, rho {rho:.2f})")
plots.coupon_history(hist, new_note, f"{OUT}/21_coupon_history.png")
inv, desk, months = history.backtest(new_note, desk_model, hist, gs_spread, anchor=anchor)
save(inv, "backtest_investor", index=False)
save(desk, "backtest_desk_pnl", index=False)
plots.investor_backtest(inv, f"{OUT}/22_investor_backtest.png")
plots.desk_hedge_backtest(desk, f"{OUT}/23_desk_hedge_backtest.png")
lives = months.groupby("launch")["month"].max()
example_launch = lives.idxmax() if (lives == lives.max()).sum() == 1 else lives[lives == lives.max()].index[-1]
r2 = plots.pnl_explain(months[months["launch"] == example_launch], months, f"{OUT}/24_pnl_explain.png")
parts = list(plots.EXPLAIN_COLORS)
explain_summary = pd.DataFrame({"std_per_month": months[parts + ["note_pnl"]].std(),
                                "sum_of_abs": months[parts + ["note_pnl"]].abs().sum()})
save(explain_summary, "pnl_explain_summary")
summary.update(backtest_irr_mean=inv["irr"].mean(), backtest_called_after_2y=float(np.mean(inv["called"] == 2)),
               backtest_never_called=float(np.mean(inv["called"] == 0)),
               desk_std_unhedged=desk["none"].std(), desk_std_dv01=desk["dv01"].std(),
               desk_std_dv01_slope=desk["dv01_slope"].std(), pnl_explain_r2=r2)
print(f"    investor IRR mean {inv['irr'].mean():.2%} | desk P&L std: none {desk['none'].std():.1%},"
      f" DV01 {desk['dv01'].std():.1%}, DV01+slope {desk['dv01_slope'].std():.1%} | P&L explain R2 {r2:.0%}")

save(pd.Series(summary, name="value"), "pricing_summary")
step("Done")

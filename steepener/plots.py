"""Charts (matplotlib) and one interactive 3D vol surface (plotly)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import norm  # noqa: E402

from .pricing import CmsRate  # noqa: E402

TENOR_COLORS = {2: "#eb6834", 10: "#2a78d6"}  # one fixed color per swap tenor
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL = ["#86b6ef", "#3987e5", "#184f95"]  # light to dark
POSITIVE, NEGATIVE, NEUTRAL = "#2a78d6", "#e34948", "#9a9993"
ACCENT = "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK_2, "ytick.color": INK_2,
    "font.size": 9, "legend.frameon": False, "axes.axisbelow": True, "lines.linewidth": 2, "figure.dpi": 110,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _kinks(ax, note, scale=1e4, axis="h"):
    k_low, k_high = note.strikes
    for k, label in [(k_low, f"floor binds below {k_low * 1e4:.0f}bp"), (k_high, f"cap binds above {k_high * 1e4:.0f}bp")]:
        if axis == "h":
            ax.axhline(k * scale, color=NEUTRAL, ls="--", lw=1)
            ax.annotate(label, (ax.get_xlim()[1], k * scale), fontsize=7.5, color=INK_2, va="bottom", ha="right",
                        xytext=(-4, 2), textcoords="offset points")
        else:
            ax.axvline(k * scale, color=NEUTRAL, ls="--", lw=1)


def curves(market, note, cms_rows, path):
    t = np.linspace(0.25, 30, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(t, market.zero(t), color=ORDINAL[2], label="ESTR (discounting)")
    ax1.plot(t, market.zero(t, "proj"), color=ORDINAL[0], label="Euribor 6M (projection)")
    ax1.set(title="EUR zero curves", xlabel="maturity (years)", ylabel="zero rate (cont.)")
    _pct(ax1, decimals=1)
    ax1.legend()

    fixings = note.fixings + note.elapsed
    fwd = [(market.swap_rate(T, 10) - market.swap_rate(T, 2)) * 1e4
           for T in note.fixings]
    cms = cms_rows.pivot(index="expiry", columns="tenor", values="cms").loc[note.fixings]
    ax2.bar(fixings - 0.18, fwd, width=0.36, color=ORDINAL[0], label="forward swap spread 10Y - 2Y")
    ax2.bar(fixings + 0.18, (cms[10] - cms[2]) * 1e4, width=0.36, color=ORDINAL[2],
            label="CMS spread (convexity adjusted)")
    ax2.set(title="Forward 2s10s at each coupon fixing", xlabel="fixing (years from issue, 2028 = 2)", ylabel="bp")
    ax2.set_ylim(min(0, min(fwd) - 10), max(175, max(fwd) + 20))
    _kinks(ax2, note)
    ax2.legend(loc="upper left")
    _save(fig, path)


def smiles(market, path, expiries=(1, 5, 9)):
    q = market.quotes["vols"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tenor in zip(axes, sorted(market.sabr)):
        for T, color in zip(expiries, ORDINAL):
            sub = q[(q["tenor"] == tenor) & (q["expiry"] == T)]
            off = np.linspace(sub["offset_bp"].min(), sub["offset_bp"].max(), 100)
            ax.plot(off, market.vol(T, tenor, 0.0, off / 1e4) * 1e4, color=color, label=f"{T}Y expiry (SABR)")
            ax.scatter(sub["offset_bp"], sub["normal_vol_bp"], color=color, s=14, zorder=3)
        ax.set(title=f"Swaption smiles, {tenor}Y swap (dots: Bloomberg, lines: SABR)", xlabel="strike - forward (bp)")
        ax.legend()
    axes[0].set_ylabel("normal vol (bp)")
    _save(fig, path)


def surfaces_3d_html(market, path):
    """Interactive SABR normal vol surfaces (expiry x strike offset), one per swap tenor."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    T = np.linspace(1, 10, 40)
    off = np.linspace(-300, 300, 61)
    tenors = sorted(market.sabr)
    fig = make_subplots(rows=1, cols=len(tenors), specs=[[{"type": "surface"}] * len(tenors)],
                        subplot_titles=[f"{n}Y swap" for n in tenors])
    colorscale = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]
    for col, tenor in enumerate(tenors, start=1):
        z = np.array([[float(market.vol(t, tenor, 0.0, o / 1e4)) * 1e4 for o in off] for t in T])
        fig.add_trace(go.Surface(x=off, y=T, z=z, colorscale=colorscale, showscale=False,
                                 hovertemplate="K-F %{x:.0f}bp<br>expiry %{y:.1f}y<br>vol %{z:.1f}bp<extra></extra>"),
                      row=1, col=col)
    axis = dict(xaxis_title="strike - forward (bp)", yaxis_title="expiry (y)", zaxis_title="normal vol (bp)")
    fig.update_layout(scene=axis, scene2=axis, height=600, margin=dict(l=0, r=0, t=60, b=0),
                      title=dict(text="EUR swaption normal vol surfaces (SABR)", x=0.5), paper_bgcolor=SURFACE)
    fig.write_html(path, include_plotlyjs=True)


def convexity(cms_rows, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for tenor, grp in cms_rows.groupby("tenor"):
        ax.plot(grp["expiry"], grp["convexity_bp"], marker="o", color=TENOR_COLORS[tenor], label=f"CMS {tenor}Y")
    ax.set(title="CMS convexity adjustment (TSR replication on the SABR smile)",
           xlabel="fixing (years), paid one year later", ylabel="CMS rate - forward swap rate (bp)")
    ax.legend()
    _save(fig, path)


def spread_distribution(market, note, path, fixing=5.0,
                        correlations=(0.6, None, 0.97), n_samples=400_000, seed=1):
    """Density of the CMS spread, sampled from the copula (the pricer itself uses quadrature),
    with the coupon it pays on a second panel."""
    long = CmsRate(market, fixing, 10, fixing + 1)
    short = CmsRate(market, fixing, 2, fixing + 1)
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal((2, n_samples))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    bins = np.linspace(-0.02, 0.025, 91)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for rho, color in zip(correlations, ORDINAL):
        r = market.corr(fixing) if rho is None else rho
        zs = r * z1 + np.sqrt(1 - r**2) * z2
        spread = long.quantile(norm.cdf(z1)) - short.quantile(norm.cdf(zs))
        hist, _ = np.histogram(spread, bins=bins, density=True)
        label = f"rho = {r:.2f}" + (" (base case, ECB history)" if rho is None else "")
        ax.plot(centers * 1e4, hist / 1e4, color=color, label=label)
    ax.set(title=f"CMS 10Y - 2Y spread fixing in {fixing + note.elapsed:g} years from issue (copula on SABR marginals)",
           ylabel="density (per bp)")
    ax.legend(loc="upper left")
    s = np.linspace(bins[0], bins[-1], 400)
    ax2.plot(s * 1e4, note.coupon(s), color=ACCENT, lw=1.8)
    ax2.set(xlabel="spread (bp)", ylabel="coupon", title="Coupon paid for that spread")
    ax2.set_ylim(0, note.cap * 1.2)
    _pct(ax2)
    for k in note.strikes:
        for a in (ax, ax2):
            a.axvline(k * 1e4, color=NEUTRAL, ls="--", lw=1)
    _save(fig, path)


def correlation_heatmap(grid, path):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    data = grid.to_numpy(float)
    ax.imshow(data, cmap=BLUES, origin="lower", aspect="auto")
    for (r, c), v in np.ndenumerate(data):
        ax.text(c, r, f"{v:.1%}", ha="center", va="center", fontsize=8,
                color="white" if v > np.percentile(data, 60) else INK)
    ax.set_xticks(range(grid.shape[1]), [f"{c:.0%}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{v:.0%}" for v in grid.index])
    ax.set(title="Note price without the call vs correlation and vol level",
           xlabel="correlation 10Y / 2Y", ylabel="vols (% of market)")
    ax.grid(False)
    _save(fig, path)


def model_comparison(prices, path, issue_price=1.0):
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(prices)
    values = np.array(list(prices.values()))
    colors = [ORDINAL[0]] * (len(names) - 1) + [ORDINAL[2]]
    bars = ax.bar(range(len(names)), values, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{v:.2%}" for v in values], fontsize=8, color=INK_2, padding=2)
    ax.axhline(issue_price, color=NEUTRAL, ls="--", lw=1)
    ax.annotate("100%", (len(names) - 0.6, issue_price), fontsize=8, color=INK_2, ha="right",
                va="bottom", xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(values.min() - 0.06, values.max() + 0.03)
    ax.set(title="Value of the note on 29/10/2026 by model (incl. accrued)", ylabel="% of notional")
    _pct(ax)
    _save(fig, path)


def waterfall(blocks, price, path, issue_price=1.0):
    labels = list(blocks) + ["Note value", "Gap to\n100%", "100%"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    level = 0.0
    for i, v in enumerate(blocks.values()):
        ax.bar(i, v, bottom=level if v >= 0 else level + v, color=POSITIVE if v >= 0 else NEGATIVE, width=0.6)
        ax.annotate(f"{v:+.1%}", (i, level + max(v, 0)), ha="center", va="bottom", fontsize=8, color=INK_2,
                    xytext=(0, 2), textcoords="offset points")
        level += v
    n = len(blocks)
    margin = issue_price - price
    ax.bar(n, price, color=NEUTRAL, width=0.6)
    ax.bar(n + 1, margin, bottom=price if margin >= 0 else price + margin, color=ORDINAL[0], width=0.6)
    ax.bar(n + 2, issue_price, color=NEUTRAL, width=0.6)
    for i, v, top in [(n, price, price), (n + 1, margin, max(price, issue_price)), (n + 2, issue_price, issue_price)]:
        ax.annotate(f"{v:.1%}" if i != n + 1 else f"{v:+.1%}", (i, top), ha="center", va="bottom", fontsize=8,
                    color=INK_2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(n + 3), labels, fontsize=8)
    ax.set_ylim(0, max(sum(v for v in blocks.values() if v > 0), issue_price) * 1.08)
    ax.set(title="What the investor holds: building blocks of the note on 29/10/2026", ylabel="% of notional")
    _pct(ax)
    _save(fig, path)


def call_profile(call_probability, expected_life, spread_paths, called_at,
                 note, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    dates = np.array(list(call_probability))
    probs = np.array(list(call_probability.values()))
    ax1.bar(dates + 2026, probs, color=ORDINAL[1], width=0.6, label="called on this date")
    ax1.plot(dates + 2026, np.cumsum(probs), color=ORDINAL[2], marker="o", lw=1.5, label="called by this date")
    ax1.set(title=f"Issuer's call: P(called) = {probs.sum():.0%}, expected life {expected_life:.1f} years",
            xlabel="call date (1 September)", ylabel="probability")
    _pct(ax1)
    ax1.legend()

    # coupon fixed at the first call date (2028) for paths called then vs kept
    first = note.call_dates[0]
    col = first - len(note.fixed_rates)  # spread fixed on the first call date
    s = spread_paths[:, col] * 1e4
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 70)
    ax2.hist(s[called_at != first], bins=bins, color=ORDINAL[0], label="kept alive")
    ax2.hist(s[called_at == first], bins=bins, color=NEGATIVE, alpha=0.8, label="called")
    ax2.axvline(note.strikes[0] * 1e4, color=NEUTRAL, ls="--", lw=1)
    ax2.axvline(note.strikes[1] * 1e4, color=NEUTRAL, ls="--", lw=1)
    ax2.set(title=f"Who gets called in {2026 + first}: simulated 2s10s at that date",
            xlabel="10Y - 2Y swap spread (bp)", ylabel="paths")
    ax2.legend()
    _save(fig, path)


def key_rates_and_vega(kr, vega, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in kr["bp_of_notional"]]
    ax1.bar(kr["tenor"].astype(str), kr["bp_of_notional"], color=colors, width=0.6)
    ax1.axhline(0, color=INK_2, lw=0.8)
    ax1.set(title="Key-rate deltas (non-callable)", xlabel="curve tenor (years)",
            ylabel="price change per +1bp (bp of notional)")
    width = 0.38
    exp = sorted(vega["expiry"].unique())
    x = np.arange(len(exp))
    for i, tenor in enumerate(sorted(vega["tenor"].unique())):
        v = vega[vega["tenor"] == tenor].set_index("expiry").loc[exp, "bp_of_notional"]
        ax2.bar(x + (i - 0.5) * width, v, width * 0.95, color=TENOR_COLORS[tenor], label=f"{tenor}Y swap smile")
    ax2.axhline(0, color=INK_2, lw=0.8)
    ax2.set_xticks(x, [f"{e:g}Y" for e in exp])
    ax2.set(title="Vega by swaption bucket (non-callable)", xlabel="swaption expiry",
            ylabel="price change per +1bp vol (bp of notional)")
    ax2.legend()
    _save(fig, path)


def shift_profiles(parallel, slope, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, df, title, xlabel in [(axes[0], parallel, "Parallel move of the curves", "parallel shift (bp)"),
                                  (axes[1], slope, "Steepening / flattening of 2s10s", "change in 2s10s (bp)")]:
        ax.plot(df["shift_bp"], df["non_callable"], color=ORDINAL[0], label="without the call")
        ax.plot(df["shift_bp"], df["callable"], color=ORDINAL[2], label="callable (the real note)")
        ax.axvline(0, color=NEUTRAL, lw=0.8)
        ax.set(title=title, xlabel=xlabel)
        _pct(ax)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("note value (% of notional)")
    _save(fig, path)


def levers(table, path, target):
    base = table.iloc[0]["participation"]
    data = table.iloc[1:].copy()
    impossible = data.loc[data["change"].isna(), "lever"].tolist()
    data = data.dropna()
    data = data.reindex(data["change"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in data["change"]]
    bars = ax.barh(data["lever"], data["change"], color=colors, height=0.6)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in data["change"]], fontsize=8, color=INK_2, padding=3)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set(title=f"What moves the leverage at a constant price (term sheet {base:.2f}x, price {target:.1%})",
           xlabel="change in participation (blue: more leverage for the investor)"
                  + (f"\nnot reachable at this price: {', '.join(impossible)}" if impossible else ""))
    ax.grid(axis="y", visible=False)
    lim = np.nanmax(np.abs(data["change"])) * 1.35
    ax.set_xlim(-lim, lim)
    _save(fig, path)


def hedge_pnl(unhedged, hedged, hedge, path):
    """Desk P&L after instant curve moves, before and after the 2Y / 10Y swap hedge."""
    lim = max(np.abs(unhedged.to_numpy()).max(), np.abs(hedged.to_numpy()).max())
    cmap = LinearSegmentedColormap.from_list("div", [NEGATIVE, "#f0efec", POSITIVE])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    def leg(n, x):
        return f"{'receive' if x >= 0 else 'pay'} {abs(x):.2f} x {n} swap"

    titles = ["Desk short the note, no hedge",
              f"Hedged: {leg('2Y', hedge['receive_2y'])}, {leg('10Y', hedge['receive_10y'])}"]
    for ax, grid, title in zip(axes, [unhedged, hedged], titles):
        data = grid.to_numpy(float)
        ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for (r, c), v in np.ndenumerate(data):
            ax.text(c, r, f"{v:+.2%}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(grid.shape[1]), [f"{c:+d}" for c in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [f"{v:+d}" for v in grid.index])
        ax.set(title=title, xlabel="change in 2s10s (bp)")
        ax.grid(False)
    axes[0].set_ylabel("parallel move (bp)")
    fig.suptitle("Desk P&L after an instant curve move (% of notional)", fontweight="bold", fontsize=11)
    _save(fig, path)


# ---------- new charts for the Bloomberg / ECB version ----------

def _heatmap(ax, data, rows, cols, fmt, cmap, vmin=None, vmax=None, fontsize=6.5):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    hi = np.nanpercentile(data, 65) if vmin is None else None
    for (r, c), v in np.ndenumerate(data):
        dark = (v > hi) if hi is not None else False
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=fontsize, color="white" if dark else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    return im


def _expiry_label(e):
    return f"{round(e * 12)}M" if e < 1 else f"{e:g}Y"


DIVERGING = LinearSegmentedColormap.from_list("div", [NEGATIVE, "#f0efec", POSITIVE])


def atm_cube(cube, path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _heatmap(ax, cube.to_numpy(float), [_expiry_label(e) for e in cube.index], [f"{c}Y" for c in cube.columns],
             "{:.0f}", BLUES)
    ax.set(title="EUR swaption ATM normal vols on 29/10/2026 (bp per year, Bloomberg VCUB)",
           xlabel="underlying swap tenor", ylabel="option expiry")
    for n in (2, 10):  # the two swaps the note is written on
        c = list(cube.columns).index(n)
        ax.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, len(cube), fill=False, ec=INK, lw=1.4))
    _save(fig, path)


def sabr_errors(market, path):
    """SABR vol - Bloomberg vol for every smile quote, per swap tenor."""
    v = market.quotes["vols"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, tenor in zip(axes, sorted(market.sabr)):
        sub = v[v["tenor"] == tenor].copy()
        sub["fit"] = [float(market.vol(T, tenor, 0.0, o / 1e4)) * 1e4 for T, o in zip(sub["expiry"], sub["offset_bp"])]
        grid = (sub["fit"] - sub["normal_vol_bp"]).to_numpy().reshape(-1, sub["offset_bp"].nunique())
        exps = sorted(sub["expiry"].unique())
        offs = sorted(sub["offset_bp"].unique())
        _heatmap(ax, grid, [_expiry_label(e) for e in exps], [f"{o:+d}" if o else "ATM" for o in offs],
                 "{:+.1f}", DIVERGING, -3, 3)
        ax.set(title=f"SABR fit error, {tenor}Y swap (bp of vol)", xlabel="strike - ATM (bp)")
    axes[0].set_ylabel("option expiry")
    _save(fig, path)


def g2pp_cube_errors(model, cube, path, max_expiry=15):
    cube = cube[cube.index <= max_expiry]
    err = np.array([[model.normal_vol(T, n) * 1e4 - cube.loc[T, n] for n in cube.columns] for T in cube.index])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    lim = np.ceil(np.abs(err).max())
    _heatmap(ax, err, [_expiry_label(e) for e in cube.index], [f"{c}Y" for c in cube.columns], "{:+.0f}",
             DIVERGING, -lim, lim)
    ax.set(title=f"G2++ vol - market vol on the whole ATM cube (bp). RMS {np.sqrt(np.mean(err**2)):.1f}bp. "
                 "Boxes: the 12 points used in the calibration",
           xlabel="underlying swap tenor", ylabel="option expiry")
    for T in (1, 2, 3, 5, 7, 10):
        for n in (2, 10):
            r, c = list(cube.index).index(T), list(cube.columns).index(n)
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    _save(fig, path)
    return pd.DataFrame(err, index=cube.index, columns=cube.columns)


def steps_waterfall(steps, path, title, ylabel="% of notional"):
    """Waterfall from a start value to an end value through named steps: [(label, value, kind)],
    kind = "total" for a full bar, "step" for a change."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    level, tops = 0.0, []
    for i, (label, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, color=NEUTRAL, width=0.6)
            level = v
            txt, top = f"{v:.2%}", v
        else:
            ax.bar(i, abs(v), bottom=min(level, level + v), color=POSITIVE if v >= 0 else NEGATIVE, width=0.6)
            txt, top = f"{v:+.2%}", max(level, level + v)
            level += v
        tops.append(top)
        ax.annotate(txt, (i, top), ha="center", va="bottom", fontsize=8, color=INK_2, xytext=(0, 2),
                    textcoords="offset points")
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=8)
    lows = [s[1] for s in steps if s[2] == "total"]
    ax.set_ylim(min(lows) - 0.06, max(tops) + 0.02)
    ax.set(title=title, ylabel=ylabel)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.grid(axis="x", visible=False)
    _pct(ax)
    _save(fig, path)


def historical_correlation(rolling, base, profile, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(rolling.index, rolling, color=ORDINAL[1], lw=1.4)
    ax1.axhline(base, color=INK_2, ls="--", lw=1)
    ax1.annotate(f"base case {base:.2f} (last 5 years)", (rolling.index[0], base), fontsize=8, color=INK_2,
                 va="bottom", xytext=(2, 3), textcoords="offset points")
    ax1.set(title="Correlation of weekly changes, EUR 2Y vs 10Y (1-year rolling, ECB AAA)",
            ylabel="correlation")
    lo, hi = rolling.quantile(0.1), rolling.quantile(0.9)
    ax2.axvspan(lo, hi, color=GRID, alpha=0.7, lw=0)
    ax2.annotate("80% of history", (lo, profile["callable"].min()), fontsize=8, color=INK_2,
                 xytext=(3, 3), textcoords="offset points")
    ax2.plot(profile["correlation"], profile["non_callable"], color=ORDINAL[0], marker="o", label="without the call")
    ax2.plot(profile["correlation"], profile["callable"], color=ORDINAL[2], marker="o", label="callable (the real note)")
    ax2.axvline(base, color=INK_2, ls="--", lw=1)
    ax2.set(title="Note value vs correlation (G2++ recalibrated each time)", xlabel="10Y / 2Y correlation",
            ylabel="value (% of notional)")
    _pct(ax2)
    ax2.legend()
    _save(fig, path)


def investor_irr(irr, groups, bond_yield, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    bins = np.linspace(min(0.015, irr.min()), max(0.065, irr.max()) + 0.001, 60)
    ax1.hist(irr, bins=bins, color=ORDINAL[1], edgecolor=SURFACE, linewidth=0.5)
    ax1.axvline(bond_yield, color=NEGATIVE, ls="--", lw=1.2)
    ax1.annotate(f"GS 10Y fixed-rate bond {bond_yield:.2%}", (bond_yield, ax1.get_ylim()[1] * 0.92), fontsize=8,
                 color=INK_2, ha="right", xytext=(-4, 0), textcoords="offset points")
    ax1.set(title=f"Investor IRR over all simulated paths (mean {irr.mean():.2%}, median {np.median(irr):.2%})",
            xlabel="IRR if bought at 100% and held to call or maturity", ylabel="paths")
    _pct(ax1, "x", 1)
    names = list(groups)
    prob = [m.mean() for m in groups.values()]
    mean_irr = [irr[m].mean() if m.any() else np.nan for m in groups.values()]
    y = np.arange(len(names))[::-1]
    bars = ax2.barh(y, prob, color=ORDINAL[1], height=0.55)
    ax2.bar_label(bars, labels=[f"{p:.0%}  (IRR {r:.2%})" for p, r in zip(prob, mean_irr)], fontsize=8,
                  color=INK_2, padding=3)
    ax2.set_yticks(y, names, fontsize=8)
    ax2.set_xlim(0, max(prob) * 1.6)
    ax2.set(title="What happens to the investor", xlabel="probability (risk-neutral)")
    _pct(ax2, "x")
    ax2.grid(axis="y", visible=False)
    _save(fig, path)


def margin_vs_funding(table, base_spread_bp, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(table["gs_spread_bp"], table["margin"], color=ORDINAL[2], marker="o")
    for _, r in table.iterrows():
        ax.annotate(f"{r['margin']:.1%}", (r["gs_spread_bp"], r["margin"]), fontsize=8, color=INK_2,
                    xytext=(0, 6), textcoords="offset points", ha="center")
    ax.axvline(base_spread_bp, color=INK_2, ls="--", lw=1)
    ax.annotate(f"GS 2035 bond: {base_spread_bp:.0f}bp", (base_spread_bp, table["margin"].min()), fontsize=8,
                color=INK_2, xytext=(4, 0), textcoords="offset points")
    ax.set(title="Issuer's margin (100% - value) if the note were launched today",
           xlabel="GS funding spread used to discount (bp over ESTR)", ylabel="margin (% of notional)")
    _pct(ax, decimals=0)
    _save(fig, path)


def historical_stress(table, base, path):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(table))[::-1]
    h = 0.38
    for off, col, color, label in [(h / 2, "non_callable", ORDINAL[0], "without the call"),
                                   (-h / 2, "callable", ORDINAL[2], "callable (the real note)")]:
        pnl = table[col] - base[col]
        bars = ax.barh(y + off, pnl, height=h * 0.95, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:+.1%}" for v in pnl], fontsize=7.5, color=INK_2, padding=3)
    ax.set_yticks(y, [f"{n}\n2Y {a:+.0f}bp, 10Y {b:+.0f}bp" for n, a, b in
                      zip(table["scenario"], table["move_2y_bp"], table["move_10y_bp"])], fontsize=8)
    ax.axvline(0, color=INK_2, lw=0.8)
    lim = np.abs(table[["non_callable", "callable"]].to_numpy() - base[["non_callable", "callable"]].to_numpy()).max()
    ax.set_xlim(-lim * 1.4, lim * 1.4)
    ax.set(title="Historical stress tests: real 2Y and 10Y moves applied to the note on 29/10/2026",
           xlabel="change in the note's value for the investor (% of notional)")
    _pct(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, path)


def coupon_history(hist, note, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax1.plot(hist.index, hist["slope"] * 1e4, color=ORDINAL[1], lw=1.2)
    for k in note.strikes:
        ax1.axhline(k * 1e4, color=NEUTRAL, ls="--", lw=1)
    ax1.annotate("floor binds below 40bp, cap above 120bp (dashed lines)", (hist.index[-1], -60), fontsize=8,
                 color=INK_2, ha="right")
    ax1.set(title="EUR 10Y - 2Y spread since 2004 (ECB AAA par yields)", ylabel="bp")
    c = note.coupon(hist["slope"])
    ax2.fill_between(hist.index, c, color=ORDINAL[0], lw=0)
    ax2.plot(hist.index, c, color=ORDINAL[2], lw=1)
    shares = f"at the floor {np.mean(c <= note.floor + 1e-12):.0%} of days, at the cap {np.mean(c >= note.cap - 1e-12):.0%}"
    ax2.set(title=f"Coupon the formula would have fixed each day: {shares}", ylabel="coupon")
    ax2.set_ylim(0, note.cap * 1.15)
    _pct(ax2)
    _save(fig, path)


def investor_backtest(table, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {2: NEGATIVE, 3: "#86b6ef", 4: "#3987e5", 5: "#eb6834", 0: "#0d366b"}
    for called, grp in table.groupby("called"):
        label = "never called" if called == 0 else f"called after {called} years"
        color = colors.get(called, NEUTRAL)
        ax1.scatter(grp["launch"], grp["irr"], s=14, color=color, label=label, zorder=3)
    bond = table["fixed_bond_yield"].iloc[0]
    ax1.axhline(bond, color=INK_2, ls="--", lw=1)
    ax1.annotate(f"GS fixed-rate bond {bond:.2%}", (table["launch"].iloc[-1], bond), fontsize=8, color=INK_2,
                 ha="right", va="bottom", xytext=(0, 3), textcoords="offset points")
    ax1.set(title="Investor IRR, note launched every month (rates re-based to today's level)",
            xlabel="launch date", ylabel="IRR")
    _pct(ax1, decimals=1)
    ax1.legend(fontsize=7.5, loc="lower center", ncol=2)
    counts = table["called"].replace(0, 10).value_counts().sort_index()
    ax2.bar([("never" if k == 10 else f"{k}y") for k in counts.index], counts / len(table), color=ORDINAL[1], width=0.6)
    ax2.set(title=f"When the note ends ({len(table)} launches)", xlabel="call after ... years", ylabel="share of notes")
    _pct(ax2)
    _save(fig, path)


def desk_hedge_backtest(table, path):
    names = {"none": "No hedge", "dv01": "DV01 hedge\n(10Y swap)", "dv01_slope": "DV01 + slope hedge\n(2Y and 10Y swaps)"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    bins = np.linspace(table[list(names)].min().min(), table[list(names)].max().max(), 40)
    for col, color in zip(names, [NEUTRAL, ORDINAL[0], ORDINAL[2]]):
        ax1.hist(table[col], bins=bins, histtype="step", color=color, lw=1.8, label=names[col].replace("\n", " "))
    ax1.axvline(0, color=INK_2, lw=0.8)
    ax1.set(title="Desk P&L at the end of each note (short the note, monthly rebalancing)",
            xlabel="P&L (% of notional)", ylabel="notes")
    _pct(ax1, "x")
    ax1.legend(fontsize=7.5)
    std = [table[c].std() for c in names]
    bars = ax2.bar(list(names.values()), std, color=[NEUTRAL, ORDINAL[0], ORDINAL[2]], width=0.55)
    ax2.bar_label(bars, labels=[f"{v:.1%}" for v in std], fontsize=8, color=INK_2, padding=2)
    ax2.set(title="Dispersion of the desk P&L (standard deviation)", ylabel="% of notional")
    ax2.tick_params(axis="x", labelsize=8)
    _pct(ax2)
    _save(fig, path)


EXPLAIN_COLORS = {"carry": "#9ec5f4", "delta": "#2a78d6", "slope": "#eb6834", "gamma": "#1baf7a",
                  "call": "#7f52c4", "residual": "#9a9993"}


def pnl_explain(example, all_months, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
    parts = list(EXPLAIN_COLORS)
    x = example["date"]
    pos, neg = np.zeros(len(x)), np.zeros(len(x))
    for p in parts:
        v = example[p].to_numpy()
        ax1.bar(x, np.where(v > 0, v, 0), bottom=pos, width=25, color=EXPLAIN_COLORS[p], label=p)
        ax1.bar(x, np.where(v < 0, v, 0), bottom=neg, width=25, color=EXPLAIN_COLORS[p])
        pos, neg = pos + np.maximum(v, 0), neg + np.minimum(v, 0)
    ax1.plot(x, example["note_pnl"], color=INK, lw=1, marker="o", ms=2.5, label="actual")
    ax1.axhline(0, color=INK_2, lw=0.8)
    ax1.set(title=f"Monthly P&L explain of one note (launched {example['launch'].iloc[0]:%b %Y}, investor view)",
            ylabel="% of notional")
    _pct(ax1, decimals=1)
    ax1.legend(ncol=4, fontsize=7.5, loc="lower left")
    explained = all_months[parts[:-1]].sum(axis=1)
    r2 = 1 - np.sum((all_months["note_pnl"] - explained) ** 2) / np.sum((all_months["note_pnl"] - all_months["note_pnl"].mean()) ** 2)
    ax2.scatter(explained, all_months["note_pnl"], s=4, color=ORDINAL[1], alpha=0.4)
    lim = np.percentile(np.abs(all_months["note_pnl"]), 99.5)
    ax2.plot([-lim, lim], [-lim, lim], color=INK_2, lw=0.8, ls="--")
    ax2.set(xlim=(-lim, lim), ylim=(-lim, lim), title=f"All notes, all months: R² = {r2:.0%}",
            xlabel="explained P&L (Greeks)", ylabel="actual P&L")
    _pct(ax2, "x", 1)
    _pct(ax2, "y", 1)
    _save(fig, path)
    return r2

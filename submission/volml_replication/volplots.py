"""
volplots.py - slide-ready figures for the volatility-forecasting replication.

One consistent style: light surface, recessive grid, thin marks, fixed colour per
model (colour follows the model, never its rank), direct labels where lines end.
Every function returns the figure and, if `save` is given, writes a PNG (220 dpi).
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import volml as V

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b9b8b2"

KEY_MODELS = ["LogHAR", "EN", "RF", "NN2_ens"]                # the four models on most slides
COLOR = {"LogHAR": SERIES[0], "RF": SERIES[1], "EN": SERIES[2], "NN2_ens": SERIES[3], "GB": SERIES[4],
         "HAR": INK2, "RF+LogHAR": INK, "LogRF": SERIES[6]}
NAME = {"HAR": "HAR", "HAR-X": "HAR-X", "LogHAR": "LogHAR", "LevHAR": "LevHAR", "SHAR": "SHAR",
        "HARQ": "HARQ", "RR": "Ridge", "LA": "Lasso", "EN": "Elastic net", "RF": "Random forest",
        "BG": "Bagging", "GB": "Gradient boosting", "NN2_1": "Neural net (single)",
        "NN2_ens": "Neural net (ensemble)", "RF+LogHAR": "RF + LogHAR (50/50)", "LogRF": "RF on log RV (ours)"}
ORDER = ["HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ", "RR", "LA", "EN", "RF", "GB", "NN2_1", "NN2_ens"]
DS_NAME = {"M_HAR": "RV lags only (M_HAR)", "M_ALL": "All predictors (M_ALL)"}
H_NAME = {1: "1 day", 5: "1 week", 22: "1 month"}
# diverging map for relative MSE: blue = beats HAR, gray = equal, red = worse
DIVERGING = LinearSegmentedColormap.from_list("rel", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c03b3a"])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])


def color(m):
    return COLOR.get(m, OTHER)


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "font.size": 10.5, "figure.dpi": 110,
    })


def _save(fig, save):
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=220, bbox_inches="tight")
    return fig


def _subtitle(ax, text):
    """Grey subtitle under the (left-aligned) title; call after ax.set_title."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=24)
    ax.text(0, 1.012, text, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def _end_labels(ax, series: dict, fmt="{:.2f}"):
    """Direct labels just right of the plot area, at each line's last value, nudged
    apart so they never collide. Text in ink; a small coloured dot carries identity."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    items = sorted(((float(s.iloc[-1]), k) for k, s in series.items()), key=lambda t: t[0])
    placed = []
    for y, _ in items:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for (y, k), yl in zip(items, placed):
        ax.scatter([1.012], [yl], s=28, color=color(k), transform=trans, clip_on=False, zorder=5)
        ax.text(1.025, yl, f"{NAME.get(k, k)}  {fmt.format(y)}", transform=trans, va="center",
                fontsize=9, color=INK)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_rv_timeline(panel, splits: dict, events: dict | None = None, save=None, figsize=None):
    """Cross-sectional median annualized realized volatility (IQR band) with the
    train / validation / test periods of each split scenario as bars underneath."""
    rv = panel.pivot(index="date", columns="symbol", values="rv")
    vol = np.sqrt(252 * rv)
    med, q1, q3 = vol.median(1), vol.quantile(.25, 1), vol.quantile(.75, 1)
    if not splits:                                     # RV panel only (e.g. for a data slide)
        fig, ax = plt.subplots(figsize=figsize or (12, 4.6))
    else:
        fig = plt.figure(figsize=figsize or (12, 5.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 0.32 * len(splits) + 0.3], hspace=0.08)
        ax = fig.add_subplot(gs[0])
    ax.fill_between(med.index, q1, q3, color=SERIES[0], alpha=0.18, lw=0, label="Interquartile range")
    ax.plot(med.index, med, color=SERIES[0], lw=1.2, label="Median across stocks")
    ax.set_ylabel("Realized volatility, annualized (%)")
    ax.set_title("Realized volatility of the Dow 30, 2016–2026")
    _subtitle(ax, "Daily 5-minute realized variance, annualized; median and interquartile range across stocks")
    ax.legend(loc="upper right")
    ax.set_xlim(med.index[0], med.index[-1])
    if splits:
        ax.tick_params(labelbottom=False)
    for name, (d0, d1) in (events or {}).items():
        d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
        y = med[(med.index >= d0) & (med.index <= d1)].max()
        ax.annotate(name, xy=(d0 + (d1 - d0) / 2, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    if not splits:
        return _save(fig, save)
    bx = fig.add_subplot(gs[1], sharex=ax)
    shades = {"train": "#9ec5f4", "val": "#fab219", "test": "#e34948"}
    for i, (name, spec) in enumerate(splits.items()):
        d_tr, d_va = spec[0], spec[1]
        excl = spec[2] if len(spec) > 2 else None
        y = len(splits) - 1 - i
        start, end = med.index[22], med.index[-1]
        for seg, a, b in (("train", start, d_tr), ("val", d_tr, d_va), ("test", d_va, end)):
            bx.barh(y, pd.Timestamp(b) - pd.Timestamp(a), left=pd.Timestamp(a), height=0.6,
                    color=shades[seg], edgecolor=SURFACE, linewidth=2)
        if excl:
            a, b = pd.Timestamp(excl[0]), pd.Timestamp(excl[1])
            bx.barh(y, b - a, left=a, height=0.6, color=SURFACE, edgecolor=MUTED, hatch="////", linewidth=0.8)
        bx.text(-0.005, y, name, transform=bx.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, color=INK)
    bx.set_yticks([])
    bx.grid(False)
    bx.spines["left"].set_visible(False)
    handles = [mpl.patches.Patch(color=c, label={"train": "Training", "val": "Validation",
                                                 "test": "Test (out-of-sample)"}[k]) for k, c in shades.items()]
    if any(len(v) > 2 and v[2] for v in splits.values()):
        handles.append(mpl.patches.Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="////",
                                         label="Removed from estimation"))
    bx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=len(handles))
    return _save(fig, save)


def fig_heatmap_vs_paper(preds, ds="M_ALL", models=None, extensions=(), save=None, figsize=None):
    """Relative MSE vs HAR: ours next to the paper, per horizon. Blue < 1 (beats HAR).
    `extensions`: our own models (no paper value), listed below a separator."""
    models = models or [m for m in ORDER if m in next(iter(preds.values()))[(ds, 1)].columns]
    n_base = len(models)
    models = list(models) + [m for m in extensions if m in next(iter(preds.values()))[(ds, 1)].columns]
    cols, data = [], []
    for h in (1, 5, 22):
        ours = V.rel_mse(preds, ds, h)
        data += [ours.reindex(models).values, pd.Series(V.PAPER[(ds, h)]).reindex(models).values]
        cols += [f"{H_NAME[h]}\nours", f"{H_NAME[h]}\npaper"]
    M = np.array(data).T
    fig, ax = plt.subplots(figsize=figsize or (8.6, 0.42 * len(models) + 1.6))
    norm = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.6)
    ax.imshow(np.clip(M, 0.6, 1.6), cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j] - 1) > 0.33 else INK,
                        fontweight="bold" if j % 2 == 0 else "normal")
            else:
                ax.text(j, i, "–", ha="center", va="center", color=MUTED)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(models)), [NAME[m] for m in models])
    ax.xaxis.tick_top()
    ax.grid(False)
    for j in (1.5, 3.5):
        ax.axvline(j, color=SURFACE, lw=4)
    if len(models) > n_base:
        ax.axhline(n_base - 0.5, color=INK, lw=1.5)
    ax.set_title(f"Relative MSE vs. HAR — {DS_NAME[ds]}", pad=12)
    ax.text(0, -0.02, "Below 1 = beats the HAR (blue). Bold = our replication (2016–2026), "
            "regular = original paper (2001–2017).", transform=ax.transAxes, fontsize=9, color=INK2, va="top")
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, save)


def fig_dumbbell(preds, ds="M_ALL", models=None, save=None, figsize=None):
    """Ours (filled) vs paper (hollow) per model and horizon."""
    models = models or KEY_MODELS + ["GB"]
    fig, axes = plt.subplots(1, 3, figsize=figsize or (12, 0.5 * len(models) + 1.8), sharey=True)
    for ax, h in zip(axes, (1, 5, 22)):
        ours = V.rel_mse(preds, ds, h).reindex(models)
        paper = pd.Series(V.PAPER[(ds, h)]).reindex(models)
        for i, m in enumerate(models):
            y = len(models) - 1 - i
            ax.plot([paper[m], ours[m]], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            ax.scatter(paper[m], y, s=70, facecolor=SURFACE, edgecolor=color(m), lw=2, zorder=2)
            ax.scatter(ours[m], y, s=70, color=color(m), edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.annotate(f"{ours[m]:.2f}", (ours[m], y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        ax.axvline(1, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_title(H_NAME[h], fontsize=11.5, pad=14)
        ax.set_yticks(range(len(models)), [NAME[m] for m in models][::-1])
        ax.set_xlabel("MSE relative to HAR")
        lo = min(np.nanmin(ours.values), np.nanmin(paper.values), 0.9)
        hi = max(np.nanmax(ours.values), np.nanmax(paper.values), 1.1)
        ax.set_xlim(lo - 0.08, hi + 0.08)
        ax.set_ylim(-0.6, len(models) - 0.3)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Replication vs. original paper — {DS_NAME[ds]}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK2, markersize=8, label="Our replication (2016–2026)"),
               plt.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2,
                          markersize=8, label="Paper (2001–2017)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, save)


def fig_cumgain(preds, ds="M_ALL", h=1, models=None, events=None, save=None, figsize=None):
    """Cumulative relative loss differential vs HAR over the test period."""
    models = models or KEY_MODELS
    C = V.cumulative_gain(preds, ds, h, models)
    n = len(C)
    fig, ax = plt.subplots(figsize=figsize or (11, 4.8))
    for m in models:
        ax.plot(C.index, C[m] / n, color=color(m), lw=2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("Cumulative gain vs. HAR\n(share of HAR MSE)")
    ax.set_title(f"Where the gains come from — {DS_NAME[ds]}, {H_NAME[h]} ahead")
    _subtitle(ax, "Running sum of (HAR loss − model loss), scaled by each stock's mean HAR loss, averaged "
                  "across stocks. End value = 1 − relative MSE.")
    ax.set_xlim(C.index[0], C.index[-1])
    for name, (d0, d1) in (events or {}).items():
        d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
        if d1 < C.index[0] or d0 > C.index[-1]:
            continue
        ax.axvspan(d0, d1, color="#f0efec", zorder=0)
        ax.text(d0, ax.get_ylim()[1], f" {name}", va="top", fontsize=9, color=INK2)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    _end_labels(ax, {m: C[m] / n for m in models}, fmt="{:+.1%}")
    return _save(fig, save)


def fig_forecast_zoom(preds, sym, ds="M_ALL", h=1, models=None, start=None, end=None, save=None, figsize=None):
    """Realized vs forecast annualized volatility for one stock around an episode."""
    models = models or ["HAR", "LogHAR", "RF"]
    P = preds[sym][(ds, h)]
    P = P.loc[start:end]
    ann = lambda x: np.sqrt(252 * x)  # noqa: E731
    fig, ax = plt.subplots(figsize=figsize or (11, 4.4))
    ax.plot(P.index, ann(P["y"]), color=INK, lw=1.2, label="Realized")
    for m in models:
        ax.plot(P.index, ann(P[m]), color=color(m), lw=1.8, label=f"{NAME[m]} forecast")
    ax.set_ylabel("Volatility, annualized (%)")
    ax.set_title(f"{sym}: forecasts around the shock ({H_NAME[h]} ahead, {ds})")
    _subtitle(ax, "Forecast made at the close of each day for the next period's realized volatility")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(models) + 1)
    return _save(fig, save)


def fig_winrate(preds, ds="M_ALL", models=None, save=None, figsize=None):
    """Share of stocks where each model beats the HAR (sequential blue)."""
    models = models or [m for m in ORDER if m in next(iter(preds.values()))[(ds, 1)].columns]
    W = pd.DataFrame({H_NAME[h]: V.win_rate(preds, ds, h).reindex(models) for h in (1, 5, 22)})
    fig, ax = plt.subplots(figsize=figsize or (5.8, 0.42 * len(models) + 1.4))
    ax.imshow(W.values, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            v = W.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.6 else INK)
    ax.set_xticks(range(3), W.columns)
    ax.set_yticks(range(len(models)), [NAME[m] for m in models])
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Share of stocks beating the HAR — {ds}", pad=12)
    return _save(fig, save)


def fig_scenarios(results: dict, ds="M_ALL", models=None, cap=2.0, save=None, figsize=None):
    """Relative MSE of key models across split scenarios (small multiples per horizon).
    Values above `cap` are drawn at the cap as a triangle with the true value written next to it."""
    models = models or KEY_MODELS
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=figsize or (12.5, 4.3), sharey=True)
    lo = 1.0
    for ax, h in zip(axes, (1, 5, 22)):
        n_clip = {}
        for m in models:
            ys = np.array([V.rel_mse(results[s], ds, h).get(m, np.nan) for s in names], float)
            lo = min(lo, np.nanmin(ys))
            yc = np.minimum(ys, cap)
            ax.plot(range(len(names)), yc, color=color(m), marker="o", ms=7, lw=1.8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            for i in np.where(ys > cap)[0]:
                k = n_clip.get(i, 0); n_clip[i] = k + 1
                ax.scatter(i, cap, marker="^", s=90, color=color(m), edgecolor=SURFACE, zorder=4)
                ax.annotate(f"{NAME.get(m, m)} {ys[i]:.1f}", (i, cap), xytext=(9, -4 - 12 * k),
                            textcoords="offset points", fontsize=8.5, color=INK)
        ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(range(len(names)), names, rotation=0)
        ax.set_title(H_NAME[h], fontsize=11.5)
        ax.set_xlim(-0.3, len(names) - 0.7)
    for ax in axes:
        ax.set_ylim(lo - 0.06, cap + 0.12)
    axes[0].set_ylabel("MSE relative to HAR")
    handles = [plt.Line2D([], [], color=color(m), marker="o", label=NAME[m]) for m in models]
    fig.suptitle(f"Does the result depend on where COVID sits? — {DS_NAME[ds]}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles=handles, loc="upper left", ncol=len(models), bbox_to_anchor=(0.01, 0.93), frameon=False)
    return _save(fig, save)


def fig_coefficients(coef: pd.DataFrame, save=None, figsize=None):
    """HAR coefficients per stock, estimated with vs without the COVID window."""
    terms = ["RVD", "RVW", "RVM"]
    fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, model in zip(axes, ("HAR", "HAR-X")):
        C = coef[coef["model"] == model]
        for k, (variant, col) in enumerate((("with COVID", SERIES[1]), ("without 2020", SERIES[0]))):
            sub = C[C["sample"] == variant]
            for i, t in enumerate(terms):
                x = i + (k - 0.5) * 0.32
                ax.scatter(x + rng.uniform(-0.06, 0.06, len(sub)), sub[t], s=16, color=col, alpha=0.75,
                           edgecolor="none")
                ax.hlines(sub[t].median(), x - 0.12, x + 0.12, color=INK, lw=2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(3), ["Daily RV", "Weekly RV", "Monthly RV"])
        ax.set_title(model, fontsize=11.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("OLS coefficient (in-sample)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[1], label="Estimated with COVID (2020) in sample"),
               plt.Line2D([], [], marker="o", ls="", color=SERIES[0], label="Estimated without 2020"),
               plt.Line2D([], [], color=INK, lw=2, label="Median across stocks")]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.0))
    fig.suptitle("COVID flips the monthly HAR coefficient", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, save)


def fig_boxplot(preds, ds="M_ALL", h=1, models=None, save=None, figsize=None):
    models = models or [m for m in ORDER if m in next(iter(preds.values()))[(ds, h)].columns]
    R = V.per_stock_relative(preds, ds, h)[models]
    fig, ax = plt.subplots(figsize=figsize or (11, 4.4))
    bp = ax.boxplot([R[m].dropna() for m in models], patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED), flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none"))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(color(m))
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xticks(range(1, len(models) + 1), [NAME[m] for m in models], rotation=35, ha="right")
    ax.set_ylabel("MSE relative to HAR (per stock)")
    ax.set_title(f"Cross-section of relative MSE — {DS_NAME[ds]}, {H_NAME[h]} ahead")
    return _save(fig, save)


def fig_deciles(dec: pd.DataFrame, save=None, figsize=None):
    """Relative MSE per decile of realized variance (paper Fig. 5)."""
    fig, ax = plt.subplots(figsize=figsize or (10, 4.2))
    for m in dec.columns:
        ax.plot(dec.index + 1, dec[m], color=color(m), marker="o", ms=5, lw=1.8)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile of realized variance in the test set (1 = calmest, 10 = most volatile)")
    ax.set_ylabel("MSE relative to HAR")
    ax.set_title("Accuracy across calm and turbulent days")
    ax.set_xlim(0.7, 10.3)
    ax.legend(handles=[plt.Line2D([], [], color=color(m), marker="o", label=NAME.get(m, m)) for m in dec.columns],
              loc="upper left", bbox_to_anchor=(1.01, 1))
    return _save(fig, save)


def fig_vi(VI: pd.DataFrame, save=None):
    """Variable importance (ALE-based, normalized), one panel per model (paper Fig. 7)."""
    cols = list(VI.columns)
    order = VI.mean(1).sort_values().index
    fig, axes = plt.subplots(1, len(cols), figsize=(3.1 * len(cols), 4.2), sharey=True)
    pal = {"HAR-X": OTHER, "EN": COLOR["EN"], "RF": COLOR["RF"], "NN": COLOR["NN2_ens"]}
    for ax, m in zip(np.atleast_1d(axes), cols):
        ax.barh(range(len(order)), VI.loc[order, m], color=pal.get(m, SERIES[0]), height=0.62,
                edgecolor=SURFACE, linewidth=1)
        for i, v in enumerate(VI.loc[order, m]):
            ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=8.5, color=INK2)
        ax.set_title({"NN": "Neural net", "EN": "Elastic net", "RF": "Random forest"}.get(m, m), fontsize=11.5)
        ax.set_yticks(range(len(order)), order)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
    fig.suptitle("Which predictors matter? (ALE-based variable importance, average across stocks)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, save)


def fig_ale(fitted: dict, X: pd.DataFrame, cols, sym="", save=None):
    pal = {"HAR-X": INK2, "EN": COLOR["EN"], "RF": COLOR["RF"], "NN": COLOR["NN2_ens"]}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.6))
    for ax, c in zip(np.atleast_1d(axes), cols):
        for name, m in fitted.items():
            e, a, _ = V.ale(m, X, c)
            ax.plot(e, a, color=pal.get(name, OTHER), lw=1.8, label=name)
        ax.set_title(c, fontsize=11.5)
        ax.set_xlabel(c)
    np.atleast_1d(axes)[0].set_ylabel("Accumulated local effect on RV")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle(f"How predictions respond to each input — {sym} (paper Fig. 6)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, save)


def fig_splits(splits: dict, start, end, events: dict | None = None, save=None, figsize=None):
    """Compact chart of the train / validation / test periods of each split scenario."""
    fig, bx = plt.subplots(figsize=figsize or (12, 0.62 * len(splits) + 1.3))
    shades = {"train": "#9ec5f4", "val": "#fab219", "test": "#e34948"}
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for i, (name, spec) in enumerate(splits.items()):
        d_tr, d_va = pd.Timestamp(spec[0]), pd.Timestamp(spec[1])
        excl = spec[2] if len(spec) > 2 else None
        y = len(splits) - 1 - i
        for seg, a, b in (("train", start, d_tr), ("val", d_tr, d_va), ("test", d_va, end)):
            bx.barh(y, b - a, left=a, height=0.62, color=shades[seg], edgecolor=SURFACE, linewidth=2)
        if excl:
            a, b = pd.Timestamp(excl[0]), pd.Timestamp(excl[1])
            bx.barh(y, b - a, left=a, height=0.62, color=SURFACE, edgecolor=MUTED, hatch="////", linewidth=0.8)
        bx.text(-0.008, y, name, transform=bx.get_yaxis_transform(), ha="right", va="center", fontsize=11, color=INK)
    for name, (d0, d1) in (events or {}).items():
        d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
        bx.axvspan(d0, d1, color=INK2, alpha=0.10, lw=0, zorder=0)
        bx.text(d0 + (d1 - d0) / 2, len(splits) - 0.45, name, ha="center", va="bottom", fontsize=9.5, color=INK2)
    bx.set_yticks([]); bx.set_ylim(-0.5, len(splits) - 0.1)
    bx.set_xlim(start, end)
    bx.grid(axis="y", visible=False)
    bx.spines["left"].set_visible(False)
    handles = [mpl.patches.Patch(color=c, label={"train": "Training", "val": "Validation",
                                                 "test": "Test (out-of-sample)"}[k]) for k, c in shades.items()]
    if any(len(v) > 2 and v[2] for v in splits.values()):
        handles.append(mpl.patches.Patch(facecolor=SURFACE, edgecolor=MUTED, hatch="////", label="Removed from estimation"))
    bx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(handles), fontsize=10)
    return _save(fig, save)


def fig_crosssection(preds, models=None, h=1, lo=0.6, hi=1.42, save=None, figsize=None):
    """One dot per stock: relative MSE vs HAR with RV lags only (grey) and all predictors (colour),
    our cross-sectional average (bar) and the paper's average (diamond); right: stocks beating HAR."""
    models = models or ["LogHAR", "EN", "RF", "GB", "NN2_ens"]
    T = {ds: V.per_stock_relative(preds, ds, h) for ds in ("M_HAR", "M_ALL")}
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=figsize or (12.5, 4.3))
    for i, m in enumerate(models):
        y0 = len(models) - 1 - i
        for ds, dy, col in (("M_HAR", 0.2, OTHER), ("M_ALL", -0.2, color(m))):
            v = T[ds][m].dropna()
            inside = (v >= lo) & (v <= hi)
            jit = rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(v[inside], y0 + dy + jit[inside.values], s=26, color=col, alpha=0.9,
                       edgecolor="white", lw=0.5, zorder=3)
            over = v[v > hi]
            if len(over):
                ax.scatter(hi, y0 + dy, marker=">", s=46, color=col, zorder=3)
                lab = f"{over.idxmax()} {over.max():.1f}" if len(over) == 1 else f"{len(over)} stocks up to {over.max():.1f}"
                ax.annotate(lab, (hi, y0 + dy), xytext=(-9, 0), textcoords="offset points", ha="right",
                            va="center", fontsize=8.5, color=INK2)
            ax.plot([v.mean()] * 2, [y0 + dy - 0.15, y0 + dy + 0.15], color=INK if ds == "M_ALL" else INK2,
                    lw=2.4, zorder=4)
            pap = V.PAPER.get((ds, h), {}).get(m)
            if pap is not None:
                ax.scatter(pap, y0 + dy, marker="D", s=46, facecolor="white", edgecolor=INK, lw=1.4, zorder=5)
        n0, n1 = int((T["M_HAR"][m] < 1).sum()), int((T["M_ALL"][m] < 1).sum())
        ax.text(hi + 0.045, y0, f"{n0} → {n1}", va="center", ha="left", fontsize=12, fontweight="bold",
                color=color(m), clip_on=False)
    ax.text(hi + 0.045, len(models) - 0.45, "stocks\nbeating HAR", va="bottom", ha="left", fontsize=8.5,
            color=INK2, clip_on=False)
    ax.axvline(1, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_yticks(range(len(models)), [NAME[m] for m in models][::-1], fontsize=11)
    ax.set_xlim(lo - 0.01, hi + 0.02); ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlabel(f"MSE relative to HAR, per stock ({H_NAME[h]} ahead)  ·  left of the dashed line = better than HAR",
                  fontsize=10)
    ax.grid(axis="y", visible=False)
    hd = [plt.Line2D([], [], marker="o", ls="", color=OTHER, ms=7, label="RV lags only (one dot = one stock)"),
          plt.Line2D([], [], marker="o", ls="", color=color("RF"), ms=7, label="All predictors"),
          plt.Line2D([], [], color=INK, lw=2.4, label="Our average"),
          plt.Line2D([], [], marker="D", ls="", mfc="white", mec=INK, ms=7, label="Paper average (2001–2017)")]
    ax.legend(handles=hd, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=9.5)
    return _save(fig, save)

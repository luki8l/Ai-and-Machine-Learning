"""Builds replication.ipynb from the cells below (run once: python build_notebook.py)."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))   # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))     # noqa: E731

md(r"""
# A machine learning approach to volatility forecasting: replication and extension to 2015–2026

**Paper:** Christensen, Siggaard & Veliyev (2022/2023), *A machine learning approach to volatility forecasting*, Journal of Financial Econometrics (arXiv:2601.13014).

**Research question.** The paper shows (DJIA stocks, 2001–2017) that off-the-shelf ML (regularization, random forest, neural nets) beats the HAR model family in forecasting realized variance, and that the edge grows with the information set and the forecast horizon. **Does this result hold out-of-sample in a new, non-overlapping market regime (2015–2026), using only open data?**

**Data.** 5-minute realized measures for the Dow 30 from **VOLARE** (Cipollini et al., 2026, arXiv:2602.19732, https://volare.unime.it), open-access and built from cleaned tick data, plus public macro series (FRED, Philadelphia Fed, Yahoo Finance).

**Design (as in the paper).** 70/10/20 train/validation/test split; HAR, HAR-X, LogHAR, LevHAR, SHAR, HARQ vs. ridge, lasso, elastic net, random forest, gradient boosting, neural nets; information sets $\mathcal M_{HAR}$ (RV lags only) and $\mathcal M_{ALL}$ (+ extra predictors); horizons 1, 5, 22 days; relative MSE + Diebold–Mariano tests; ALE-based variable importance.

All code lives in `volml.py`; this notebook runs it and produces every table and figure.
""")

code(r"""
# !pip install pandas numpy scipy scikit-learn matplotlib joblib openpyxl yfinance pyarrow
import os, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import volml as V

QUICK = True        # True: 8 stocks + lighter models for a first pass (~10-20 min). False: full run.
DRY_RUN = False     # True: simulated data, no downloads needed (pipeline test only)

# Data source: "volare" (preferred) | "alpaca" (5-min bars, 2016+) | "yahoo" (daily range proxy, 2001+) | "binance" (crypto)
# Fallback files are created with:  python fetch_data.py alpaca|yahoo|binance  (see README)
DATA_SOURCE = "alpaca"
PATHS = {"volare": "data/volare", "alpaca": "data/alpaca/realized.csv",
         "yahoo": "data/yahoo/realized.csv", "binance": "data/binance/realized.csv"}
cfg = V.Config(volare_path=PATHS[DATA_SOURCE])
if DATA_SOURCE == "yahoo":
    cfg.start = "2001-01-01"      # paper window + extension; set cfg.end = "2017-12-31" to mimic the paper sample
# per-stock implied vol from LSEG (python fetch_iv_lseg.py) is used automatically if present
cfg.iv_file = "data/lseg/iv.csv" if os.path.exists("data/lseg/iv.csv") else None
if QUICK and DATA_SOURCE != "binance":
    cfg.symbols = ["AAPL", "MSFT", "JPM", "KO", "BA", "CVX", "WMT", "IBM"]
    cfg.rf_trees, cfg.nn_seeds, cfg.nn_top, cfg.refit_trees = 200, 5, 3, 126
os.makedirs(cfg.results_dir, exist_ok=True)
cfg
""")

md(r"""
## 1. Data

**If VOLARE access is pending:** run `python fetch_data.py alpaca --key ... --secret ...` (free Alpaca paper account, true 5-min RV from 2016) or `python fetch_data.py yahoo` (no account, daily Garman–Klass proxy from 2001) and set `DATA_SOURCE` above accordingly. With Yahoo, HARQ and SHAR are skipped automatically (no intraday data).

**VOLARE (realized measures).** Go to https://volare.unime.it → *Download Data* → asset type *Stocks*, measure set with the 5-minute measures (`rv5`, `rq5`, `rsp5`, `rsn5`) plus open/close/volume. Put the downloaded file(s) into `data/volare/`. The loader matches column names case-insensitively and prints what it found; if a name differs, add it to `_ALIASES` in `volml.py`.

**Macro (downloaded automatically, cached in `data/cache/`).** VIX (FRED `VIXCLS`), economic policy uncertainty (FRED `USEPUINDXD`), 3-month T-bill (FRED `DTB3`, first-differenced), Hang Seng squared log return (Yahoo `^HSI`), ADS business conditions index (Philadelphia Fed).

**Firm-level extras.** Momentum (`M1W`, 5-day sum of open-to-close returns) and dollar volume (`DVOL`) are built from VOLARE. Model-free implied volatility (OptionMetrics) and earnings dates are not openly available: they can be added through `cfg.iv_file` / `cfg.ea_file` (CSV) if you get them, e.g. implied vols from LSEG Workspace.
""")

code(r"""
if DRY_RUN:
    panel, macro = V.simulate_panel(n_stocks=6, n_days=2800)
    panel = panel.rename(columns={"rv5": "rv", "rq5": "rq", "rsp5": "rsp", "rsn5": "rsn"})
    panel[["rv", "rsp", "rsn"]] *= 1e4; panel["rq"] *= 1e8
    cfg.symbols = sorted(panel.symbol.unique())
else:
    panel = V.load_volare(cfg.volare_path, symbols=None if DATA_SOURCE == "binance" else cfg.symbols)
    if DATA_SOURCE == "binance":
        cfg.symbols = sorted(panel.symbol.unique())
    macro = V.load_macro(cfg)

ea = pd.read_csv(cfg.ea_file, parse_dates=["date"]) if cfg.ea_file else None
iv = pd.read_csv(cfg.iv_file, parse_dates=["date"]) if cfg.iv_file else None
panel.groupby("symbol")["date"].agg(["min", "max", "count"]).head(30)
""")

md("### Descriptive statistics (cf. paper Table 1; realized volatility annualized in %)")
code(r"""
ann = lambda x: np.sqrt(252 * x)     # RV is stored in (daily %)^2
desc = []
for s, g in panel.groupby("symbol"):
    f = V.build_features(g, macro, cfg, None if ea is None else ea[ea.symbol == s],
                         None if iv is None else iv[iv.symbol == s])
    t = pd.DataFrame({"RVD": ann(f["RVD"]), "RVW": ann(f["RVW"]), "RVM": ann(f["RVM"])})
    for c in V.extra_cols(f):
        t[c] = f[c]
    desc.append(t.describe().T.assign(skew=t.skew(), kurt=t.kurt() + 3))
desc = pd.concat(desc).groupby(level=0).mean()[["mean", "50%", "max", "min", "std", "skew", "kurt"]]
desc.round(2)
""")

md("## 2. Forecasting (rolling out-of-sample, all models, both information sets, h = 1, 5, 22)")
code(r"""
tag = f"{DATA_SOURCE}_{'quick' if QUICK else 'full'}{'_iv' if iv is not None else ''}"
cfg.results_dir = os.path.join("results", tag)            # one folder per run variant
os.makedirs(cfg.results_dir, exist_ok=True)
pkl = os.path.join(cfg.results_dir, "preds.pkl")
print("results ->", cfg.results_dir)
if os.path.exists(pkl):
    preds, info = pd.read_pickle(pkl)
else:
    t0 = time.time()
    preds, info = V.run_all(cfg, panel, macro, ea, iv)
    pd.to_pickle((preds, info), pkl)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
s0 = next(iter(info))
print("sample split:", info[s0][("M_HAR", 1)]["n"])
print("tuned hyperparameters (example):", {k: v for k, v in info[s0][("M_ALL", 1)].items() if k != "n"})
""")

md(r"""
## 3. Main results: relative MSE vs. HAR (cf. paper Tables 2–7)

Entries are cross-sectional averages of MSE(model)/MSE(HAR); < 1 means the model beats the HAR. Stars: one-sided Diebold–Mariano test (model better than HAR) rejected for more than 50% of stocks at 10% (\*), 5% (\*\*), 1% (\*\*\*). The paper's numbers are shown next to ours.
""")
code(r"""
# HAR rows of the paper's Tables 2-7 (NN2_ens here = best 5 of 10 nets; paper NN2^10 = best 10 of 100)
PAPER = {
 ("M_HAR", 1):  dict(LogHAR=.995, LevHAR=1.073, SHAR=1.009, HARQ=1.059, RR=1.000, LA=1.003, EN=.999, BG=1.147, RF=1.020, GB=1.054, NN2_1=.966, NN2_ens=.958),
 ("M_ALL", 1):  {"HAR-X": .966, "LogHAR": .901, "LevHAR": 1.003, "SHAR": 1.080, "HARQ": 1.289, "RR": .919, "LA": .936, "EN": .916, "BG": .961, "RF": .901, "GB": .962, "NN2_1": .893, "NN2_ens": .885},
 ("M_HAR", 5):  dict(LogHAR=1.091, LevHAR=1.089, SHAR=.937, HARQ=1.043, RR=1.019, LA=1.024, EN=1.019, BG=1.056, RF=.959, GB=1.164, NN2_1=.947, NN2_ens=.937),
 ("M_ALL", 5):  {"HAR-X": 1.037, "LogHAR": .970, "LevHAR": 1.076, "SHAR": .913, "HARQ": 1.459, "RR": .954, "LA": .958, "EN": .925, "BG": .840, "RF": .796, "GB": .930, "NN2_1": .865, "NN2_ens": .846},
 ("M_HAR", 22): dict(LogHAR=1.052, LevHAR=1.030, SHAR=1.034, HARQ=.994, RR=1.100, LA=1.107, EN=1.111, BG=1.171, RF=1.019, GB=1.219, NN2_1=.989, NN2_ens=.958),
 ("M_ALL", 22): {"HAR-X": 1.290, "LogHAR": .919, "LevHAR": 1.328, "SHAR": 1.312, "HARQ": 1.569, "RR": 1.111, "LA": 1.378, "EN": 1.107, "BG": .621, "RF": .604, "GB": .829, "NN2_1": .816, "NN2_ens": .786},
}
summary = {}
for ds in cfg.datasets:
    for h in cfg.horizons:
        ratio, txt = V.relative_table(preds, ds, h)
        txt.to_csv(f"{cfg.results_dir}/table_{ds}_h{h}.csv")       # full pairwise matrix
        summary[(ds, f"h={h}", "ours")] = txt.loc["HAR"]
        summary[(ds, f"h={h}", "paper")] = pd.Series(PAPER[(ds, h)]).map("{:.3f}".format)
summary = pd.DataFrame(summary).drop(index="HAR", errors="ignore").fillna("")
summary.to_csv(f"{cfg.results_dir}/summary_vs_paper.csv")
summary
""")

md("Full pairwise matrix (paper layout: row = benchmark, column = model), example $\\mathcal M_{ALL}$, h = 1:")
code(r"""V.relative_table(preds, "M_ALL", 1)[1]""")

md("### Robustness: QLIKE loss (robust to noise in the RV proxy, Patton 2011)")
code(r"""
ql = {(ds, f"h={h}"): V.relative_table(preds, ds, h, loss="qlike")[1].loc["HAR"]
      for ds in cfg.datasets for h in cfg.horizons}
pd.DataFrame(ql).drop(index="HAR").fillna("")
""")

md(r"""
### Robustness: how much does one episode drive the result?

MSE on realized variance in levels is dominated by a few extreme days. In our test window the tariff shock around 2 April 2025 is such an episode. We report the share of the HAR's squared error that falls into March–April 2025 and the relative MSEs without forecast origins in that window.
""")
code(r"""
EPISODE = ("2025-03-01", "2025-04-30")
for h in cfg.horizons:
    share, frac = V.error_share(preds, EPISODE, "M_HAR", h)
    print(f"h={h}: {share:.0%} of HAR squared error comes from {frac:.0%} of test days")
ex = {}
for ds in cfg.datasets:
    for h in cfg.horizons:
        ex[(ds, f"h={h}", "all")] = V.relative_table(preds, ds, h)[1].loc["HAR"]
        ex[(ds, f"h={h}", "ex episode")] = V.relative_table(preds, ds, h, exclude=EPISODE)[1].loc["HAR"]
ex = pd.DataFrame(ex).drop(index="HAR").fillna("")
ex.to_csv(f"{cfg.results_dir}/robustness_ex_episode.csv")
ex
""")

md("## 4. Cross-section of relative MSE (cf. paper Figure 3)")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, ds in zip(axes, cfg.datasets):
    R = V.per_stock_relative(preds, ds, 1).drop(columns="HAR")
    ax.boxplot(R.values, tick_labels=R.columns)
    ax.axhline(1, color="grey", lw=1, ls="--"); ax.set_title(f"{ds}, h=1"); ax.tick_params(axis="x", rotation=60)
axes[0].set_ylabel("MSE relative to HAR")
plt.tight_layout(); plt.savefig(f"{cfg.results_dir}/fig3_boxplot.png", dpi=200); plt.show()
""")

md("## 5. Accuracy across the volatility distribution (cf. paper Figure 5)")
code(r"""
best_nn = [c for c in preds[s0][("M_ALL", 1)].columns if c.endswith("_ens")][0]
dec = V.decile_table(preds, "M_ALL", 1, models=["HAR-X", "LogHAR", "EN", "RF", best_nn])
ax = dec.loc[[0, 1, 5, 8, 9]].T.plot.bar(figsize=(10, 4)); ax.axhline(1, color="grey", ls="--", lw=1)
ax.set_ylabel("MSE relative to HAR"); ax.legend(title="RV decile (0 = calmest)")
plt.tight_layout(); plt.savefig(f"{cfg.results_dir}/fig5_deciles.png", dpi=200); plt.show()
dec.round(3)
""")

md(r"""
## 6. Opening the black box: ALE and variable importance (cf. paper Figures 6–7)

Models are fitted on the initial training set ($\mathcal M_{ALL}$, h = 1); importance = standard deviation of the centered accumulated local effect, normalized to sum to one, averaged across stocks.
""")
code(r"""
d_tr, d_va = V.split_dates(panel[panel.date >= cfg.start], cfg)
vi_syms = cfg.symbols[:8]                       # VI is the slowest part; 8 stocks are enough for the figure
vis, fitted_aapl = [], None
for s in vi_syms:
    vi, fitted, tr = V.vi_for_stock(panel[panel.symbol == s], macro, cfg, d_tr, d_va,
                                    ea=None if ea is None else ea[ea.symbol == s],
                                    iv=None if iv is None else iv[iv.symbol == s])
    vis.append(vi)
    if s == vi_syms[0]:
        fitted_aapl, tr_aapl = fitted, tr
VI = sum(vis) / len(vis)
fig, axes = plt.subplots(1, VI.shape[1], figsize=(14, 4), sharex=True)
for ax, m in zip(axes, VI.columns):
    VI[m].sort_values().plot.barh(ax=ax); ax.set_title(m)
plt.tight_layout(); plt.savefig(f"{cfg.results_dir}/fig7_variable_importance.png", dpi=200); plt.show()
VI.round(3)
""")
code(r"""
# ALE curves for the first stock (paper Figure 6 uses Apple)
cols = [c for c in ["RVD", "RVW", "IV", "VIX", "M1W"] if c in tr_aapl]
fig, axes = plt.subplots(1, len(cols), figsize=(16, 3.8))
for ax, c in zip(axes, cols):
    for name, m in fitted_aapl.items():
        e, a, _ = V.ale(m, tr_aapl, c)
        ax.plot(e, a, label=name)
    ax.set_title(f"ALE: {c}"); ax.set_xlabel(c)
axes[0].legend(); plt.tight_layout(); plt.savefig(f"{cfg.results_dir}/fig6_ale.png", dpi=200); plt.show()
""")

md(r"""
## 7. Deviations from the original paper (state these on a slide)

| Paper | This replication | Why |
|---|---|---|
| NYSE TAQ, 29 DJIA stocks, 2001–2017 | VOLARE 5-min RV, Dow 30, 2015–latest | TAQ is not available to us; VOLARE is open and built from cleaned tick data with the same 5-min RV estimator |
| Model-free IV per stock (OptionMetrics) | 30-day ATM implied vol from LSEG Workspace (if `fetch_iv_lseg.py` was run), otherwise omitted; VIX keeps the market-implied-vol channel | OptionMetrics not available to us |
| Earnings announcement dummy | Omitted (optional CSV hook) | No clean open source for 2016–2026 |
| Tuning and RF/GB re-estimated daily | OLS daily; regularization every 22 days, trees every 63 days (configurable) | Compute budget on laptops; hyperparameters tuned once on the validation block |
| NN: L-ReLU, dropout, 100 seeds → best 10 | scikit-learn MLP, ReLU, L2 + early stopping, 10 seeds → best 5 | Paper reports ReLU makes no material difference (fn. 7) |
| Adaptive / post lasso, bagging, MCS, VaR | Not run by default (BG available via `cfg.models`) | Scope for one week |

Additional: QLIKE as a second loss function; explicit no-look-ahead rule for multi-day targets.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(nb, "replication.ipynb")
print("wrote replication.ipynb with", len(C), "cells")

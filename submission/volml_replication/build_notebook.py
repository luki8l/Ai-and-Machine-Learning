"""Builds replication.ipynb (run: python build_notebook.py)."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))   # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))     # noqa: E731

# --------------------------------------------------------------------------- #
md(r"""
# A machine learning approach to volatility forecasting: replication and extension to 2016–2026

**Paper:** Christensen, Siggaard & Veliyev, *A machine learning approach to volatility forecasting*, Journal of Financial Econometrics (arXiv:2601.13014). Original sample: 29 DJIA stocks, 2001–2017.

**Our question.** Does the paper's result (off-the-shelf ML beats the HAR family, more so with extra predictors and at longer horizons) hold out-of-sample in a new regime (2016–2026) with open data?

**Hypotheses.** H1: with RV lags only (M_HAR) ML ≈ HAR. H2: with extra predictors (M_ALL) ML beats HAR. H3: the ML edge grows with the horizon (1 day → 1 week → 1 month).

**Data.** 5-minute realized measures for the Dow 30 from **VOLARE** (Cipollini et al. 2026, arXiv:2602.19732; cleaned tick data), 30-day ATM implied volatility per stock from **LSEG Workspace**, macro series from FRED, the Philadelphia Fed and Yahoo Finance.

**Three sample splits** (the paper itself shows that results depend on whether a crisis is in the training or the test sample, Appendix A.1):

| Scenario | Training | Validation | Test | COVID (2020) is … |
|---|---|---|---|---|
| **Baseline** | 70% (2016–2023) | 10% | 20% (2024–2026) | in the training sample |
| **COVID in test** | 2016 – Jun 2019 | Jul – Dec 2019 | 2020 – 2026 | out-of-sample (as in the paper's 1,000-day robustness check) |
| **Without 2020** | as baseline | as baseline | as baseline | removed from every estimation window |

All code is in `volml.py` (models, evaluation) and `volplots.py` (figures). Figures for the slides are written to `results/<run>/slides/`, tables to `results/<run>/slides_tables.xlsx`.
""")

code(r"""
# !pip install pandas numpy scipy scikit-learn matplotlib joblib openpyxl yfinance pyarrow
import os, time, copy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
import volml as V, volplots as P

QUICK = False       # True: 8 stocks + lighter models for a quick check
DRY_RUN = False     # True: simulated data (pipeline test only, no downloads)

# Data source: "volare" (main) | "alpaca" (5-min bars, robustness) | "yahoo" | "binance"
DATA_SOURCE = "volare"
PATHS = {"volare": "data/volare", "alpaca": "data/alpaca/realized.csv",
         "yahoo": "data/yahoo/realized.csv", "binance": "data/binance/realized.csv"}

cfg = V.Config(volare_path=PATHS[DATA_SOURCE])
cfg.start = "2016-01-01"                                  # common start for all sources and the LSEG IV
cfg.iv_file = "data/lseg/iv.csv" if os.path.exists("data/lseg/iv.csv") else None
if QUICK and DATA_SOURCE != "binance":
    cfg.symbols = ["AAPL", "MSFT", "JPM", "KO", "BA", "CVX", "WMT", "IBM"]
    cfg.rf_trees, cfg.nn_seeds, cfg.nn_top, cfg.refit_trees = 200, 5, 3, 126

# The three split scenarios (see table above). Remove names from RUN to skip a scenario.
SCENARIOS = {
    "baseline":   dict(label="Baseline",      split=None,                         train_exclude=None),
    "covid_test": dict(label="COVID in test", split=("2019-06-28", "2019-12-31"), train_exclude=None),
    "ex2020":     dict(label="Without 2020",  split=None,                         train_exclude=("2020-01-01", "2020-12-31")),
}
RUN = ["baseline", "covid_test", "ex2020"]

EVENTS = {"COVID": ("2020-02-20", "2020-05-29"), "2022 bear market": ("2022-01-03", "2022-12-30"),
          "Tariff shock": ("2025-04-01", "2025-04-30")}
TARIFF = EVENTS["Tariff shock"]
P.setup_style()
cfg
""")

# --------------------------------------------------------------------------- #
md(r"""
## 1. Data

VOLARE: `data/volare/realized_variance_stocks.csv` (the loader uses `rv5`, `rq5`, `rsp5`, `rsn5`, `open_price`, `close_price`, `volume`). LSEG IV: `data/lseg/iv.csv` (from `lseg/LSEG_IV_Download.xlsx` → `python lseg/convert_lseg_excel.py …`). Macro series are downloaded once and cached in `data/cache/`.
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
panel = panel[panel["date"] >= cfg.start].reset_index(drop=True)

ea = pd.read_csv(cfg.ea_file, parse_dates=["date"]) if cfg.ea_file else None
iv = pd.read_csv(cfg.iv_file, parse_dates=["date"]) if cfg.iv_file else None
print("implied vol:", "none" if iv is None else f"{iv.symbol.nunique()} stocks, {iv.date.min().date()} -> {iv.date.max().date()}")
TAG = f"{DATA_SOURCE}_{'quick' if QUICK else 'full'}{'_iv' if iv is not None else ''}"
OUT = os.path.join("results", TAG); FIG = os.path.join(OUT, "slides"); os.makedirs(FIG, exist_ok=True)
TABLES = {}                                   # everything collected here is exported to Excel at the end
panel.groupby("symbol")["date"].agg(["min", "max", "count"]).T
""")

md("### Table 1 — descriptive statistics (realized volatility annualized in %)")
code(r"""
ann = lambda x: np.sqrt(252 * x)
desc = []
for s, g in panel.groupby("symbol"):
    f = V.build_features(g, macro, cfg, None, None if iv is None else iv[iv.symbol == s])
    t = pd.DataFrame({"RVD": ann(f["RVD"]), "RVW": ann(f["RVW"]), "RVM": ann(f["RVM"])})
    for c in V.extra_cols(f):
        t[c] = f[c]
    desc.append(t.describe().T.assign(skew=t.skew(), kurt=t.kurt() + 3))
desc = pd.concat(desc).groupby(level=0).mean()[["mean", "50%", "max", "min", "std", "skew", "kurt"]].round(2)
TABLES["T1_descriptives"] = desc
desc
""")

code(r"""
def scenario_cfg(name):
    c = copy.deepcopy(cfg)
    c.split, c.train_exclude = SCENARIOS[name]["split"], SCENARIOS[name]["train_exclude"]
    return c

SPLITS = {}
for name in SCENARIOS:
    d_tr, d_va = V.split_dates(panel, scenario_cfg(name))
    SPLITS[SCENARIOS[name]["label"]] = (d_tr, d_va, SCENARIOS[name]["train_exclude"])
    print(f"{SCENARIOS[name]['label']:14s} train <= {d_tr.date()}  val <= {d_va.date()}  test after"
          + (f"   (removed: {SCENARIOS[name]['train_exclude']})" if SCENARIOS[name]["train_exclude"] else ""))
P.fig_rv_timeline(panel, SPLITS, EVENTS, save=f"{FIG}/01_rv_timeline_splits.png"); plt.show()
P.fig_splits(SPLITS, panel["date"].min(), panel["date"].max(), EVENTS, save=f"{FIG}/02_splits.png"); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 2. Forecasting

Rolling out-of-sample forecasts for every model, both information sets and h = 1, 5, 22, for each scenario. Each run is cached in `results/<run>/<scenario>/preds.pkl`; re-running the notebook only computes what is missing. On a 12-core laptop roughly: baseline ≈ 25 min, without 2020 ≈ 25 min, COVID in test ≈ 35–45 min (longer test period, shorter training windows).
""")
code(r"""
def run_scenario(name):
    c = scenario_cfg(name)
    path = os.path.join(OUT, name, "preds.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    res = V.run_all(c, panel, macro, ea, iv)
    pd.to_pickle(res, path)
    print(f"{name}: done in {(time.time() - t0) / 60:.1f} min")
    return res

RESULTS, INFO = {}, {}
for name in RUN:
    RESULTS[name], INFO[name] = run_scenario(name)
    s0 = next(iter(INFO[name]))
    print(f"{SCENARIOS[name]['label']:14s}", INFO[name][s0][("M_HAR", 1)]["n"])
preds = RESULTS["baseline"]                    # sections 3-5 use the baseline split
""")

md(r"""
### Our extension: random forest on log RV

The random forest fails at the one-month horizon (section 3). Our fix, in the spirit of the LogHAR: the same forest (500 trees, mtry = p/3, min node 5, re-estimated every 63 days on the same rolling window) is fitted to log RV and mapped back to levels with Duan's smearing factor from its out-of-bag residuals (estimation window only, no look-ahead). Baseline split, all predictors. Cached in `results/<run>/baseline/logrf.csv`; computing it takes roughly 5–15 min.
""")
code(r"""
path = os.path.join(OUT, "baseline", "logrf.csv")
if os.path.exists(path):
    extra = V.load_extra(path)
else:
    t0 = time.time()
    extra = V.run_extra(scenario_cfg("baseline"), panel, macro, only=("LogRF",), datasets=("M_ALL",), ea=ea, iv=iv)
    V.save_extra(extra, path)
    print(f"LogRF: done in {(time.time() - t0) / 60:.1f} min")
RESULTS["baseline"] = V.merge_models(RESULTS["baseline"], extra)
preds = RESULTS["baseline"]
logrf = pd.DataFrame({P.H_NAME[h]: V.vs_benchmark(preds, "M_ALL", h).loc[["RF", "LogRF", "LogHAR"], "txt"]
                      for h in cfg.horizons})
logrf.index = ["Random forest", "Random forest on log RV", "LogHAR"]
TABLES["T12_logrf_extension"] = logrf
logrf
""")

# --------------------------------------------------------------------------- #
md(r"""
## 3. Main result (baseline split): replication vs. the paper

Entries: cross-sectional average of MSE(model)/MSE(HAR); below 1 = beats the HAR. Stars: one-sided Diebold–Mariano test rejected for more than half of the stocks at 10% (\*), 5% (\*\*), 1% (\*\*\*), as in the paper's Tables 2–7.
""")
code(r"""
summary = {}
for ds in cfg.datasets:
    for h in cfg.horizons:
        summary[(ds, f"h={h}", "ours")] = V.vs_benchmark(preds, ds, h)["txt"]
        summary[(ds, f"h={h}", "paper")] = pd.Series(V.PAPER[(ds, h)]).map("{:.3f}".format)
summary = pd.DataFrame(summary).reindex(P.ORDER).fillna("")
summary.index = [P.NAME[m] for m in summary.index]
TABLES["T2_vs_paper_baseline"] = summary
summary
""")
code(r"""
P.fig_heatmap_vs_paper(V.add_combination(preds), "M_ALL", extensions=("LogRF", "RF+LogHAR"),
                       save=f"{FIG}/02_heatmap_vs_paper_M_ALL.png"); plt.show()
P.fig_heatmap_vs_paper(preds, "M_HAR", save=f"{FIG}/03_heatmap_vs_paper_M_HAR.png"); plt.show()
""")
code(r"""
P.fig_dumbbell(preds, "M_ALL", models=P.KEY_MODELS + ["GB", "LogRF"], save=f"{FIG}/04_dumbbell_vs_paper_M_ALL.png"); plt.show()
""")
md("Full pairwise matrix in the paper's layout (row = benchmark, column = model), $\\mathcal M_{ALL}$, 1 day:")
code(r"""
pairwise = V.relative_table(preds, "M_ALL", 1)[1]
TABLES["T3_pairwise_M_ALL_h1"] = pairwise
pairwise
""")
md("### Robustness: QLIKE loss and win rates")
code(r"""
ql = pd.DataFrame({(ds, f"h={h}"): V.vs_benchmark(preds, ds, h, loss="qlike")["txt"]
                   for ds in cfg.datasets for h in cfg.horizons}).reindex(P.ORDER).fillna("")
TABLES["T4_qlike_baseline"] = ql
display(ql)
P.fig_winrate(preds, "M_ALL", save=f"{FIG}/05_winrate_M_ALL.png"); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 4. Where do the gains come from?

MSE on realized variance in levels is dominated by a few extreme days. The figures below show how much of the result is driven by the tariff shock in April 2025.
""")
code(r"""
for h in cfg.horizons:
    share, frac = V.error_share(preds, TARIFF, "M_HAR", h)
    print(f"h={h:2d}: {share:.0%} of the HAR's squared error comes from {frac:.0%} of test days (Apr 2025)")
P.fig_cumgain(preds, "M_ALL", 1, events=EVENTS, save=f"{FIG}/06_cumulative_gain_M_ALL_h1.png"); plt.show()
P.fig_cumgain(preds, "M_ALL", 22, events=EVENTS, save=f"{FIG}/07_cumulative_gain_M_ALL_h22.png"); plt.show()
""")
code(r"""
ex = {}
for ds in cfg.datasets:
    for h in cfg.horizons:
        ex[(ds, f"h={h}", "all")] = V.vs_benchmark(preds, ds, h)["txt"]
        ex[(ds, f"h={h}", "ex Apr 2025")] = V.vs_benchmark(preds, ds, h, exclude=("2025-03-01", "2025-04-30"))["txt"]
ex = pd.DataFrame(ex).reindex(P.ORDER).fillna("")
TABLES["T5_ex_tariff_shock"] = ex
ex
""")
code(r"""
P.fig_forecast_zoom(preds, "AAPL" if "AAPL" in preds else next(iter(preds)), "M_ALL", 1, ["HAR", "LogHAR", "RF"],
                    "2025-02-15", "2025-06-30", save=f"{FIG}/08_forecast_zoom_tariff.png"); plt.show()
dec = V.decile_table(preds, "M_ALL", 1, models=P.KEY_MODELS)
TABLES["T6_deciles_M_ALL_h1"] = dec.round(3)
P.fig_deciles(dec, save=f"{FIG}/09_deciles_M_ALL_h1.png"); plt.show()
P.fig_boxplot(preds, "M_ALL", 1, save=f"{FIG}/10_boxplot_M_ALL_h1.png"); plt.show()
# slide version: one dot per stock, RV lags only vs all predictors, our vs the paper's average
P.fig_crosssection(preds, save=f"{FIG}/10b_crosssection_h1.png"); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
## 5. Split sensitivity: does it matter where COVID sits?

Same models, same data, three splits. Read the figure per model: a flat line = robust to the split; a model that only wins in one scenario owes its result to the sample design.

**Insanity filter.** In "COVID in test", models estimated on calm 2016–2019 data extrapolate wildly in March 2020 (LogHAR's relative MSE runs into the billions). For the scenario comparison every h-day forecast is therefore clipped to the range of h-day average RV realized up to the forecast day (expanding min/max, no look-ahead), identically for all models incl. HAR. It touches less than 0.5% of forecasts. The unfiltered table is shown too.
""")
code(r"""
labels = {n: SCENARIOS[n]["label"] for n in RESULTS}
RESULTS_F = {n: V.insanity_filter(r, panel) for n, r in RESULTS.items()}
# equal-weight combination of random forest and LogHAR (weights fixed at 1/2, nothing estimated)
RESULTS_F = {n: V.add_combination(r, ("RF", "LogHAR"), "RF+LogHAR") for n, r in RESULTS_F.items()}
def scen_table(res):
    out = {}
    for ds in cfg.datasets:
        for h in cfg.horizons:
            for n, r in res.items():
                out[(ds, f"h={h}", labels[n])] = V.vs_benchmark(r, ds, h)["txt"]
    out = pd.DataFrame(out).reindex(P.ORDER + ["RF+LogHAR"]).fillna("")
    out.index = [P.NAME[m] for m in out.index]
    return out
scen = scen_table(RESULTS_F)
TABLES["T7_scenarios"] = scen
TABLES["T7b_scenarios_unfiltered"] = scen_table(RESULTS)
print("Filtered (used for slides):"); display(scen)
print("Unfiltered:"); display(TABLES["T7b_scenarios_unfiltered"])
named = {labels[n]: r for n, r in RESULTS_F.items()}
P.fig_scenarios(named, "M_ALL", models=P.KEY_MODELS + ["RF+LogHAR"], save=f"{FIG}/11_scenarios_M_ALL.png"); plt.show()
P.fig_scenarios(named, "M_HAR", save=f"{FIG}/12_scenarios_M_HAR.png"); plt.show()
""")
code(r"""
# how concentrated is the HAR's error in each scenario? (COVID window and tariff shock)
rows = []
for n, r in RESULTS.items():
    for wname, w in (("COVID Feb-May 2020", EVENTS["COVID"]), ("Tariff shock Apr 2025", TARIFF)):
        share, frac = V.error_share(r, w, "M_HAR", 1)
        if frac > 0:
            rows.append({"scenario": labels[n], "window": wname, "share of HAR SSE": f"{share:.0%}",
                         "share of test days": f"{frac:.0%}"})
conc = pd.DataFrame(rows); TABLES["T8_error_concentration"] = conc; conc
""")
code(r"""
if "covid_test" in RESULTS:
    P.fig_cumgain(RESULTS["covid_test"], "M_ALL", 1, events=EVENTS,
                  save=f"{FIG}/13_cumulative_gain_covid_test_h1.png"); plt.show()
""")
md(r"""
### Why the linear models suffer: COVID flips the HAR coefficients

OLS in levels minimizes squared errors, so the ~70 COVID days dominate the fit. We estimate HAR and HAR-X on the baseline in-sample window with and without 2020.
""")
code(r"""
_, d_va_base = V.split_dates(panel, scenario_cfg("baseline"))
coef = pd.concat([
    V.linear_coefficients(panel, macro, cfg, d_va_base, iv=iv).assign(sample="with COVID"),
    V.linear_coefficients(panel, macro, cfg, d_va_base, exclude=SCENARIOS["ex2020"]["train_exclude"], iv=iv).assign(sample="without 2020"),
])
P.fig_coefficients(coef, save=f"{FIG}/14_har_coefficients_covid.png"); plt.show()
coef_tab = coef.groupby(["model", "sample"]).agg(
    RVD_median=("RVD", "median"), RVW_median=("RVW", "median"), RVM_median=("RVM", "median"),
    share_RVM_negative=("RVM", lambda x: (x < 0).mean())).round(3)
TABLES["T9_har_coefficients"] = coef_tab
coef_tab
""")
md("### Key numbers (computed, for the speaker notes)")
code(r"""
def line(n, ds, h, m):
    r = V.vs_benchmark(RESULTS[n], ds, h).loc[m]
    return f"{P.NAME[m]:22s} rel. MSE {r['rel']:.3f}{r['stars']:3s}  beats HAR for {r['win']:.0%} of stocks"
for n in RESULTS:
    print(f"\n=== {labels[n]} ===")
    for ds in cfg.datasets:
        for h in cfg.horizons:
            best = V.rel_mse(RESULTS[n], ds, h).idxmin()
            print(f"{ds} h={h:2d}: best = {line(n, ds, h, best)}")
""")

# --------------------------------------------------------------------------- #
md(r"""
## 6. Opening the black box: variable importance and ALE (baseline, paper Figures 6–7)

Models fitted on the initial training sample ($\mathcal M_{ALL}$, h = 1). Importance = standard deviation of the centered accumulated local effect, normalized to sum to one, averaged across stocks.
""")
code(r"""
c_base = scenario_cfg("baseline")
d_tr, d_va = V.split_dates(panel, c_base)
vis, fitted0, tr0, sym0 = [], None, None, None
for s in cfg.symbols[:8]:
    if s not in set(panel.symbol):
        continue
    vi, fitted, tr = V.vi_for_stock(panel[panel.symbol == s], macro, c_base, d_tr, d_va,
                                    ea=None if ea is None else ea[ea.symbol == s],
                                    iv=None if iv is None else iv[iv.symbol == s])
    vis.append(vi)
    if fitted0 is None:
        fitted0, tr0, sym0 = fitted, tr, s
VI = pd.concat(vis).groupby(level=0).mean()
TABLES["T10_variable_importance"] = VI.round(3)
P.fig_vi(VI, save=f"{FIG}/15_variable_importance.png"); plt.show()
cols = [c for c in ["RVD", "RVW", "IV", "VIX", "M1W"] if c in tr0]
P.fig_ale(fitted0, tr0, cols, sym0, save=f"{FIG}/16_ale_{sym0}.png"); plt.show()
VI.round(3)
""")

# --------------------------------------------------------------------------- #
md(r"""
## 7. Robustness: data source (VOLARE vs. Alpaca 5-minute bars)

Runs automatically if a baseline run on the other source exists (e.g. `results/alpaca_full_iv/baseline/preds.pkl`; run the notebook once with `DATA_SOURCE = "alpaca"`).
""")
code(r"""
other = "alpaca" if DATA_SOURCE == "volare" else "volare"
alt = os.path.join("results", TAG.replace(DATA_SOURCE, other, 1), "baseline", "preds.pkl")
if os.path.exists(alt):
    preds_alt, _ = pd.read_pickle(alt)
    src = pd.DataFrame({(f"h={h}", s): V.rel_mse(p, "M_ALL", h).round(3)
                        for h in cfg.horizons for s, p in ((DATA_SOURCE, preds), (other, preds_alt))})
    src = src.reindex(P.ORDER); TABLES["T11_data_source"] = src; display(src)
else:
    print("no run on", other, "found at", alt)
""")

# --------------------------------------------------------------------------- #
md("## 8. Export")
code(r"""
xlsx = os.path.join(OUT, "slides_tables.xlsx")
with pd.ExcelWriter(xlsx) as w:
    for name, t in TABLES.items():
        t.to_excel(w, sheet_name=name[:31])
print("tables ->", xlsx)
print("figures ->", FIG); print("\n".join(sorted(os.listdir(FIG))))
""")

md(r"""
## 9. Deviations from the original paper

| Paper | This replication | Why |
|---|---|---|
| NYSE TAQ, 29 DJIA stocks, 2001–2017 | VOLARE 5-min RV (cleaned tick data), Dow 30, 2016–2026 | TAQ not available; VOLARE is open and uses the same 5-min RV estimator |
| Model-free IV (OptionMetrics) | 30-day ATM implied vol (LSEG Workspace), call/put average | OptionMetrics not available |
| Earnings announcement dummy | omitted | no clean open source for 2016–2026 |
| One split (70/10/20) + 1,000/2,000-day robustness | three splits: baseline, COVID in test, without 2020 | the COVID year dominates OLS in levels |
| Tuning / trees re-estimated daily | OLS daily; regularization every 22 days, trees every 63 days | compute budget |
| NN: Leaky ReLU, dropout, best 10 of 100 nets | ReLU, L2 + early stopping, best 5 of 10 nets | paper fn. 7: ReLU no material difference |
| Adaptive/post lasso, bagging, MCS, VaR | not run | scope |

Added: QLIKE loss, win rates, error-concentration analysis, cumulative loss differentials, data-source robustness (Alpaca).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(nb, "replication.ipynb")
print("wrote replication.ipynb with", len(C), "cells")

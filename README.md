# ML vs. HAR volatility forecasting: replication 2015–2026 with open data

Replication and time extension of **Christensen, Siggaard & Veliyev, "A machine learning approach to volatility forecasting"** (Journal of Financial Econometrics; arXiv:2601.13014) for the course *AI and Machine Learning* (Uni Liechtenstein). Project presentation: **23.10.**

## 1. The pitch in three sentences

1. The paper finds (DJIA stocks, 2001–2017) that off-the-shelf ML beats the HAR benchmark for realized-variance forecasts, mostly when extra predictors are added and at longer horizons (random forest: about 40% lower MSE than HAR at the monthly horizon).
2. We test whether this holds **out-of-sample in a new, non-overlapping regime (2015–2026)** using only **open data**.
3. Hypotheses: **H1** ML ≈ HAR with RV lags only (M_HAR); **H2** ML > HAR once extra predictors are added (M_ALL); **H3** the ML edge increases with the horizon (1 → 5 → 22 days).

## 2. Data: how we solved the data problem

| Need | Paper | We use | Access |
|---|---|---|---|
| 5-min realized variance, quarticity, semivariances | NYSE TAQ (WRDS) | **VOLARE** (Cipollini et al. 2026, arXiv:2602.19732): Dow 30, from 2015-01-02, cleaned tick data, updated monthly | Free, https://volare.unime.it → Download Data → Stocks |
| VIX, T-bill, EPU | various | FRED `VIXCLS`, `DTB3`, `USEPUINDXD` | automatic |
| ADS business conditions | Philly Fed | Philly Fed xlsx | automatic |
| Hang Seng squared return | – | Yahoo `^HSI` (yfinance) | automatic |
| Momentum, dollar volume | TAQ | built from VOLARE open/close/volume | automatic |
| Implied vol per stock, earnings dates | OptionMetrics, – | **optional** CSV (`cfg.iv_file`, `cfg.ea_file`), e.g. from LSEG Workspace | manual |

Why not the alternatives: the Oxford-Man Realized Library was discontinued in mid-2022; LSEG intraday bars only go back about one year; Yahoo intraday only 60 days.
**Fallback** if VOLARE fails: `volml.realized_from_bars()` turns any 5-min bars (e.g. Alpaca free API, US stocks since 2016; Binance public klines for crypto) into the same daily measures.

### Fallbacks while VOLARE access is pending (`fetch_data.py`)

All fallbacks write the same daily format, so only `DATA_SOURCE` in the notebook's first cell changes.

| Plan | Command | Data | Account | Notes |
|---|---|---|---|---|
| **A (recommended)** | `python fetch_data.py alpaca --key KEY --secret SECRET` | True 5-min RV, RQ, semivariances, Dow 30, 2016 → today | Free Alpaca paper account (email sign-up, open to non-US residents) → API keys in the dashboard | Closest to the paper; ~10–20 min download, resumable |
| **B (zero sign-up)** | `python fetch_data.py yahoo` | Daily Garman–Klass range variance, Dow 30, 2001 → today | None | Noisier proxy; HARQ/SHAR dropped automatically; but covers the paper's own 2001–2017 window (set `cfg.end = "2017-12-31"`) plus the extension |
| C (extension) | `python fetch_data.py binance` | True 5-min RV for BTC, ETH, SOL, BNB, XRP, 2018 → today | None | Different asset class; good as an add-on slide, not as the replication |

Suggested order: start Plan B right now (works instantly), set up the Alpaca account in parallel and switch to Plan A once its download is done; switch to VOLARE if access arrives in time. The code is identical for all three.

### Optional: per-stock implied volatility from LSEG Workspace (uni licence)

The paper's IV variable (OptionMetrics model-free IV) is one of its three most important predictors (Figure 7). The closest substitute we can get is LSEG's 30-day at-the-money implied vol:

**Without code (Excel):** open `lseg/LSEG_IV_Download.xlsx` with the LSEG Workspace Excel add-in, refresh, save as values, then `python lseg/convert_lseg_excel.py LSEG_IV_values.xlsx` (instructions on the first sheet).

**With code:**

1. Open LSEG Workspace and log in with the uni account (the script uses the desktop session).
2. `pip install lseg-data`
3. `python fetch_iv_lseg.py` → writes `data/lseg/iv.csv` (date, symbol, iv)
4. Re-run the notebook: the IV file is detected automatically, results go to a separate folder `results/<source>_<quick|full>_iv/`, so you can compare runs with and without IV.

RICs are `<TICKER>ATMIV.U` (e.g. `AAPLATMIV.U`), fields `TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORCALLOPTIONS` / `...PUTOPTIONS` (taken from the LSEG developer forum; not tested by us). If a stock returns no data, look the RIC up in Workspace. IV is only used if it covers at least 90% of a stock's days.

### Data cleaning (Alpaca bars)

Raw consolidated bars contain occasional bad prints (e.g. AMGN on 25 Oct 2016 with an RV ~5,600× its median; the NYSE opening-auction malfunction on 24 Jan 2023 hit VZ, V, MCD, WMT, NKE). `realized_from_bars` therefore removes isolated one-bar spikes before computing RV (bar-level version of the Barndorff-Nielsen et al. 2009 rule used by the Oxford-Man library and VOLARE: drop a price if it deviates from the median of its ±5-bar neighbourhood by more than 10 mean absolute deviations and by more than 1%; real jumps that persist are kept).

If you downloaded before this change: `python fetch_data.py rebuild` recomputes `data/alpaca/realized.csv` from the cached bars (no keys, no download, a few minutes), keeps the old file as `realized_uncleaned.csv` and writes `cleaning_log.csv` with every flagged day.

**Action item #1 (today):** one person downloads the VOLARE stock file(s) into `data/volare/` and runs the first two notebook cells. The loader prints the columns it found; if a name is not recognized, add it to `_ALIASES` in `volml.py`.

## 3. How to run

```bash
pip install pandas numpy scipy scikit-learn matplotlib joblib openpyxl yfinance pyarrow jupyter
jupyter notebook replication.ipynb
```

- `DRY_RUN = True` runs everything on simulated data (no downloads), to check the setup.
- `QUICK = True`: 8 stocks, lighter models, roughly 10–20 min on a laptop. Do this first.
- `QUICK = False`: all 30 stocks, paper settings. Measured on one core: about 70 s per stock × horizon × information set → about 3.5 CPU hours in total, i.e. 30–60 min on a 4–8-core laptop (runs in parallel over stocks). Results are cached in `results/`.

Outputs in `results/<source>_<quick|full>[_iv]/`: `summary_vs_paper.csv` (our HAR-relative MSEs next to the paper's), the full pairwise tables (`table_<set>_h<h>.csv`, paper layout with DM-test stars), and Figures 3, 5, 6, 7 as PNG.

## 4. What is replicated, and deviations

Same as the paper: 70/10/20 split; targets = average RV over the next 1/5/22 days; HAR, HAR-X, LogHAR (Jensen bias correction), LevHAR, SHAR, HARQ (insanity filter); ridge, lasso, elastic net tuned on the validation block; random forest with Breiman–Cutler defaults (500 trees, mtry = p/3, min node 5); gradient boosting tuned over depth/learning rate/trees; pyramid neural net trained once (fixed window) with Adam (lr 0.001), early stopping and seed ensembles; negative forecasts replaced by the in-sample minimum; relative MSE averaged across stocks; one-sided Diebold–Mariano tests; ALE-based variable importance.

| Deviation | Reason |
|---|---|
| Sample 2015–2026 instead of 2001–2017; Dow 30 as listed in VOLARE | Data availability; this is the point of the extension |
| No per-stock implied vol / earnings dummy by default | Not openly available (CSV hooks exist) |
| Regularized models re-estimated every 22 days, trees every 63 days (paper: daily); hyperparameters tuned once | Compute budget; configurable in `Config` |
| NN in scikit-learn: ReLU, L2 + early stopping, best 5 of 10 seeds (paper: Leaky ReLU, dropout, best 10 of 100) | Simpler and reproducible; paper fn. 7 says ReLU makes no material difference |
| Not run: adaptive/post lasso, Model Confidence Set, VaR application | Scope for one week (bagging available via `cfg.models`) |
| **Added:** QLIKE loss; explicit no-look-ahead rule for multi-day targets | Robustness (Patton 2011) |

## 5. Slide outline (20 min)

1. Title + question (1 min)
2. Why volatility forecasting matters; HAR as the benchmark (2 min)
3. The original paper in one slide: setup + key results (2 min)
4. Our hypotheses H1–H3 (1 min)
5. Data: the replication problem and VOLARE as the open solution; sample, split dates (2 min)
6. Models and evaluation design (one diagram: rolling window, no look-ahead) (2 min)
7. Results 1: M_HAR, h = 1 — ours vs. paper (2 min)
8. Results 2: M_ALL, h = 1/5/22 — ours vs. paper (3 min)
9. Cross-section (box plot) + volatility deciles: where does ML win? (2 min)
10. Black box: variable importance (ALE) (1.5 min)
11. Deviations and limitations (1 min)
12. Conclusion: does the result survive the new regime? Next steps (0.5 min)

## 6. Proposed split for five people and timeline

| Person | Task |
|---|---|
| A | Data: VOLARE download, loader check, macro data, descriptive table |
| B | Run QUICK, then FULL; sanity checks; keep `results/` |
| C | Results tables vs. paper; interpretation of H1–H3 |
| D | Figures (box plot, deciles, ALE/VI); robustness (QLIKE) |
| E | Slides and storyline; deviations slide; rehearsal lead |

- **Fri 16.10 – Sat 17.10:** data download, DRY_RUN and QUICK run
- **Sun 18.10 – Mon 19.10:** FULL run, tables and figures
- **Tue 20.10 – Wed 21.10:** interpretation, slides
- **Thu 22.10:** rehearsal, freeze code + data for submission
- **Fri 23.10:** presentation; submit notebook, `volml.py`, data files

## Files

- `replication.ipynb` – the notebook to run and submit
- `volml.py` – all code (data, features, models, evaluation, ALE)
- `fetch_data.py` – fallback data downloads (Alpaca / Yahoo / Binance)
- `fetch_iv_lseg.py` – optional per-stock implied vol from LSEG Workspace
- `build_notebook.py` – regenerates the notebook from source

Cite VOLARE as: Cipollini, Cruciani, Gallo, Insana, Otranto & Spagnolo (2026), *VOLatility Archive for Realized Estimates (VOLARE)*, arXiv:2602.19732.

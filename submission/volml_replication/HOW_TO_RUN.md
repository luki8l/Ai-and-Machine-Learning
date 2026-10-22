# ML vs. HAR volatility forecasting – replication of Christensen, Siggaard & Veliyev (2016–2026)

Group project, course *AI and Machine Learning*, University of Liechtenstein.

## Run it (about 10–20 minutes)

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook replication.ipynb    # then: Kernel -> Restart & Run All
```

- All data are included (`data/`), so no download or API key is needed.
- The out-of-sample forecasts of all models are included in `results/volare_full_iv/` (`preds.pkl` per split, `baseline/logrf.csv`). The notebook loads them and recomputes all tables, tests and figures. Only the variable-importance section refits models (a few minutes).
- `replication.ipynb` is saved with all outputs, so it can also be read without running it.
- `python audit_checks.py` runs 25 sanity checks (targets, no look-ahead, Diebold–Mariano test, OLS, LogHAR, ALE, COVID window, insanity filter, combination, log-RF).

## Re-estimate everything from scratch (optional)

Delete `results/volare_full_iv/*/preds.pkl` and `results/volare_full_iv/baseline/logrf.csv`, then run the notebook again. Every model is then trained again by our code (rolling windows, three splits). On an 8–12-core laptop this took about 35 min (baseline), 45 min (COVID in test), 25 min (without 2020) and 10 min (random forest on log RV). The forecasts are pickled with pandas 2.3; use the versions in `requirements.txt`.

## Files

| File | Content |
|---|---|
| `replication.ipynb` | Main notebook: data, all models, three splits, tables, figures |
| `volml.py` | Data loading, features, models, rolling estimation, evaluation (DM test, ALE) |
| `volplots.py` | Figures used on the slides |
| `audit_checks.py` | Sanity checks |
| `data/volare/` | VOLARE 5-min realized measures, Dow 30 (Cipollini et al. 2026, arXiv:2602.19732, free download) |
| `data/lseg/iv.csv` | 30-day ATM implied volatility per stock (LSEG Workspace, university licence; for course use only) |
| `data/cache/` | VIX, 3-month T-bill, EPU (FRED), ADS index (Philadelphia Fed), Hang Seng (Yahoo) |
| `results/volare_full_iv/` | Cached forecasts, slide figures (`slides/`), all tables (`slides_tables.xlsx`) |
| `fetch_iv_lseg.py`, `lseg/`, `fetch_data.py`, `build_notebook.py` | How the IV data were downloaded; alternative data sources (Alpaca/Yahoo); script that generates the notebook |
| `presentation/` | Final slides |

Details on method, deviations from the paper and our extensions: `README.md`.

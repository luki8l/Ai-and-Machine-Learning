"""
volml.py - Replication (and time extension) of
    Christensen, Siggaard & Veliyev (2022/2023),
    "A machine learning approach to volatility forecasting",
    Journal of Financial Econometrics.

Data: VOLARE (https://volare.unime.it) - open-access 5-minute realized measures
for the Dow 30 constituents from 2015-01-02 onwards, plus public macro series
(FRED, Philadelphia Fed, Yahoo Finance).

Everything a forecast at day t uses is known at the close of day t. Rolling
re-estimation only uses rows whose targets are fully realized at the forecast
origin (no look-ahead, also for the 5- and 22-day horizons).
"""
from __future__ import annotations

import copy
import glob
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Dow 30 members marked in VOLARE (Table A.1 of the VOLARE paper, arXiv:2602.19732)
DOW30_VOLARE = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]


@dataclass
class Config:
    volare_path: str = "data/volare"          # file or folder with VOLARE stock data
    symbols: list = field(default_factory=lambda: list(DOW30_VOLARE))
    start: str = "2015-01-01"
    end: str | None = None                     # None = use everything available
    horizons: tuple = (1, 5, 22)
    datasets: tuple = ("M_HAR", "M_ALL")
    train_frac: float = 0.70                   # paper: 70 / 10 / 20
    val_frac: float = 0.10
    # estimation schedule (paper: daily rolling for everything except NN)
    refit_linear: int = 1                      # OLS HAR family: daily re-estimation
    refit_reg: int = 22                        # ridge / lasso / EN: re-estimate every N days
    refit_trees: int = 63                      # RF / BG / GB: re-estimate every N days
    # models
    models: tuple = ("HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ",
                     "RR", "LA", "EN", "RF", "GB", "NN")
    rf_trees: int = 500                        # Breiman-Cutler defaults (paper Table 15)
    rf_min_leaf: int = 5
    nn_archs: tuple = ((4, 2),)                # NN2 of the paper; add (2,), (8,4,2), (16,8,4,2)
    nn_seeds: int = 10                         # paper: 100 nets, keep best 10
    nn_top: int = 5
    nn_max_epochs: int = 500
    nn_patience: int = 50                      # paper: 100
    nn_lr: float = 1e-3
    nn_l2: float = 1e-4
    nn_batch: int = 32
    n_jobs: int = -1                           # parallel over stocks
    cache_dir: str = "data/cache"
    results_dir: str = "results"
    offline_macro: bool = False                # True = skip downloads (e.g. for tests)
    ea_file: str | None = None                 # optional CSV: symbol,date of earnings announcements
    iv_file: str | None = None                 # optional CSV: date,symbol,iv (e.g. from LSEG)
    seed: int = 42


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #

_ALIASES = {
    "date": ["date", "day", "datetime", "timestamp", "time", "trade_date"],
    "symbol": ["symbol", "ticker", "asset", "sym", "name"],
    "rv": ["rv5", "rv_5", "rv5min", "rv5_ss", "rv"],
    "rq": ["rq5", "rq_5", "rq5_ss", "rq"],
    "rsp": ["rsp5", "rsp_5", "rsp5_ss", "rs_p5", "rsp", "rs_plus", "rv_pos"],
    "rsn": ["rsn5", "rsn_5", "rsn5_ss", "rs_n5", "rsn", "rs_minus", "rv_neg"],
    "open": ["open", "o", "open_price"],
    "close": ["close", "c", "close_price"],
    "volume": ["volume", "vol", "v"],
}


def _pick(cols, names):
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def _read_any(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def load_volare(path: str, symbols=None) -> pd.DataFrame:
    """Load VOLARE stock realized measures (single file or folder of files).

    Returns a long frame: date, symbol, rv, rq, rsp, rsn, open, close, volume
    (missing optional columns are simply absent). Column names are matched
    case-insensitively; check the printout once after downloading.
    """
    files = [path] if os.path.isfile(path) else sorted(
        f for ext in ("csv", "parquet", "txt", "xlsx")
        for f in glob.glob(os.path.join(path, f"**/*.{ext}"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No VOLARE files found under '{path}'.")
    frames = []
    for f in files:
        raw = _read_any(f)
        ren = {}
        for key, names in _ALIASES.items():
            c = _pick(raw.columns, names)
            if c is not None:
                ren[c] = key
        df = raw.rename(columns=ren)
        if "symbol" not in df.columns:      # one file per symbol
            df["symbol"] = os.path.splitext(os.path.basename(f))[0].split("_")[0].upper()
        missing = [k for k in ("date", "rv") if k not in df.columns]
        if missing:
            raise ValueError(f"{f}: could not find columns {missing}. "
                             f"Columns present: {list(raw.columns)}. "
                             "Add the right name to _ALIASES in volml.py.")
        keep = [k for k in _ALIASES if k in df.columns]
        frames.append(df[keep])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    if symbols:
        df = df[df["symbol"].isin(symbols)]
    num = [c for c in df.columns if c not in ("date", "symbol")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    df = df[df["rv"] > 0]
    # scale to (daily %)^2 for readability; does not affect relative MSEs
    if df["rv"].median() < 0.01:
        for c in ("rv", "rsp", "rsn"):
            if c in df:
                df[c] = df[c] * 1e4
        if "rq" in df:
            df["rq"] = df["rq"] * 1e8
    print(f"VOLARE: {df['symbol'].nunique()} symbols, {df['date'].min().date()} "
          f"-> {df['date'].max().date()}, columns: {[c for c in df.columns]}")
    return df.reset_index(drop=True)


def realized_from_bars(bars: pd.DataFrame, session=("09:30", "15:55")) -> pd.DataFrame:
    """Fallback if VOLARE is unavailable: turn intraday bars (e.g. 5-min bars from
    the Alpaca free API, or Binance klines) into VOLARE-like daily measures.

    `bars`: columns timestamp (exchange local time, bar START time), symbol, open,
    close, volume. Default session keeps the 78 regular-hours 5-min bars 09:30-15:55 ET.
    Use session=None for 24/7 markets (crypto, UTC days).
    Returns date, symbol, rv, rq, rsp, rsn, open, close, volume (raw units).
    """
    b = bars.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    b = b.sort_values(["symbol", "timestamp"]).set_index("timestamp")
    if session:
        b = b.between_time(*session)
    b["date"] = b.index.normalize()
    out = []
    for (sym, d), g in b.groupby(["symbol", "date"]):
        r = np.diff(np.log(np.r_[g["open"].iloc[0], g["close"].to_numpy()]))
        m = len(r)
        if m < 30:
            continue
        out.append({"date": d, "symbol": sym, "rv": np.sum(r ** 2),
                    "rq": m / 3 * np.sum(r ** 4), "rsp": np.sum(r[r > 0] ** 2),
                    "rsn": np.sum(r[r < 0] ** 2), "open": g["open"].iloc[0],
                    "close": g["close"].iloc[-1], "volume": g["volume"].sum()})
    return pd.DataFrame(out)


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ADS_URL = ("https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
           "ads/ADS_Index_Most_Current_Vintage.xlsx")


def _fred(series_id):
    raw = pd.read_csv(FRED_URL.format(series_id))
    s = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(raw.iloc[:, 0]), name=series_id)
    return s.dropna()


def _ads():
    raw = pd.read_excel(ADS_URL)
    d = raw.iloc[:, 0].astype(str).str.replace(":", "-", regex=False)
    return pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").values,
                     index=pd.to_datetime(d, errors="coerce"), name="ADS").dropna()


def _hsi():
    import yfinance as yf  # pip install yfinance
    px = yf.download("^HSI", start="2014-01-01", progress=False, auto_adjust=False)["Close"]
    px = px.squeeze()
    return (np.log(px).diff() ** 2 * 1e4).rename("HSI").dropna()


def load_macro(cfg: Config) -> pd.DataFrame:
    """Public macro predictors of the paper's M_ALL set.

    VIX (FRED VIXCLS), EPU (FRED USEPUINDXD, daily), US3M (FRED DTB3, first
    difference), HSI (squared Hang Seng log return, Yahoo), ADS (Philadelphia Fed).
    Each download is cached; failures are reported and the variable is skipped.
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)
    getters = {"VIX": lambda: _fred("VIXCLS"), "EPU": lambda: _fred("USEPUINDXD"),
               "US3M": lambda: _fred("DTB3"), "HSI": _hsi, "ADS": _ads}
    out = {}
    for name, fn in getters.items():
        cache = os.path.join(cfg.cache_dir, f"{name}.csv")
        try:
            if os.path.exists(cache):
                s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
            elif cfg.offline_macro:
                continue
            else:
                s = fn()
                s.to_csv(cache)
            out[name] = s.rename(name)
            print(f"macro {name}: {s.index.min().date()} -> {s.index.max().date()}")
        except Exception as e:  # noqa: BLE001
            print(f"macro {name}: skipped ({type(e).__name__}: {e})")
    if not out:
        return pd.DataFrame()
    return pd.concat(out.values(), axis=1).sort_index()


# --------------------------------------------------------------------------- #
# 2. Features (all known at the close of day t)
# --------------------------------------------------------------------------- #

BASE = ["RVD", "RVW", "RVM"]


def build_features(stock: pd.DataFrame, macro: pd.DataFrame, cfg: Config,
                   ea: pd.DataFrame | None = None, iv: pd.DataFrame | None = None):
    s = stock.set_index("date").sort_index()
    f = pd.DataFrame(index=s.index)
    f["RVD"] = s["rv"]
    f["RVW"] = s["rv"].rolling(5).mean()
    f["RVM"] = s["rv"].rolling(22).mean()
    if "rq" in s:
        f["RQRVD"] = np.sqrt(s["rq"]) * s["rv"]               # HARQ interaction
    if {"rsp", "rsn"} <= set(s.columns):
        f["RVP"], f["RVN"] = s["rsp"], s["rsn"]
    if {"open", "close"} <= set(s.columns):
        # open-to-close return: split-safe with unadjusted prices
        r = np.log(s["close"] / s["open"]) * 100
        f["RNEG_D"] = np.minimum(r, 0)
        f["RNEG_W"] = np.minimum(r.rolling(5).mean(), 0)
        f["RNEG_M"] = np.minimum(r.rolling(22).mean(), 0)
        f["M1W"] = r.rolling(5).sum()                          # 1-week momentum
        f["RET"] = r                                           # for VaR / plots
    if {"volume", "close"} <= set(s.columns):
        f["DVOL"] = np.log(s["volume"] * s["close"]).diff()    # paper: diff of log $VOL
    if ea is not None and len(ea):
        dates = set(pd.to_datetime(ea["date"]))
        # indicator for an announcement on day t+1 (dates are public in advance)
        nxt = pd.Series(f.index, index=f.index).shift(-1)
        f["EA"] = nxt.isin(dates).astype(float)
    if iv is not None and len(iv):
        f["IV"] = iv.set_index("date")["iv"].reindex(f.index).ffill(limit=5)
    if macro is not None and len(macro):
        m = macro.copy()
        if "US3M" in m:
            m["US3M"] = m["US3M"].diff()
        # EPU and ADS are published with a delay -> lag one day to be safe
        for c in ("EPU", "ADS"):
            if c in m:
                m[c] = m[c].shift(1)
        m = m.reindex(m.index.union(f.index)).ffill(limit=5).reindex(f.index)
        f = f.join(m)
    for h in cfg.horizons:                                     # targets
        f[f"y{h}"] = s["rv"][::-1].rolling(h).mean()[::-1].shift(-1)
    return f


def extra_cols(f: pd.DataFrame):
    cand = ["IV", "EA", "M1W", "DVOL", "VIX", "EPU", "US3M", "HSI", "ADS"]
    return [c for c in cand if c in f.columns and f[c].notna().mean() > 0.9]


# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #

class OLS:
    """HAR-type linear regression. log=True gives the LogHAR (Jensen-corrected)."""

    def __init__(self, cols, log_cols=(), log_target=False, clip=False):
        self.cols, self.log_cols, self.log_target, self.clip = cols, set(log_cols), log_target, clip

    def _X(self, X):
        Z = X[self.cols].to_numpy(dtype=float).copy()
        for j, c in enumerate(self.cols):
            if c in self.log_cols:
                Z[:, j] = np.log(np.maximum(Z[:, j], 1e-10))
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, X, y):
        y = np.asarray(y, float)
        t = np.log(y) if self.log_target else y
        Z = self._X(X)
        self.b, *_ = np.linalg.lstsq(Z, t, rcond=None)
        res = t - Z @ self.b
        self.s2 = res.var()
        self.lo, self.hi = y.min(), y.max()
        return self

    def predict(self, X):
        p = self._X(X) @ self.b
        if self.log_target:
            p = np.exp(p + 0.5 * self.s2)
        if self.clip:                                  # insanity filter (BPQ 2016)
            p = np.clip(p, self.lo, self.hi)
        return p


class SK:
    """Thin wrapper: standardize X (on the estimation window), fit an sklearn model."""

    def __init__(self, cols, make, scale=True):
        self.cols, self.make, self.scale = cols, make, scale

    def fit(self, X, y):
        Z = X[self.cols].to_numpy(float)
        self.mu, self.sd = (Z.mean(0), Z.std(0) + 1e-12) if self.scale else (0, 1)
        self.m = self.make().fit((Z - self.mu) / self.sd, np.asarray(y, float))
        return self

    def predict(self, X):
        return self.m.predict((X[self.cols].to_numpy(float) - self.mu) / self.sd)


class NNEnsemble:
    """Feed-forward net, Adam (lr 1e-3), early stopping on the validation block,
    ensemble over random seeds (best `top` of `seeds` by validation MSE)."""

    def __init__(self, cols, hidden, cfg: Config):
        self.cols, self.hidden, self.cfg = cols, hidden, cfg

    def _one(self, seed, Xtr, ytr, Xva, yva):
        from sklearn.neural_network import MLPRegressor
        c = self.cfg
        m = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", solver="adam",
                         learning_rate_init=c.nn_lr, alpha=c.nn_l2, batch_size=c.nn_batch,
                         random_state=seed, shuffle=True)
        best, best_state, wait = np.inf, None, 0
        for _ in range(c.nn_max_epochs):
            m.partial_fit(Xtr, ytr)
            v = np.mean((m.predict(Xva) - yva) ** 2)
            if v < best - 1e-9:
                best, wait = v, 0
                best_state = (copy.deepcopy(m.coefs_), copy.deepcopy(m.intercepts_))
            else:
                wait += 1
                if wait >= c.nn_patience:
                    break
        m.coefs_, m.intercepts_ = best_state
        return best, m

    def fit(self, Xtr, ytr, Xva, yva):
        Z = Xtr[self.cols].to_numpy(float)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-12
        self.ym, self.ys = float(np.mean(ytr)), float(np.std(ytr))
        tz = lambda X: (X[self.cols].to_numpy(float) - self.mu) / self.sd  # noqa: E731
        ty = lambda y: (np.asarray(y, float) - self.ym) / self.ys          # noqa: E731
        fits = [self._one(self.cfg.seed + k, tz(Xtr), ty(ytr), tz(Xva), ty(yva))
                for k in range(self.cfg.nn_seeds)]
        fits.sort(key=lambda t: t[0])
        self.nets = [m for _, m in fits[: self.cfg.nn_top]]
        self.lo = float(np.min(ytr))
        return self

    def predict(self, X, k=None):
        Z = (X[self.cols].to_numpy(float) - self.mu) / self.sd
        nets = self.nets if k is None else self.nets[:k]
        p = np.mean([n.predict(Z) for n in nets], axis=0) * self.ys + self.ym
        return np.maximum(p, self.lo)


def _make_reg(kind, alpha, l1):
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    if kind == "RR":
        return lambda: Ridge(alpha=alpha)
    if kind == "LA":
        return lambda: Lasso(alpha=alpha, max_iter=20000)
    return lambda: ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000)


REG_GRID = {
    "RR": [(a, None) for a in np.logspace(-3, 5, 40)],
    "LA": [(a, None) for a in np.logspace(-5, 2, 40)],
    "EN": [(a, l1) for l1 in (0.1, 0.3, 0.5, 0.7, 0.9) for a in np.logspace(-5, 2, 30)],
}


def linear_spec(name, ds, f):
    """Columns of each HAR-type model; in M_ALL every HAR model gets the extra
    predictors (distributed-lag 'X' version, as in the paper)."""
    ex = extra_cols(f) if ds == "M_ALL" else []
    has = set(f.columns)
    if name in ("HAR", "HAR-X"):
        return OLS(BASE + (ex if name == "HAR-X" else []))
    if name == "LogHAR":
        logc = BASE + [c for c in ("VIX", "IV") if c in ex]
        return OLS(BASE + ex, log_cols=logc, log_target=True)
    if name == "LevHAR" and {"RNEG_D", "RNEG_W", "RNEG_M"} <= has:
        return OLS(BASE + ["RNEG_D", "RNEG_W", "RNEG_M"] + ex)
    if name == "SHAR" and {"RVP", "RVN"} <= has:
        return OLS(["RVP", "RVN", "RVW", "RVM"] + ex)
    if name == "HARQ" and "RQRVD" in has:
        return OLS(["RVD", "RQRVD", "RVW", "RVM"] + ex, clip=True)
    return None


def ml_cols(ds, f):
    return BASE + (extra_cols(f) if ds == "M_ALL" else [])


# --------------------------------------------------------------------------- #
# 4. Forecasting engine (one stock, one horizon, one information set)
# --------------------------------------------------------------------------- #

def _rolling(model_factory, D, ycol, start, h, window, every):
    """Rolling-window forecasts for rows start..n-1. At origin i only rows
    s <= i-h are used (their h-day target is realized by day i)."""
    n = len(D)
    pred = np.full(n - start, np.nan)
    for i in range(start, n, every):
        hi = i - h + 1
        lo = max(0, hi - window)
        m = model_factory().fit(D.iloc[lo:hi], D[ycol].iloc[lo:hi])
        floor = D[ycol].iloc[lo:hi].min()
        j = min(i + every, n)
        p = np.asarray(m.predict(D.iloc[i:j]), float)
        bad = ~np.isfinite(p) | (p <= 0)                # paper: replace negatives
        p[bad] = floor
        pred[i - start:j - start] = p
    return pred


def run_one(f, ds, h, cfg: Config, d_tr, d_va):
    ycol = f"y{h}"
    models = list(cfg.models)
    if ds == "M_HAR" and "HAR-X" in models:
        models.remove("HAR-X")                          # identical to HAR there
    specs = {m: linear_spec(m, ds, f) for m in models
             if m in ("HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ")}
    specs = {k: v for k, v in specs.items() if v is not None}
    cols = sorted(set(sum([s.cols for s in specs.values()], [])) | set(ml_cols(ds, f)))
    D = f[cols + [ycol]].dropna()
    dates = D.index
    n_tr = int((dates <= d_tr).sum())
    n_va = int(((dates > d_tr) & (dates <= d_va)).sum())
    start = n_tr + n_va
    if n_tr < 300 or len(D) - start < 50:
        return None, {}
    y = D[ycol]
    tr_fit = slice(0, n_tr - h + 1)                     # embargo: targets end before val
    va = slice(n_tr, n_tr + n_va)
    out = pd.DataFrame({"y": y.iloc[start:]}, index=dates[start:])
    info = {}

    # (a) HAR family: OLS, rolling window = train+val, daily re-estimation
    for name, spec in specs.items():
        fac = lambda spec=spec: copy.deepcopy(spec)  # noqa: E731
        out[name] = _rolling(fac, D, ycol, start, h, n_tr + n_va, cfg.refit_linear)

    mc = ml_cols(ds, f)
    # (b) regularization: tune on validation, then rolling (window = train)
    for kind in ("RR", "LA", "EN"):
        if kind not in models:
            continue
        best = (np.inf, None)
        for a, l1 in REG_GRID[kind]:
            m = SK(mc, _make_reg(kind, a, l1)).fit(D.iloc[tr_fit], y.iloc[tr_fit])
            e = np.mean((m.predict(D.iloc[va]) - y.iloc[va]) ** 2)
            if e < best[0]:
                best = (e, (a, l1))
        a, l1 = best[1]
        info[kind] = {"alpha": a, "l1_ratio": l1}
        fac = lambda kind=kind, a=a, l1=l1: SK(mc, _make_reg(kind, a, l1))  # noqa: E731
        out[kind] = _rolling(fac, D, ycol, start, h, n_tr, cfg.refit_reg)

    # (c) trees
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    if "RF" in models or "BG" in models:
        for name, mf in (("RF", 1 / 3), ("BG", 1.0)):
            if name not in models:
                continue
            fac = lambda mf=mf: SK(mc, lambda: RandomForestRegressor(  # noqa: E731
                n_estimators=cfg.rf_trees, max_features=mf, min_samples_leaf=cfg.rf_min_leaf,
                n_jobs=1, random_state=cfg.seed), scale=False)
            out[name] = _rolling(fac, D, ycol, start, h, n_tr + n_va, cfg.refit_trees)
    if "GB" in models:
        best = (np.inf, None)
        for depth in (1, 2):
            for lr in (0.01, 0.1):
                g = GradientBoostingRegressor(max_depth=depth, learning_rate=lr, n_estimators=500,
                                              random_state=cfg.seed)
                g.fit(D.iloc[tr_fit][mc].to_numpy(), y.iloc[tr_fit])
                for k, p in enumerate(g.staged_predict(D.iloc[va][mc].to_numpy()), 1):
                    if k % 50 == 0:
                        e = np.mean((p - y.iloc[va].to_numpy()) ** 2)
                        if e < best[0]:
                            best = (e, (depth, lr, k))
        depth, lr, nt = best[1]
        info["GB"] = {"depth": depth, "lr": lr, "trees": nt}
        fac = lambda: SK(mc, lambda: GradientBoostingRegressor(  # noqa: E731
            max_depth=depth, learning_rate=lr, n_estimators=nt, random_state=cfg.seed), scale=False)
        out["GB"] = _rolling(fac, D, ycol, start, h, n_tr, cfg.refit_trees)

    # (d) neural nets: fixed window (fit once on train, early stopping on val)
    if "NN" in models:
        for arch in cfg.nn_archs:
            tag = f"NN{len(arch)}"
            nn = NNEnsemble(mc, arch, cfg).fit(D.iloc[tr_fit], y.iloc[tr_fit], D.iloc[va], y.iloc[va])
            Xte = D.iloc[start:]
            out[f"{tag}_1"] = nn.predict(Xte, k=1)
            out[f"{tag}_ens"] = nn.predict(Xte)
    info["n"] = {"train": n_tr, "val": n_va, "test": len(D) - start,
                 "test_start": str(dates[start].date()), "test_end": str(dates[-1].date())}
    return out, info


def run_stock(sym, stock, macro, cfg, d_tr, d_va, ea=None, iv=None):
    f = build_features(stock, macro, cfg, ea, iv)
    res, info = {}, {}
    for ds in cfg.datasets:
        for h in cfg.horizons:
            r, i = run_one(f, ds, h, cfg, d_tr, d_va)
            if r is not None:
                res[(ds, h)], info[(ds, h)] = r, i
    return sym, res, info


def split_dates(panel: pd.DataFrame, cfg: Config):
    d = np.sort(panel["date"].unique())
    d = d[22:]                                           # first month lost to lags
    return pd.Timestamp(d[int(len(d) * cfg.train_frac) - 1]), \
        pd.Timestamp(d[int(len(d) * (cfg.train_frac + cfg.val_frac)) - 1])


def run_all(cfg: Config, panel: pd.DataFrame, macro: pd.DataFrame, ea=None, iv=None):
    from joblib import Parallel, delayed
    panel = panel[(panel["date"] >= cfg.start)]
    if cfg.end:
        panel = panel[panel["date"] <= cfg.end]
    d_tr, d_va = split_dates(panel, cfg)
    print(f"train <= {d_tr.date()} < val <= {d_va.date()} < test")
    syms = [s for s in cfg.symbols if s in set(panel["symbol"])]
    jobs = (delayed(run_stock)(
        s, panel[panel["symbol"] == s], macro, cfg, d_tr, d_va,
        None if ea is None else ea[ea["symbol"] == s],
        None if iv is None else iv[iv["symbol"] == s]) for s in syms)
    out = Parallel(n_jobs=cfg.n_jobs, verbose=5)(jobs)
    preds = {s: r for s, r, _ in out}
    info = {s: i for s, _, i in out}
    return preds, info


# --------------------------------------------------------------------------- #
# 5. Evaluation
# --------------------------------------------------------------------------- #

def qlike(y, p):
    return y / p - np.log(y / p) - 1


def dm_test(e_bench, e_model, h=1):
    """One-sided Diebold-Mariano: H1 = model has lower expected loss than bench.
    HAC variance (Newey-West, h-1 lags) + Harvey-Leybourne-Newbold correction."""
    d = np.asarray(e_bench) - np.asarray(e_model)
    d = d[np.isfinite(d)]
    T = len(d)
    dc = d - d.mean()
    lag = max(h - 1, 0)
    v = dc @ dc / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v / T)
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return stat, 1 - stats.t.cdf(stat, df=T - 1)


def loss_frames(preds, ds, h, loss="mse"):
    """Per-stock loss series for every model -> {sym: DataFrame(date x model)}."""
    out = {}
    for s, r in preds.items():
        if (ds, h) not in r:
            continue
        P = r[(ds, h)]
        y = P["y"].to_numpy()[:, None]
        M = P.drop(columns="y")
        L = (M.to_numpy() - y) ** 2 if loss == "mse" else qlike(y, M.to_numpy())
        out[s] = pd.DataFrame(L, index=P.index, columns=M.columns)
    return out


def relative_table(preds, ds, h, loss="mse"):
    """Paper Table 2-7 layout: entry (row=benchmark, col=model) = cross-sectional
    average of loss(model)/loss(benchmark). Stars: DM test rejected for >50% of
    stocks at 10% (*), 5% (**), 1% (***)."""
    L = loss_frames(preds, ds, h, loss)
    models = list(next(iter(L.values())).columns)
    ratio = pd.DataFrame(np.nan, index=models, columns=models)
    txt = pd.DataFrame("-", index=models, columns=models)
    for b in models:
        for m in models:
            if b == m:
                continue
            rs, ps = [], []
            for s, df in L.items():
                rs.append(df[m].mean() / df[b].mean())
                ps.append(dm_test(df[b], df[m], h)[1])
            ratio.loc[b, m] = np.mean(rs)
            ps = np.array(ps)
            star = ("***" if np.mean(ps < .01) > .5 else "**" if np.mean(ps < .05) > .5
                    else "*" if np.mean(ps < .10) > .5 else "")
            txt.loc[b, m] = f"{np.mean(rs):.3f}{star}"
    return ratio, txt


def per_stock_relative(preds, ds, h, bench="HAR", loss="mse"):
    L = loss_frames(preds, ds, h, loss)
    return pd.DataFrame({s: df.mean() / df[bench].mean() for s, df in L.items()}).T


def decile_table(preds, ds="M_ALL", h=1, bench="HAR", models=None, q=10):
    """Paper Figure 5: relative MSE within deciles of realized RV in the test set."""
    rows = []
    for s, r in preds.items():
        if (ds, h) not in r:
            continue
        P = r[(ds, h)]
        P = P.assign(bin=pd.qcut(P["y"], q, labels=False))
        for b, g in P.groupby("bin"):
            e = (g.drop(columns=["y", "bin"]).sub(g["y"], axis=0)) ** 2
            rows.append((e.mean() / e[bench].mean()).rename((s, b)))
    T = pd.DataFrame(rows)
    T.index = pd.MultiIndex.from_tuples(T.index, names=["symbol", "decile"])
    T = T.groupby(level="decile").mean()
    return T[models] if models else T


# --------------------------------------------------------------------------- #
# 6. Accumulated local effects + variable importance (paper Sec. 4.2)
# --------------------------------------------------------------------------- #

def ale(model, X: pd.DataFrame, col: str, K: int = 100):
    """First-order ALE (Apley & Zhu 2020), paper eq. (28)-(29)."""
    z = X[col].to_numpy(float)
    edges = np.unique(np.quantile(z, np.linspace(0, 1, K + 1)))
    if len(edges) < 3:
        return edges, np.zeros(len(edges)), np.zeros(len(z))
    idx = np.clip(np.searchsorted(edges, z, side="left"), 1, len(edges) - 1)
    lo, hi = X.copy(), X.copy()
    lo[col], hi[col] = edges[idx - 1], edges[idx]
    diff = model.predict(hi) - model.predict(lo)
    eff = np.zeros(len(edges))
    for k in range(1, len(edges)):
        m = idx == k
        eff[k] = diff[m].mean() if m.any() else 0.0
    acc = np.cumsum(eff)
    centered_obs = acc[idx] - acc[idx].mean()
    return edges, acc - acc[idx].mean(), centered_obs


def variable_importance(model, X: pd.DataFrame, cols):
    """Paper eq. (30)-(31): std of centered ALE, normalized to sum to one."""
    I = {c: np.std(ale(model, X, c)[2], ddof=1) for c in cols}
    tot = sum(I.values())
    return pd.Series({c: v / tot for c, v in I.items()}).sort_values(ascending=False)


def vi_for_stock(stock, macro, cfg, d_tr, d_va, which=("HAR-X", "EN", "RF", "NN")):
    """Fit the 'preferred' models on the initial training set (M_ALL, h=1) and
    return VI per model (as in paper Figure 7) plus the fitted models."""
    f = build_features(stock, macro, cfg)
    mc = ml_cols("M_ALL", f)
    D = f[mc + ["y1"]].dropna()
    n_tr = int((D.index <= d_tr).sum())
    n_va = int(((D.index > d_tr) & (D.index <= d_va)).sum())
    tr, va = D.iloc[:n_tr], D.iloc[n_tr:n_tr + n_va]
    fitted = {}
    if "HAR-X" in which:
        fitted["HAR-X"] = OLS(mc).fit(tr, tr["y1"])
    if "EN" in which:
        best = min(((np.mean((SK(mc, _make_reg("EN", a, l1)).fit(tr, tr["y1"]).predict(va)
                              - va["y1"]) ** 2), a, l1) for a, l1 in REG_GRID["EN"]))
        fitted["EN"] = SK(mc, _make_reg("EN", best[1], best[2])).fit(tr, tr["y1"])
    if "RF" in which:
        from sklearn.ensemble import RandomForestRegressor
        fitted["RF"] = SK(mc, lambda: RandomForestRegressor(
            n_estimators=cfg.rf_trees, max_features=1 / 3, min_samples_leaf=cfg.rf_min_leaf,
            random_state=cfg.seed, n_jobs=1), scale=False).fit(tr, tr["y1"])
    if "NN" in which:
        fitted["NN"] = NNEnsemble(mc, cfg.nn_archs[0], cfg).fit(tr, tr["y1"], va, va["y1"])
    vi = pd.DataFrame({k: variable_importance(m, tr, mc) for k, m in fitted.items()})
    return vi, fitted, tr


# --------------------------------------------------------------------------- #
# 7. Simulated VOLARE-like data (dry run without downloads)
# --------------------------------------------------------------------------- #

def simulate_panel(n_stocks=5, n_days=2800, seed=0):
    """Long-memory log-RV (HAR-type) with fat tails; same columns as VOLARE."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    vix = np.empty(n_days)
    lv = np.log(15.0)
    for t in range(n_days):
        lv = 0.02 * np.log(15) + 0.98 * lv + 0.08 * rng.standard_normal()
        vix[t] = np.exp(lv)
    rows = []
    for k in range(n_stocks):
        lrv = np.full(n_days, -0.5)
        for t in range(22, n_days):
            lrv[t] = (-0.05 + 0.35 * lrv[t - 1] + 0.35 * lrv[t - 5:t].mean()
                      + 0.2 * lrv[t - 22:t].mean() + 0.3 * np.log(vix[t - 1] / 15)
                      + 0.45 * rng.standard_t(5) / np.sqrt(5 / 3))
        rv = np.exp(lrv) * 1e-4
        share = np.clip(0.5 + 0.1 * rng.standard_normal(n_days), 0.05, 0.95)
        r = np.sqrt(rv) * rng.standard_normal(n_days)
        open_ = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n_days)))
        rows.append(pd.DataFrame({
            "date": dates, "symbol": f"S{k:02d}", "rv5": rv, "rq5": rv ** 2 * rng.gamma(4, 0.25 * 3, n_days) / 3 * 78,
            "rsp5": rv * share, "rsn5": rv * (1 - share), "open": open_,
            "close": open_ * np.exp(r), "volume": rng.lognormal(15, 0.3, n_days)}))
    macro = pd.DataFrame({"VIX": vix, "EPU": rng.lognormal(4.5, 0.4, n_days),
                          "US3M": np.cumsum(rng.normal(0, 0.02, n_days)) + 2,
                          "HSI": rng.exponential(1.0, n_days),
                          "ADS": np.cumsum(rng.normal(0, 0.05, n_days))}, index=dates)
    return pd.concat(rows, ignore_index=True), macro

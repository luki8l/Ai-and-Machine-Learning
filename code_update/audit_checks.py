"""Sanity checks for volml.py: targets, no look-ahead, DM test size, OLS, LogHAR, ALE, COVID window.
Run from the project folder:  python audit_checks.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import volml as V
ok = lambda name, cond: print(("PASS " if cond else "FAIL ") + name)

# ---------- 1. targets & HAR features on a toy series ----------
dates = pd.bdate_range("2020-01-01", periods=60)
rv = np.arange(1, 61, dtype=float)
st = pd.DataFrame({"date": dates, "symbol": "X", "rv": rv})
cfg = V.Config(horizons=(1, 5, 22), offline_macro=True)
f = V.build_features(st, None, cfg)
t = 30
ok("y1[t] = RV[t+1]", f["y1"].iloc[t] == rv[t+1])
ok("y5[t] = mean RV[t+1..t+5]", np.isclose(f["y5"].iloc[t], rv[t+1:t+6].mean()))
ok("y22[t] = mean RV[t+1..t+22]", np.isclose(f["y22"].iloc[t], rv[t+1:t+23].mean()))
ok("RVD[t] = RV[t]", f["RVD"].iloc[t] == rv[t])
ok("RVW[t] = mean RV[t-4..t]", np.isclose(f["RVW"].iloc[t], rv[t-4:t+1].mean()))
ok("RVM[t] = mean RV[t-21..t]", np.isclose(f["RVM"].iloc[t], rv[t-21:t+1].mean()))
ok("last 22 rows have no y22 (no future)", f["y22"].iloc[-22:].isna().all())

# ---------- 2. no look-ahead in the rolling estimation ----------
class Spy:
    log = []
    def __init__(self): pass
    def fit(self, X, y):
        Spy.log.append(X.index[-1]); self.last = X.index[-1]; return self
    def predict(self, X):
        Spy.origins = getattr(Spy, "origins", []) + list(X.index); return np.ones(len(X))
D = f[["RVD", "y22"]].dropna()
for h, ycol in ((1, "y1"), (5, "y5"), (22, "y22")):
    D = f[["RVD", ycol]].dropna(); Spy.log, Spy.origins = [], []
    start = 25
    V._rolling(lambda: Spy(), D, ycol, start, h, window=20, every=1)
    worst = True
    for k, i in enumerate(range(start, len(D))):
        last_train = D.index.get_loc(Spy.log[k])
        # target of last training row covers days last_train+1 .. last_train+h  -> must be <= origin i
        worst &= (last_train + h) <= i
    ok(f"h={h}: every training target is realized by the forecast origin", worst)

# ---------- 3. DM test: size under H0 and power ----------
rng = np.random.default_rng(0)
for h in (1, 5, 22):
    rej = 0
    for _ in range(2000):
        e = rng.standard_normal(500 + h)
        d = np.convolve(e, np.ones(h) / h, "valid")[:500]       # MA(h-1) differential, mean 0
        lb = 1 + d; lm = np.ones(500)                            # loss_bench - loss_model = d
        rej += V.dm_test(lb, lm, h)[1] < 0.05
    ok(f"DM size h={h}: rejection rate {rej/2000:.3f} close to 0.05", 0.03 < rej / 2000 < 0.075)
p = V.dm_test(np.ones(500) + 0.2 + rng.standard_normal(500), np.ones(500), 1)[1]
ok(f"DM detects a better model (p={p:.4f})", p < 0.01)
p2 = V.dm_test(np.ones(500), np.ones(500) + 0.2 + rng.standard_normal(500), 1)[1]
ok(f"DM does not reward a worse model (p={p2:.3f})", p2 > 0.5)

# ---------- 4. OLS = textbook normal equations; LogHAR bias correction ----------
X = pd.DataFrame(rng.standard_normal((300, 3)), columns=["RVD", "RVW", "RVM"])
y = 1 + X @ [0.4, 0.3, 0.2] + 0.1 * rng.standard_normal(300)
m = V.OLS(["RVD", "RVW", "RVM"]).fit(X, y)
Z = np.column_stack([np.ones(300), X.values]); b = np.linalg.solve(Z.T @ Z, Z.T @ y)
ok("OLS coefficients = (X'X)^-1 X'y", np.allclose(m.b, b))
yl = np.exp(0.5 + 0.3 * rng.standard_normal(5000)); Xl = pd.DataFrame({"RVD": np.ones(5000), "RVW": np.ones(5000), "RVM": np.ones(5000)})
ml = V.OLS(["RVD", "RVW", "RVM"], log_cols=["RVD", "RVW", "RVM"], log_target=True).fit(Xl, yl)
ok("LogHAR forecast is unbiased in levels (Jensen correction)", abs(ml.predict(Xl[:1])[0] / yl.mean() - 1) < 0.02)

# ---------- 5. ALE of a linear model = its slope ----------
Xa = pd.DataFrame(rng.uniform(0, 10, (2000, 3)), columns=["RVD", "RVW", "RVM"])
ya = 2 * Xa.RVD + 0.5 * Xa.RVW
la = V.OLS(["RVD", "RVW", "RVM"]).fit(Xa, ya)
e, a, _ = V.ale(la, Xa, "RVD")
ok("ALE slope of linear model = coefficient (2.0)", np.isclose(np.polyfit(e, a, 1)[0], 2.0, atol=1e-6))
vi = V.variable_importance(la, Xa, ["RVD", "RVW", "RVM"])
ok(f"VI ranks RVD > RVW > RVM ({vi.round(2).to_dict()})", vi["RVD"] > vi["RVW"] > vi["RVM"])

# ---------- 6. COVID-window removal keeps calendar window aligned ----------
d2 = pd.DataFrame({"a": 1.0}, index=pd.bdate_range("2019-01-01", "2021-12-31"))
dd = V.drop_window(d2, ("2020-01-01", "2020-12-31"))
ok("2020 rows removed, 2019/2021 kept", (dd.index.year != 2020).all() and len(dd) == len(d2) - (d2.index.year == 2020).sum())
try:
    V.drop_window(d2, ("2020-01-01", "2026-01-01"), d_va="2024-01-01"); ok("exclusion into test refused", False)
except ValueError:
    ok("exclusion into test refused", True)

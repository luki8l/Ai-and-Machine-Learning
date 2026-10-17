"""
fetch_data.py - instant fallbacks if VOLARE access is not granted in time.

All three write a daily file in the same format the notebook reads
(date, symbol, rv, rq, rsp, rsn, open, close, volume), so only
`cfg.volare_path` changes in the notebook.

  A) Alpaca (recommended fallback): true 5-min realized variance, Dow 30, 2016 -> today.
     Needs a FREE Alpaca paper account (email sign-up, open to non-US residents)
     -> dashboard -> API keys.  ~10-20 min download for 30 stocks.
        python fetch_data.py alpaca --key YOUR_KEY_ID --secret YOUR_SECRET

  B) Yahoo (zero sign-up): daily OHLC -> Garman-Klass range variance, Dow 30, 2001 -> today.
     Noisier proxy, no intraday measures (HARQ/SHAR are dropped automatically),
     but covers the paper's full 2001-2017 window plus the extension.
        python fetch_data.py yahoo

  C) Binance (zero sign-up): true 5-min realized variance for crypto, 2018 -> today.
     Changes the asset class (an extension, not a replication).
        python fetch_data.py binance --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd

from volml import DOW30_VOLARE, realized_from_bars

OUT = "data"


# --------------------------------------------------------------------------- #
# A) Alpaca 5-min bars (SIP feed, raw prices)
# --------------------------------------------------------------------------- #
ALPACA_URL = "https://data.alpaca.markets/v2/stocks/bars"


def parse_alpaca(payload: dict) -> pd.DataFrame:
    """Alpaca JSON page -> bars frame (timestamp in New York time = bar start)."""
    rows = []
    for sym, bars in (payload.get("bars") or {}).items():
        for b in bars or []:
            rows.append((b["t"], sym, b["o"], b["h"], b["l"], b["c"], b["v"]))
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    if len(df):
        df["timestamp"] = (pd.to_datetime(df["timestamp"], utc=True)
                           .dt.tz_convert("America/New_York").dt.tz_localize(None))
    return df


def fetch_alpaca(key, secret, symbols, start="2016-01-01", end=None, feed="sip"):
    import requests
    os.makedirs(f"{OUT}/alpaca/bars", exist_ok=True)
    end = end or (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    daily = []
    for sym in symbols:
        path = f"{OUT}/alpaca/bars/{sym}.csv"
        if os.path.exists(path):                       # resume after interruptions
            bars = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            parts, token, n = [], None, 0
            while True:
                params = {"symbols": sym, "timeframe": "5Min", "start": start, "end": end,
                          "limit": 10000, "adjustment": "raw", "feed": feed}
                if token:
                    params["page_token"] = token
                r = requests.get(ALPACA_URL, headers=hdr, params=params, timeout=60)
                if r.status_code == 429:               # rate limit (free: 200 req/min)
                    time.sleep(5)
                    continue
                r.raise_for_status()
                js = r.json()
                parts.append(parse_alpaca(js))
                n += 1
                token = js.get("next_page_token")
                if not token:
                    break
                time.sleep(0.35)
            bars = pd.concat(parts, ignore_index=True)
            bars.to_csv(path, index=False)
        d = realized_from_bars(bars)                   # regular session 09:30-16:00 ET
        print(f"{sym}: {len(bars):,} bars -> {len(d):,} days "
              f"({d['date'].min().date()} -> {d['date'].max().date()})")
        daily.append(d)
    out = pd.concat(daily, ignore_index=True)
    out.to_csv(f"{OUT}/alpaca/realized.csv", index=False)
    print(f"wrote {OUT}/alpaca/realized.csv  -> cfg.volare_path = '{OUT}/alpaca/realized.csv'")
    return out


# --------------------------------------------------------------------------- #
# B) Yahoo daily OHLC -> Garman-Klass
# --------------------------------------------------------------------------- #
def garman_klass(o, h, l, c):
    return 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2


def fetch_yahoo(symbols, start="2001-01-01"):
    import yfinance as yf
    os.makedirs(f"{OUT}/yahoo", exist_ok=True)
    out = []
    for sym in symbols:
        px = yf.download(sym, start=start, auto_adjust=False, progress=False)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        px = px.dropna(subset=["Open", "High", "Low", "Close"])
        px = px[(px["High"] > px["Low"]) & (px["Open"] > 0)]
        d = pd.DataFrame({"date": px.index, "symbol": sym,
                          "rv": garman_klass(px["Open"], px["High"], px["Low"], px["Close"]).values,
                          "open": px["Open"].values, "close": px["Close"].values,
                          "volume": px["Volume"].values})
        d = d[d["rv"] > 0]
        print(f"{sym}: {len(d):,} days ({d['date'].min().date()} -> {d['date'].max().date()})")
        out.append(d)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(f"{OUT}/yahoo/realized.csv", index=False)
    print(f"wrote {OUT}/yahoo/realized.csv  -> cfg.volare_path = '{OUT}/yahoo/realized.csv'")
    return out


# --------------------------------------------------------------------------- #
# C) Binance public 5-min klines (no account)
# --------------------------------------------------------------------------- #
BINANCE_URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/5m/{s}-5m-{y}-{m:02d}.zip"


def parse_binance(csv_bytes: bytes, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv_bytes), header=None)
    if not str(df.iloc[0, 0]).isdigit():               # some files carry a header row
        df = df.iloc[1:].astype(float)
    t = df.iloc[:, 0].astype("int64")
    unit = np.where(t > 10 ** 14, "us", "ms")          # spot files use microseconds from 2025
    ts = pd.to_datetime(np.where(unit == "us", t // 1000, t), unit="ms")
    return pd.DataFrame({"timestamp": ts, "symbol": symbol, "open": df.iloc[:, 1].astype(float),
                         "high": df.iloc[:, 2].astype(float), "low": df.iloc[:, 3].astype(float),
                         "close": df.iloc[:, 4].astype(float), "volume": df.iloc[:, 5].astype(float)})


def fetch_binance(symbols, start="2018-01"):
    import requests
    os.makedirs(f"{OUT}/binance", exist_ok=True)
    months = pd.period_range(start, pd.Timestamp.utcnow().strftime("%Y-%m"), freq="M")[:-1]
    out = []
    for sym in symbols:
        parts = []
        for p in months:
            r = requests.get(BINANCE_URL.format(s=sym, y=p.year, m=p.month), timeout=60)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                parts.append(parse_binance(z.read(z.namelist()[0]), sym))
        bars = pd.concat(parts, ignore_index=True)
        d = realized_from_bars(bars, session=None)     # 24/7: UTC calendar days
        print(f"{sym}: {len(d):,} days ({d['date'].min().date()} -> {d['date'].max().date()})")
        out.append(d)
    out = pd.concat(out, ignore_index=True)
    out.to_csv(f"{OUT}/binance/realized.csv", index=False)
    print(f"wrote {OUT}/binance/realized.csv  -> cfg.volare_path = '{OUT}/binance/realized.csv'")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["alpaca", "yahoo", "binance"])
    ap.add_argument("--key")
    ap.add_argument("--secret")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--start")
    a = ap.parse_args()
    if a.source == "alpaca":
        fetch_alpaca(a.key or os.environ["APCA_API_KEY_ID"], a.secret or os.environ["APCA_API_SECRET_KEY"],
                     a.symbols or DOW30_VOLARE, start=a.start or "2016-01-01")
    elif a.source == "yahoo":
        fetch_yahoo(a.symbols or DOW30_VOLARE, start=a.start or "2001-01-01")
    else:
        fetch_binance(a.symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
                      start=a.start or "2018-01")

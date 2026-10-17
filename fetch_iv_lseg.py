"""
fetch_iv_lseg.py - per-stock implied volatility from LSEG Workspace (Refinitiv) via the
uni licence. This is the paper's "IV" predictor (the paper uses OptionMetrics model-free IV;
LSEG's 30-day at-the-money IV is the closest substitute available to us).

Requirements
  * LSEG Workspace desktop app open and logged in (uni account) on the same computer
  * pip install lseg-data
  * python fetch_iv_lseg.py                 (all 30 stocks, 2016 -> today)
    python fetch_iv_lseg.py --symbols AAPL MSFT --start 2016-01-01

Output: data/lseg/iv.csv with columns date, symbol, iv (annualized %, mean of the
30-day ATM call and put IV where both exist). The notebook picks it up automatically.

Instrument/field names follow LSEG's developer forum: RIC = ticker + "ATMIV.U"
(e.g. AAPLATMIV.U), fields TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORCALLOPTIONS / ...PUTOPTIONS.
If a stock fails, test the RIC in Workspace first (search the RIC, check the history).
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from volml import DOW30_VOLARE

CALL = "TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORCALLOPTIONS"
PUT = "TR.30DAYATTHEMONEYIMPLIEDVOLATILITYINDEXFORPUTOPTIONS"


def _one(ld, ric, start, end):
    df = ld.get_data(universe=[ric], fields=[f"{CALL}.Date", CALL, PUT],
                     parameters={"SDate": start, "EDate": end, "Frq": "D"})
    if df is None or len(df) == 0:
        return pd.DataFrame()
    cols = [c for c in df.columns if str(c).lower() not in ("instrument",)]
    dcol = next(c for c in cols if "date" in str(c).lower())
    vals = [c for c in cols if c != dcol]
    out = pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce").dt.tz_localize(None)})
    num = df[vals].apply(pd.to_numeric, errors="coerce")
    out["iv"] = num.mean(axis=1, skipna=True)          # avg of call/put ATM IV
    return out.dropna()


def fetch(symbols, start="2016-01-01", end=None):
    import lseg.data as ld
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    ld.open_session()                                    # desktop session (Workspace must be running)
    os.makedirs("data/lseg", exist_ok=True)
    years = pd.period_range(start[:4], end[:4], freq="Y")
    frames, failed = [], []
    for sym in symbols:
        ric = f"{sym}ATMIV.U"
        parts = []
        for y in years:                                  # one year per request (data-point limits)
            s = max(pd.Timestamp(start), y.start_time).strftime("%Y-%m-%d")
            e = min(pd.Timestamp(end), y.end_time).strftime("%Y-%m-%d")
            for attempt in range(3):
                try:
                    parts.append(_one(ld, ric, s, e))
                    break
                except Exception as ex:  # noqa: BLE001
                    if attempt == 2:
                        print(f"  {ric} {y}: {type(ex).__name__}: {ex}")
                    time.sleep(2)
        d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(d) == 0:
            failed.append(sym)
            print(f"{sym}: no data ({ric})")
            continue
        d = d.drop_duplicates("date").sort_values("date").assign(symbol=sym)
        print(f"{sym}: {len(d):,} days ({d['date'].min().date()} -> {d['date'].max().date()}), "
              f"median IV {d['iv'].median():.1f}")
        frames.append(d)
    ld.close_session()
    out = pd.concat(frames, ignore_index=True)[["date", "symbol", "iv"]]
    out.to_csv("data/lseg/iv.csv", index=False)
    print(f"wrote data/lseg/iv.csv ({out['symbol'].nunique()} stocks)."
          + (f" Failed: {failed}" if failed else ""))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end")
    a = ap.parse_args()
    fetch(a.symbols or DOW30_VOLARE, a.start, a.end)

"""Convert the filled LSEG workbook (values, one sheet per ticker) into data/lseg/iv.csv.

Usage (from the project folder):  python lseg/convert_lseg_excel.py LSEG_IV_values.xlsx
Each ticker sheet: a date column plus call-IV and put-IV columns (headers as LSEG writes them).
iv = mean of call and put ATM IV (annualized %), as used by volml (cfg.iv_file).
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


def to_dates(col: pd.Series) -> pd.Series:
    """Datetimes, date strings or raw Excel serial numbers -> datetime."""
    num = pd.to_numeric(col, errors="coerce")
    if num.between(20000, 80000).sum() > 0.5 * num.notna().sum() > 0:
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    return pd.to_datetime(col, errors="coerce")

SKIP = {"Anleitung", "Kontrolle"}


def sheet_to_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    # find the first row that contains a parsable date in the first column -> data starts there
    dates = to_dates(raw.iloc[:, 0])
    data = raw[dates.notna()].copy()
    if data.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": dates[dates.notna()].dt.tz_localize(None).values})
    vals = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    out["iv"] = vals.mean(axis=1, skipna=True).values
    out["symbol"] = ticker
    return out.dropna()


def main(path, out="data/lseg/iv.csv"):
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    frames, missing = [], []
    for name, raw in sheets.items():
        if name in SKIP:
            continue
        f = sheet_to_frame(raw, name.strip().upper())
        if f.empty:
            missing.append(name)
            continue
        f = f.drop_duplicates("date").sort_values("date")
        print(f"{name}: {len(f):,} days ({f['date'].min().date()} -> {f['date'].max().date()}), "
              f"median IV {f['iv'].median():.1f}")
        frames.append(f)
    if not frames:
        sys.exit("No data found - were the formulas converted to values before saving?")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = pd.concat(frames)[["date", "symbol", "iv"]]
    res.to_csv(out, index=False)
    print(f"wrote {out} with {res['symbol'].nunique()} stocks." + (f" Empty sheets: {missing}" if missing else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "LSEG_IV_values.xlsx")

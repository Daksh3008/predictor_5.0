# src/fetch_data.py
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from .utils import ensure_dir


# ---------------------------------------------------------
# EXACT FIX FOR YOUR MULTIINDEX FORMAT
# ---------------------------------------------------------
def flatten_yf_multiindex(df):
    """
    Your df columns look like:
       MultiIndex([('Close', 'DEEPAKNTR.BO'), ...])
    We ONLY keep the first level: Close, Open, High, Low, Volume, Adj Close.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]   # take first level only
    return df


# ---------------------------------------------------------
# DEEPAKNTR FETCHER — using your exact output format
# ---------------------------------------------------------
def fetch_deepakntr(out_path, start="2000-01-01"):
    print("\nFetching DEEPAKNTR.BO ...")

    df = yf.download("DEEPAKNTR.BO", start=start, auto_adjust=False)

    if df.empty:
        raise ValueError("Yahoo returned EMPTY dataset for DEEPAKNTR.BO")

    df = df.reset_index()
    df = flatten_yf_multiindex(df)

    # Ensure correct columns
    required = ["Date","Open","High","Low","Close","Adj Close","Volume"]
    for col in required:
        if col not in df.columns:
            print(f"WARNING: Missing {col}, filling with NaN")
            df[col] = np.nan

    df = df[required]
    df = df.sort_values("Date").reset_index(drop=True)

    ensure_dir(str(Path(out_path).parent))
    df.to_csv(out_path, index=False)

    print(f"Saved → {out_path}")


# ---------------------------------------------------------
# BRENT FETCH
# ---------------------------------------------------------
def fetch_brent(out_path, start="2000-01-01"):
    print("\nFetching Brent (BZ=F) ...")
    df = yf.download("BZ=F", start=start, auto_adjust=False)
    if df.empty:
        raise ValueError("Yahoo returned empty brent dataset")
    df = df.reset_index()
    df = flatten_yf_multiindex(df)
    df = df[["Date","Close"]].sort_values("Date")
    ensure_dir(str(Path(out_path).parent))
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------
# USD/INR FETCH
# ---------------------------------------------------------
def fetch_usdinr(out_path, start="2000-01-01"):
    print("\nFetching USD/INR (INR=X) ...")
    df = yf.download("INR=X", start=start, auto_adjust=False)
    if df.empty:
        raise ValueError("Yahoo returned empty USD/INR dataset")
    df = df.reset_index()
    df = flatten_yf_multiindex(df)
    df = df[["Date","Close"]].sort_values("Date")
    ensure_dir(str(Path(out_path).parent))
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


def fetch_all():
    fetch_deepakntr("data/raw_data/deepakntr_bo_data.csv")
    fetch_brent("data/raw_data/brent.csv")
    fetch_usdinr("data/raw_data/usd_inr.csv")
    print("\n✔ All data fetched successfully!")


if __name__ == "__main__":
    fetch_all()

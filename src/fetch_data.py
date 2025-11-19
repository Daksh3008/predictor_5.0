# src/fetch_data.py
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from .utils import ensure_dir
from src.news_generator import generate_full_synthetic_articles
from src.real_news_fetcher import fetch_and_save_real_news


def flatten_yf_multiindex(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def fetch_deepakntr(out_path, start="2000-01-01"):
    print("\nFetching DEEPAKNTR.BO ...")
    df = yf.download("DEEPAKNTR.BO", start=start, auto_adjust=False)
    if df.empty:
        raise ValueError("Yahoo returned EMPTY dataset for DEEPAKNTR.BO")
    df = df.reset_index()
    df = flatten_yf_multiindex(df)
    df = df[["Date","Open","High","Low","Close","Adj Close","Volume"]]
    df = df.sort_values("Date").reset_index(drop=True)
    ensure_dir(str(Path(out_path).parent))
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


def fetch_brent(out_path, start="2000-01-01"):
    print("\nFetching Brent...")
    df = yf.download("BZ=F", start=start, auto_adjust=False).reset_index()
    df = flatten_yf_multiindex(df)
    df = df[["Date","Close"]]
    ensure_dir(str(Path(out_path).parent))
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


def fetch_usdinr(out_path, start="2000-01-01"):
    print("\nFetching USD/INR...")
    df = yf.download("INR=X", start=start, auto_adjust=False).reset_index()
    df = flatten_yf_multiindex(df)
    df = df[["Date","Close"]]
    ensure_dir(str(Path(out_path).parent))
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


def merge_news(synth_path, real_path, out_path):
    print("\nMerging synthetic + real news...")

    df_s = pd.read_csv(synth_path)
    df_r = pd.read_csv(real_path)

    # Standardize column names
    df_s.rename(columns={"Date": "date"}, inplace=True)
    df_r.rename(columns={"Date": "date"}, inplace=True)

    # Ensure "date" column exists
    if "date" not in df_s.columns:
        raise ValueError("Synthetic news missing 'date' column")

    if df_r is None or df_r.empty:
        print("⚠ No real news found — using ONLY synthetic.")
        df_final = df_s.copy()
    else:
        print(f"✔ Merging synthetic ({len(df_s)}) + real ({len(df_r)}) news")
        df_final = pd.concat([df_s, df_r], ignore_index=True)

    # Convert strings → datetime safely
    df_final["date"] = pd.to_datetime(df_final["date"], errors="coerce")

    # Drop invalid date rows
    df_final = df_final.dropna(subset=["date"])

    # Sort
    df_final = df_final.sort_values("date").reset_index(drop=True)

    # Save
    ensure_dir(str(Path(out_path).parent))
    df_final.to_csv(out_path, index=False)

    print(f"✔ Final news saved → {out_path}")
    return df_final



def fetch_all():
    deepak = "data/raw_data/deepakntr_bo_data.csv"
    synth = "data/raw_data/news_synthetic.csv"
    real  = "data/raw_data/news_real.csv"
    final = "data/raw_data/news_final.csv"

    # Price
    fetch_deepakntr(deepak)
    fetch_brent("data/raw_data/brent.csv")
    fetch_usdinr("data/raw_data/usd_inr.csv")

    # Synthetic news
    price_df = pd.read_csv(deepak, parse_dates=["Date"])
    brent_df = pd.read_csv("data/raw_data/brent.csv", parse_dates=["Date"])
    usd_df = pd.read_csv("data/raw_data/usd_inr.csv", parse_dates=["Date"])
    generate_full_synthetic_articles(price_df, brent_df, usd_df, out_path=synth)

    # Real news
    fetch_and_save_real_news(out_path=real)

    # Merge
    merge_news(synth, real, final)

    print("\n✔ ALL DATA FETCHED + NEWS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    import traceback, sys
    print("STARTING fetch_all()", flush=True)
    try:
        fetch_all()
    except Exception:
        print("ERROR during fetch_all():", file=sys.stderr, flush=True)
        traceback.print_exc()
    else:
        print("fetch_all() finished OK", flush=True)

# src/data_processing.py

import pandas as pd
import numpy as np
from pathlib import Path
from .utils import ensure_dir

# ---------------------------------------------
# LOAD RAW BOOK DATA
# ---------------------------------------------
def load_raw(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------
# MERGE EXTERNAL DATA
# ---------------------------------------------
def merge_external(df, brent_path, usdinr_path):
    br = pd.read_csv(brent_path, parse_dates=["Date"])
    us = pd.read_csv(usdinr_path, parse_dates=["Date"])

    df = df.merge(br.rename(columns={"Close": "brent_close"}),
                  on="Date", how="left")
    df = df.merge(us.rename(columns={"Close": "usd_inr"}),
                  on="Date", how="left")

    # forward fill — NO inplace warnings
    df["brent_close"] = df["brent_close"].ffill()
    df["usd_inr"] = df["usd_inr"].ffill()

    # news_compound not available → 0
    if "news_compound" not in df.columns:
        df["news_compound"] = 0.0
    df["news_compound"] = df["news_compound"].fillna(0.0)

    return df


# ---------------------------------------------
# BASIC INDICATORS (used by all models)
# ---------------------------------------------
def compute_basic_features(df):

    # returns
    df["pct_change"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["ind_rsi_14"] = 100 - (100 / (1 + rs))
    df["ind_rsi_14"] = df["ind_rsi_14"].fillna(50)

    # Moving averages
    df["ind_ma10"] = df["Close"].rolling(10).mean()
    df["ind_ma50"] = df["Close"].rolling(50).mean()
    df["ind_ma200"] = df["Close"].rolling(200).mean()

    # EMA20
    df["ind_ema20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Bollinger
    m = df["Close"].rolling(20).mean()
    s = df["Close"].rolling(20).std()
    df["ind_bb_h"] = m + 2 * s
    df["ind_bb_l"] = m - 2 * s

    # Momentum
    df["ind_mom_10"] = df["Close"] - df["Close"].shift(10)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["ind_macd"] = macd
    df["ind_macd_sig"] = signal

    return df


# ---------------------------------------------
# LIGHTGBM-ONLY FEATURES (lags, vol, zscore, trend)
# ---------------------------------------------
def add_lightgbm_features(df):
    # Lags
    for lag in [1, 2, 5, 10, 20]:
        df[f"log_return_lag{lag}"] = df["log_return"].shift(lag)

    # Volatility
    for w in [5, 10, 20]:
        df[f"vol_{w}"] = df["log_return"].rolling(w).std()

    # Z-score of price
    for w in [20, 50]:
        df[f"zscore_{w}"] = (df["Close"] - df["Close"].rolling(w).mean()) / df["Close"].rolling(w).std()

    # Trend spread
    df["trend_spread"] = df["ind_ema20"] - df["ind_ma50"]

    # MACD histogram
    df["macd_hist"] = df["ind_macd"] - df["ind_macd_sig"]

    return df


# ---------------------------------------------
# HMM FEATURE SET
# ---------------------------------------------
def add_hmm_features(df):

    # volatility-scaled returns (main HMM driver)
    vol20 = df["log_return"].rolling(20).std()
    df["ret_scaled"] = df["log_return"] / vol20

    # add volatility
    df["vol_20"] = vol20

    # add zscore
    df["zscore_50"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"].rolling(50).std()

    return df


# ---------------------------------------------
# TARGET CREATION
# ---------------------------------------------
def create_target(df):
    # smoothed target return
    df["target_return"] = df["log_return"].ewm(span=5, adjust=False).mean()
    return df


# ---------------------------------------------
# MAIN BUILDER FUNCTION
# ---------------------------------------------
def build_features(raw_price_path, out_path, brent_path, usdinr_path):

    ensure_dir(str(Path(out_path).parent))

    df = load_raw(raw_price_path)
    df = merge_external(df, brent_path, usdinr_path)

    df = compute_basic_features(df)
    df = add_lightgbm_features(df)
    df = add_hmm_features(df)
    df = create_target(df)

    df = df.dropna().reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"✔ Features saved → {out_path}")

# ---------------------------------------------------------
# src/pipeline/backtest_unified.py
# Full clean version with isolated DFs & LightGBM modes
# ---------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
#from arch import arch_model


from src.model_training import (
    train_lstm, train_tcn, train_lightgbm
)

from src.model_training import train_lstm, train_lightgbm, train_tcn
from models.hmm_model import hmm_predict
from src.model_evaluation import save_backtest_excel
from src.utils import ensure_dir

# ---------------------------------------------------------
# LIGHTGBM MODES:
#   "none"           → non-recursive (stable)
#   "safe_recursive" → recursive price, NO indicator recompute
#   "full_recursive" → recursive price + recompute indicators
# ---------------------------------------------------------
LIGHTGBM_MODE = "none"   # change to: "safe_recursive" or "full_recursive"


# ---------------------------------------------------------
# RSI HELPER
# ---------------------------------------------------------
def compute_rsi_fast(prices, period=14):
    deltas = np.diff(prices)
    if len(deltas) < period:
        return 50.0
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        return 100.0
    rs = up / down
    rsi = 100 - 100 / (1 + rs)
    return float(rsi)


# ---------------------------------------------------------
# PRICE RECONSTRUCTION FROM SMOOTHED RETURNS
# ---------------------------------------------------------
def reconstruct_price(start_price, predicted_returns):
    price = start_price
    prices = []
    for r in predicted_returns:
        price = price * np.exp(np.clip(r, -0.05, 0.05))
        prices.append(price)
    return np.array(prices)


# ---------------------------------------------------------
# GARCH MODEL
# ---------------------------------------------------------
#def train_garch(train_log_returns):
#    model = arch_model(train_log_returns * 100, p=3, q=3)
#    res = model.fit(disp="off")
#    return res


#def garch_forecast(res):
#    forecast = res.forecast(horizon=1)
#    mu = forecast.mean.iloc[-1].values[0] / 100.0
#    return mu


# ---------------------------------------------------------
# INDICATOR RECOMPUTE (USED ONLY FOR LIGHTGBM full_recursive)
# ---------------------------------------------------------
def recompute_indicators(df, idx):
    close_series = df["Close"].iloc[:idx + 1]

    df.loc[idx, "log_return"] = (
        np.log(df.loc[idx, "Close"]) - np.log(df.loc[idx - 1, "Close"])
    )

    df.loc[idx, "pct_change"] = (df.loc[idx, "Close"] / df.loc[idx - 1, "Close"]) - 1

    # RSI
    df.loc[idx, "ind_rsi_14"] = compute_rsi_fast(close_series.values[-14:])

    # MAs / EMA
    df.loc[idx, "ind_ma10"] = close_series.tail(10).mean()
    df.loc[idx, "ind_ma50"] = close_series.tail(50).mean()
    df.loc[idx, "ind_ma200"] = close_series.tail(200).mean()
    df.loc[idx, "ind_ema20"] = close_series.ewm(span=20, adjust=False).mean().iloc[-1]

    # MACD
    macd_fast = close_series.ewm(span=12, adjust=False).mean()
    macd_slow = close_series.ewm(span=26, adjust=False).mean()
    df.loc[idx, "ind_macd"] = macd_fast.iloc[-1] - macd_slow.iloc[-1]
    df.loc[idx, "ind_macd_sig"] = (
        close_series.ewm(span=9, adjust=False).mean().iloc[-1]
    )

    # Momentum
    if idx >= 10:
        df.loc[idx, "ind_mom_10"] = close_series.iloc[-1] - close_series.iloc[-11]
    else:
        df.loc[idx, "ind_mom_10"] = 0.0

    # BBands
    rolling = close_series.rolling(20)
    df.loc[idx, "ind_bb_h"] = rolling.mean().iloc[-1] + 2 * rolling.std().iloc[-1]
    df.loc[idx, "ind_bb_l"] = rolling.mean().iloc[-1] - 2 * rolling.std().iloc[-1]


# ---------------------------------------------------------
# MAIN BACKTEST PIPELINE
# ---------------------------------------------------------
def main(processed_path, out_excel, train_end):

    print("Loading processed data...")
    df_master = pd.read_csv(processed_path, parse_dates=["Date"])
    df_master = df_master.sort_values("Date").reset_index(drop=True)

    # Split
    train_df = df_master[df_master["Date"] <= train_end].copy()
    test_df = df_master[df_master["Date"] > train_end].copy()

    # Validation
    val_len = max(50, int(0.1 * len(train_df)))
    val_df = train_df.iloc[-val_len:]
    train_df2 = train_df.iloc[:-val_len]

    # Feature lists
    feature_cols = [
        "Close","High","Low","Open","Adj Close","Volume",
        "brent_close","usd_inr","news_compound",
        "log_return","pct_change",
        "ind_rsi_14","ind_ma10","ind_ma50","ind_ma200",
        "ind_ema20","ind_bb_h","ind_bb_l",
        "ind_mom_10","ind_macd","ind_macd_sig"
    ]

    lgbm_cols = [
        col for col in df_master.columns
        if col not in ["Date","target_return","state_final"]
        and not col.startswith("state_")
    ]

    ensure_dir("models/")

    # -----------------------------------------------------
    # TRAIN MODELS
    # -----------------------------------------------------

    
    print("Training LSTM...")
    lstm_cfg = {
        "seq_len": 120,
        "lstm_hidden": 192,
        "lstm_layers": 3,
        "dropout": 0.3,
        "attn_dim": 128,
        "attn_heads": 2,
        "mlp_hidden": 128,
        "lr": 0.0002576021731458815,
        "weight_decay": 7.9e-7
    }

    lstm_model, lstm_scaler = train_lstm(
        train_df2, val_df, feature_cols,
        cfg=lstm_cfg, checkpoint_path="models/lstm_best.pth"
    )

    print("Training TCN...")
    tcn_model, tcn_scaler = train_tcn(
        train_df2, val_df, feature_cols,
        checkpoint_path="models/tcn_best.pth"
    )

    


    print("Training LightGBM...")
    lgbm_model = train_lightgbm(train_df2, val_df, lgbm_cols)

    #print("Training GARCH...")
    #garch_res = train_garch(train_df["log_return"].values)

    print("Training HMM...")
    hmm_model, hmm_states_train = hmm_predict(train_df)
    all_states = hmm_predict(df_master)[1]

    # -----------------------------------------------------
    # Create isolated DFs for each model
    # -----------------------------------------------------
    df_lstm = df_master.copy()
    df_tcn = df_master.copy()
    df_lgbm = df_master.copy()
    df_hmm = df_master.copy()

    # -----------------------------------------------------
    # BACKTEST LOOP
    # -----------------------------------------------------
    print("Running backtest...")

    seq_len = 120

    lstm_preds = []
    tcn_preds = []
    lgbm_preds = []
    hmm_preds = []
    actual_prices = []
    dates = []

    for i in tqdm(range(len(test_df))):
        idx = len(train_df) + i
        end = idx
        start = end - seq_len

        if start < 0:
            lstm_preds.append(np.nan)
            tcn_preds.append(np.nan)
            lgbm_preds.append(np.nan)
            hmm_preds.append(np.nan)
            continue

        # actual price from master
        actual_price = df_master.iloc[end]["Close"]
        actual_prices.append(actual_price)
        dates.append(df_master.iloc[end]["Date"])

        # -------------------------------
        # LSTM prediction
        # -------------------------------
        window_lstm = df_lstm.iloc[start:end]
        X_lstm = lstm_scaler.transform(window_lstm[feature_cols].astype(np.float32))
        X_lstm = torch.tensor(X_lstm).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            r_lstm = lstm_model(X_lstm).item()
        lstm_preds.append(r_lstm)

        # -------------------------------
        # TCN prediction
        # -------------------------------
        window_tcn = df_tcn.iloc[start:end]
        X_tcn = tcn_scaler.transform(window_tcn[feature_cols].astype(np.float32))
        X_tcn = torch.tensor(X_tcn).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            r_tcn = tcn_model(X_tcn).item()
        tcn_preds.append(r_tcn)


        # -------------------------------
        # LightGBM (mode controlled)
        # -------------------------------
        if LIGHTGBM_MODE == "none":
            lgb_feat = df_master.iloc[end][lgbm_cols].values.reshape(1, -1)
            r_lgb = lgbm_model.predict(lgb_feat)[0]

        else:
            last_row = df_lgbm.iloc[end].copy()
            x_lgb = last_row[lgbm_cols].values.reshape(1, -1)
            r_lgb = lgbm_model.predict(x_lgb)[0]

            prev_price = df_lgbm.loc[end - 1, "Adj Close"]
            next_price = prev_price * np.exp(np.clip(r_lgb, -0.05, 0.05))

            df_lgbm.loc[end, "Adj Close"] = next_price
            df_lgbm.loc[end, "Close"] = next_price

            if LIGHTGBM_MODE == "safe_recursive":
                indicator_cols = [
                    "log_return","pct_change",
                    "ind_rsi_14","ind_ma10","ind_ma50","ind_ma200",
                    "ind_ema20","ind_bb_h","ind_bb_l",
                    "ind_mom_10","ind_macd","ind_macd_sig"
                ]
                for col in indicator_cols:
                    df_lgbm.loc[end, col] = df_lgbm.loc[end - 1, col]

            elif LIGHTGBM_MODE == "full_recursive":
                recompute_indicators(df_lgbm, end)

        lgbm_preds.append(r_lgb)

        # -------------------------------
        # GARCH prediction
        # -------------------------------
        #r_garch = garch_forecast(garch_res)
        #garch_preds.append(r_garch)

        # -------------------------------
        # HMM regime
        # -------------------------------
        hmm_preds.append(all_states[end])

    # -----------------------------------------------------
    # RECONSTRUCT PRICES
    # -----------------------------------------------------
    base_price = train_df["Close"].iloc[-1]

    lstm_price = reconstruct_price(base_price, lstm_preds)
    tcn_price = reconstruct_price(base_price, tcn_preds)
    lgbm_price = reconstruct_price(base_price, lgbm_preds)
    ensemble_price = (lstm_price + tcn_price + lgbm_price) / 3.0

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------
    results = {
        "lstm": lstm_price,
        "tcn": tcn_price,
        "lgbm": lgbm_price,
        "hmm": hmm_preds,
        "ensemble": ensemble_price,
        "rmse_lstm": np.sqrt(np.mean((lstm_price - actual_prices)**2)),
        "rmse_tcn": np.sqrt(np.mean((tcn_price - actual_prices)**2)),
        "rmse_lgbm": np.sqrt(np.mean((lgbm_price - actual_prices)**2)),
        "rmse_ensemble": np.sqrt(np.mean((ensemble_price - actual_prices)**2)),
    }

    save_backtest_excel(out_excel, results, dates, np.array(actual_prices))
    print(f"\nBacktest saved to → {out_excel}")


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--train_end", type=str, required=True)
    args = parser.parse_args()

    main(args.processed, args.out, args.train_end)

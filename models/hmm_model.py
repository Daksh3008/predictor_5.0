# src/hmm_model.py

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.signal import medfilt


# --------------------------------------------------------
# Build feature matrix for HMM
# --------------------------------------------------------
def build_hmm_features(df):
    """
    Uses volatility-adjusted returns, volatility, and price zscore.
    """
    X = np.column_stack([
        df["ret_scaled"].values,
        df["vol_20"].values,
        df["zscore_50"].values
    ])
    X = np.nan_to_num(X, nan=0.0)
    return X


# --------------------------------------------------------
# Train HMM with 4 regimes
# --------------------------------------------------------
def train_hmm(df, n_states=4, seed=42):
    X = build_hmm_features(df)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=seed,
        verbose=False
    )

    model.fit(X)
    return model


# --------------------------------------------------------
# Decode regime sequence using Viterbi + smoothing
# --------------------------------------------------------
def decode_regimes(model, df):
    X = build_hmm_features(df)

    # Viterbi optimal path
    raw_states = model.predict(X)

    # 3-day smoothing (median filter)
    smoothed = medfilt(raw_states, kernel_size=3)

    return smoothed


# --------------------------------------------------------
# Re-label regimes into: 0=sideways,1=up,2=down,3=chaos
# --------------------------------------------------------
def relabel_states(df, states):
    df = df.copy()
    df["state_raw"] = states

    # Calculate per-state statistics
    state_info = []
    for s in np.unique(states):
        sdf = df[df["state_raw"] == s]

        mean_ret = sdf["log_return"].mean()
        vol = sdf["log_return"].std()
        slope = np.polyfit(range(len(sdf)), sdf["Close"], 1)[0] if len(sdf) > 10 else 0

        state_info.append((s, mean_ret, vol, slope))

    # Sort states by intuitive ranking
    # Uptrend: highest slope
    # Downtrend: lowest slope
    # Chaos: highest volatility
    # Sideways: lowest volatility & slope near zero

    df_info = pd.DataFrame(state_info, columns=["state", "mean_ret", "vol", "slope"])

    # Identify extremes
    up_state = df_info.loc[df_info["slope"].idxmax(), "state"]
    down_state = df_info.loc[df_info["slope"].idxmin(), "state"]
    chaos_state = df_info.loc[df_info["vol"].idxmax(), "state"]

    # Sideways = the remaining state
    all_states = set(df_info["state"])
    sideways_state = list(all_states - {up_state, down_state, chaos_state})[0]

    # Map to final labels
    label_map = {
        sideways_state: 0,
        up_state: 1,
        down_state: 2,
        chaos_state: 3
    }

    df["state_final"] = df["state_raw"].map(label_map)
    return df["state_final"].values


# --------------------------------------------------------
# Main entry point: train + predict unified
# --------------------------------------------------------
def hmm_predict(df):
    """
    Returns: (hmm_model, state_predictions)
    """
    model = train_hmm(df)
    states = decode_regimes(model, df)
    final_states = relabel_states(df, states)
    return model, final_states

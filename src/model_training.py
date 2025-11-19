# src/model_training.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from src.utils import save_scaler
from tqdm import tqdm
import time

from models.tcn import TCN  # models folder (top level)
from models.lstm_attention import LSTMAttn  # your lstm file inside src/models
from torch.utils.data import DataLoader, TensorDataset



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------
# SEQUENCE GENERATOR (for LSTM + TCN)
# -----------------------------------------------------------


def prepare_sequences_tensor(df, feature_cols, target_col, seq_len=120):
    """
    FAST version using numpy stride tricks.
    Builds ALL sequences instantly (milliseconds).
    """
    import numpy as np
    import torch

    data = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)

    N = len(data)
    if N <= seq_len:
        return torch.empty((0, seq_len, len(feature_cols))), torch.empty((0,))

    # Use stride trick to generate (N-seq_len, seq_len, features) without python loops
    shape = (N - seq_len, seq_len, data.shape[1])
    strides = (data.strides[0], data.strides[0], data.strides[1])
    X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).copy()

    y = target[seq_len:]

    return torch.tensor(X), torch.tensor(y)


# -----------------------------------------------------------
# TRAIN LSTM
# -----------------------------------------------------------


def train_lstm(train_df, val_df, feature_cols, cfg, checkpoint_path):
    from sklearn.preprocessing import StandardScaler
    from src.utils import save_scaler

    seq_len = cfg.get("seq_len", 120)
    hidden = cfg.get("lstm_hidden", 192)
    layers = 2   # TEMPORARILY reduced from 3 for speed
    dropout = cfg.get("dropout", 0.3)
    attn_dim = cfg.get("attn_dim", 128)
    attn_heads = cfg.get("attn_heads", 2)
    mlp_hidden = cfg.get("mlp_hidden", 128)
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 1e-5)

    target_col = "target_return"

    # --- SCALE FEATURES ---
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_df_s = train_df.copy()
    val_df_s = val_df.copy()

    train_df_s[feature_cols] = scaler.transform(train_df_s[feature_cols])
    val_df_s[feature_cols] = scaler.transform(val_df_s[feature_cols])

    save_scaler(scaler, "models/lstm_scaler.pkl")

    # --- BUILD SEQUENCES ONCE ---
    X_tr, y_tr = prepare_sequences_tensor(train_df_s, feature_cols, target_col, seq_len)
    X_va, y_va = prepare_sequences_tensor(val_df_s, feature_cols, target_col, seq_len)

    y_tr = torch.clamp(y_tr, -0.05, 0.05)
    y_va = torch.clamp(y_va, -0.05, 0.05)

    tr_ds = TensorDataset(X_tr, y_tr)
    va_ds = TensorDataset(X_va, y_va)

    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    model = LSTMAttn(
        input_dim=len(feature_cols),
        lstm_hidden=hidden,
        lstm_layers=layers,
        dropout=dropout,
        attn_heads=attn_heads,
        attn_dim=attn_dim,
        mlp_hidden=mlp_hidden
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    print("\n--- LSTM Training ---")
    print("Device:", next(model.parameters()).device)
    best_val = np.inf

    from tqdm import tqdm
    import time

    for epoch in range(1, 41):
        model.train()
        t0 = time.time()
        train_losses = []

        pbar = tqdm(enumerate(tr_loader), total=len(tr_loader), desc=f"LSTM Epoch {epoch}/40", leave=False)
        for step, (xb, yb) in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optim.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            if step % 10 == 0:
                pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                pred = model(xb).squeeze()
                val_losses.append(loss_fn(pred, yb).item())

        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(train_losses))
        print(f"Epoch {epoch:02d} | Train {train_loss:.6f} | Val {val_loss:.6f} | Time {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (val_loss={best_val:.6f})")

    model.load_state_dict(torch.load(checkpoint_path))
    return model, scaler



# -----------------------------------------------------------
# TRAIN TCN
# -----------------------------------------------------------
def train_tcn(train_df, val_df, feature_cols, checkpoint_path):
    from sklearn.preprocessing import StandardScaler
    from src.utils import save_scaler

    seq_len = 120
    target_col = "target_return"

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_df_s = train_df.copy()
    val_df_s = val_df.copy()

    train_df_s[feature_cols] = scaler.transform(train_df_s[feature_cols])
    val_df_s[feature_cols] = scaler.transform(val_df_s[feature_cols])

    save_scaler(scaler, "models/tcn_scaler.pkl")

    X_tr, y_tr = prepare_sequences_tensor(train_df_s, feature_cols, target_col, seq_len)
    X_va, y_va = prepare_sequences_tensor(val_df_s, feature_cols, target_col, seq_len)

    y_tr = torch.clamp(y_tr, -0.05, 0.05)
    y_va = torch.clamp(y_va, -0.05, 0.05)

    tr_ds = TensorDataset(X_tr, y_tr)
    va_ds = TensorDataset(X_va, y_va)

    tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    model = TCN(input_dim=len(feature_cols)).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    print("\n--- TCN Training ---")
    print("Device:", next(model.parameters()).device)
    best_val = np.inf

    from tqdm import tqdm
    import time

    for epoch in range(1, 41):
        model.train()
        t0 = time.time()
        train_losses = []

        pbar = tqdm(enumerate(tr_loader), total=len(tr_loader), desc=f"TCN Epoch {epoch}/40", leave=False)
        for step, (xb, yb) in pbar:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optim.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            if step % 10 == 0:
                pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                pred = model(xb).squeeze()
                val_losses.append(loss_fn(pred, yb).item())

        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d} | Val {val_loss:.6f} | Time {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best TCN model (val_loss={best_val:.6f})")

    model.load_state_dict(torch.load(checkpoint_path))
    return model, scaler



# -----------------------------------------------------------
# TRAIN LIGHTGBM
# -----------------------------------------------------------
def train_lightgbm(train_df, val_df, feature_cols):

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 42,
    }

    if torch.cuda.is_available():
        params["device"] = "gpu"

    train_data = lgb.Dataset(train_df[feature_cols], label=train_df["target_return"])
    val_data = lgb.Dataset(val_df[feature_cols], label=val_df["target_return"])

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    bst = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=callbacks
    )

    return bst

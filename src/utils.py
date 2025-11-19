# src/utils.py
import os
import random
import numpy as np
import torch
from pathlib import Path
import json
import pickle, os

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_scaler(scaler, path):
    ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)
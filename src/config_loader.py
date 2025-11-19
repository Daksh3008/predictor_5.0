# src/config_loader.py
import yaml
from .utils import load_json, save_json, ensure_dir
from pathlib import Path

def load_config(path="config/lstm_best.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_config_as_json(yaml_path="config/lstm_best.yaml", out_json="models/lstm_attention_model/config.json"):
    cfg = load_config(yaml_path)
    ensure_dir(str(Path(out_json).parent))
    save_json(cfg, out_json)
    return cfg

from pathlib import Path
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Chuyển path string → Path object cho tiện
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = Path(v)
    return cfg
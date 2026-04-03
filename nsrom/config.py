# nsrom/config.py
from pathlib import Path

DATA_ROOT = Path(".")

def set_data_root(path):
    global DATA_ROOT
    DATA_ROOT = Path(path)

def data_path(*parts):
    return DATA_ROOT / Path(*parts)

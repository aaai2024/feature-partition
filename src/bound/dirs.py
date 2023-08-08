__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "RES_DIR",
]

from pathlib import Path


BASE_DIR = None
DATA_DIR = None
MODELS_DIR = None
RES_DIR = None


def _update_all_paths():
    r""" Sets all path names based on the base directory """
    global BASE_DIR, DATA_DIR, MODELS_DIR, RES_DIR

    BASE_DIR = Path(".").absolute()

    DATA_DIR = BASE_DIR / ".data"
    MODELS_DIR = BASE_DIR / "models"
    RES_DIR = BASE_DIR / "res"


_update_all_paths()

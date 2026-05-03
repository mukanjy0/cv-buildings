from pathlib import Path

ROOT = Path(".").resolve()

DATASETS = {
    "oxford": {
        "img_dir": ROOT / "oxford" / "data" / "img",
        "gt_dir": ROOT / "oxford" / "data" / "gt",
    },
    "paris": {
        "img_dir": ROOT / "paris" / "data" / "img",
        "gt_dir": ROOT / "paris" / "data" / "gt",
    },
}

CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def ensure_dirs():
    (CACHE_DIR / "features").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rankings").mkdir(parents=True, exist_ok=True)
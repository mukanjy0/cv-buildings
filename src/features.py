from pathlib import Path
import cv2 as cv
import numpy as np
from tqdm import tqdm

from .paths import DATASETS, CACHE_DIR, IMG_EXTS


def list_images(img_dir: Path):
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    ids = [p.stem for p in paths]
    return ids, paths


def save_ids(ids, path: Path):
    with open(path, "w") as f:
        for image_id in ids:
            f.write(image_id + "\n")


def load_ids(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_sift():
    if not hasattr(cv, "SIFT_create"):
        raise RuntimeError("cv.SIFT_create not found. Install opencv-contrib-python or recent opencv-python.")
    return cv.SIFT_create()


def get_image_path(dataset: str, image_id: str):
    img_dir = DATASETS[dataset]["img_dir"]

    for ext in IMG_EXTS:
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p

    return None


def sift_meanstd_from_bbox(dataset: str, image_id: str, bbox, detector=None):
    if detector is None:
        detector = create_sift()

    image_path = get_image_path(dataset, image_id)
    if image_path is None:
        raise FileNotFoundError(f"Image not found for id: {image_id}")

    gray = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {image_path}")

    x1, y1, x2, y2 = bbox

    keypoints, desc = detector.detectAndCompute(gray, None)

    if desc is None or len(desc) == 0:
        return np.zeros(256, dtype=np.float32)

    keep = []
    for i, kp in enumerate(keypoints):
        x, y = kp.pt
        if x1 <= x <= x2 and y1 <= y <= y2:
            keep.append(i)

    if len(keep) == 0:
        return np.zeros(256, dtype=np.float32)

    desc = desc[keep].astype(np.float32)

    mean = desc.mean(axis=0)
    std = desc.std(axis=0)

    return np.concatenate([mean, std]).astype(np.float32)


def extract_sift_meanstd(image_path: Path, detector=None):
    if detector is None:
        detector = create_sift()

    gray = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {image_path}")

    _, desc = detector.detectAndCompute(gray, None)

    if desc is None or len(desc) == 0:
        return np.zeros(256, dtype=np.float32)

    desc = desc.astype(np.float32)
    mean = desc.mean(axis=0)
    std = desc.std(axis=0)

    return np.concatenate([mean, std]).astype(np.float32)


def extract_or_load_sift_meanstd(dataset: str, force=False):
    cfg = DATASETS[dataset]

    out_dir = CACHE_DIR / "features" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = out_dir / "sift_meanstd_features.npy"
    ids_path = out_dir / "sift_meanstd_ids.txt"

    if feat_path.exists() and ids_path.exists() and not force:
        ids = load_ids(ids_path)
        X = np.load(feat_path)
        print(f"Loaded cached {dataset}: {X.shape}")
        return ids, X

    ids, paths = list_images(cfg["img_dir"])
    detector = create_sift()

    feats = []
    for path in tqdm(paths, desc=f"Extracting SIFT mean/std for {dataset}"):
        feats.append(extract_sift_meanstd(path, detector))

    X = np.vstack(feats).astype(np.float32)

    np.save(feat_path, X)
    save_ids(ids, ids_path)

    print(f"Saved {dataset}: {X.shape}")
    return ids, X
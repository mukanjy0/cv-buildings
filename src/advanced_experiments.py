import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from .evaluation import compile_compute_ap, evaluate_experiment, list_query_files, parse_query_file
from .features import get_image_path, list_images, load_ids, save_ids
from .paths import CACHE_DIR, DATASETS, RESULTS_DIR
from .retrieval import l2_normalize, write_ranking_file


SUMMARY_COLUMNS = [
    "dataset",
    "experiment",
    "descriptor",
    "representation",
    "metric",
    "normalization",
    "vocab_size",
    "extra_params",
    "map",
    "num_queries",
    "status",
    "error",
]

CONFIG_KEY_COLUMNS = [
    "dataset",
    "experiment",
    "descriptor",
    "representation",
    "metric",
    "normalization",
    "vocab_size",
    "extra_params",
]


class DependencyUnavailable(RuntimeError):
    pass


@dataclass
class ExperimentConfig:
    dataset: str
    experiment: str
    descriptor: str
    representation: str
    metric: str
    normalization: str = ""
    vocab_size: int | None = None
    extra_params: dict | None = None


def create_sift():
    if not hasattr(cv, "SIFT_create"):
        raise DependencyUnavailable("cv.SIFT_create is unavailable")
    return cv.SIFT_create()


def create_surf():
    xfeatures2d = getattr(cv, "xfeatures2d", None)
    if xfeatures2d is None or not hasattr(xfeatures2d, "SURF_create"):
        raise DependencyUnavailable("cv.xfeatures2d.SURF_create is unavailable")
    return xfeatures2d.SURF_create()


def create_detector(descriptor):
    if descriptor in ("sift", "rootsift"):
        return create_sift()
    if descriptor == "surf":
        return create_surf()
    raise ValueError(f"Unknown local descriptor: {descriptor}")


def rootsift_transform(desc, l2=True):
    if desc is None or len(desc) == 0:
        return desc
    desc = desc.astype(np.float32, copy=True)
    desc /= (np.sum(np.abs(desc), axis=1, keepdims=True) + 1e-12)
    desc = np.sqrt(np.maximum(desc, 0.0))
    if l2:
        desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
    return desc.astype(np.float32)


def compute_ap_executable():
    exe = RESULTS_DIR.parent / "compute_ap.exe"
    if exe.exists():
        return exe
    return compile_compute_ap()


def summary_path(sorted_=False):
    name = "advanced_summary_sorted.csv" if sorted_ else "advanced_summary.csv"
    return RESULTS_DIR / name


def append_summary_row(row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = summary_path(False)
    df = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
    if out.exists():
        existing = pd.read_csv(out)
        if row["status"] != "ok":
            keys = ["dataset", "experiment", "descriptor", "representation", "metric",
                    "normalization", "vocab_size", "extra_params", "status", "error"]
            existing_cmp = existing.copy()
            row_cmp = pd.DataFrame([row])
            for key in keys:
                existing_cmp[key] = existing_cmp[key].fillna("").astype(str)
                row_cmp[key] = row_cmp[key].fillna("").astype(str)
            duplicate = (existing_cmp[keys] == row_cmp.loc[0, keys]).all(axis=1).any()
            if duplicate:
                write_sorted_summary()
                return
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)
    write_sorted_summary()


def write_sorted_summary():
    out = summary_path(False)
    if not out.exists():
        return
    df = pd.read_csv(out)
    ok = df[df["status"].eq("ok")].copy()
    failed = df[~df["status"].eq("ok")].copy()
    if "map" in ok:
        ok = ok.sort_values("map", ascending=False, na_position="last")
    pd.concat([ok, failed], ignore_index=True).to_csv(summary_path(True), index=False)


def base_row(cfg: ExperimentConfig):
    row = asdict(cfg)
    row["extra_params"] = json.dumps(row["extra_params"] or {}, sort_keys=True)
    row["map"] = np.nan
    row["num_queries"] = 0
    row["status"] = "ok"
    row["error"] = ""
    return row


def config_key_row(cfg: ExperimentConfig):
    row = base_row(cfg)
    return {key: row[key] for key in CONFIG_KEY_COLUMNS}


def successful_result_exists(cfg: ExperimentConfig):
    out = summary_path(False)
    if not out.exists():
        return False
    df = pd.read_csv(out)
    if df.empty or "status" not in df:
        return False
    same_name = (
        df["dataset"].fillna("").astype(str).eq(cfg.dataset)
        & df["experiment"].fillna("").astype(str).eq(cfg.experiment)
        & df["status"].fillna("").astype(str).eq("ok")
    )
    if same_name.any():
        return True
    key = config_key_row(cfg)
    cmp = df.copy()
    for col, value in key.items():
        cmp[col] = cmp[col].fillna("").astype(str)
        cmp = cmp[cmp[col].eq("" if pd.isna(value) else str(value))]
        if cmp.empty:
            return False
    return cmp["status"].eq("ok").any()


def image_ids(dataset):
    ids, _ = list_images(DATASETS[dataset]["img_dir"])
    return ids


def descriptor_cache_path(dataset, descriptor, image_id):
    return CACHE_DIR / "descriptors" / dataset / descriptor / f"{image_id}.npz"


def extract_local_descriptor(dataset, image_id, descriptor="sift", detector=None, force=False):
    path = descriptor_cache_path(dataset, descriptor, image_id)
    skip_cache_write = False
    if path.exists() and not force:
        try:
            with np.load(path) as data:
                return data["xy"].astype(np.float32), data["desc"].astype(np.float32)
        except Exception as exc:
            print(f"Recomputing corrupt descriptor cache {path}: {exc}")
            try:
                path.unlink(missing_ok=True)
            except OSError as unlink_exc:
                print(f"Could not remove corrupt cache {path}; using uncached recomputation: {unlink_exc}")
                skip_cache_write = True

    image_path = get_image_path(dataset, image_id)
    if image_path is None:
        raise FileNotFoundError(f"Missing image: {dataset}/{image_id}")
    gray = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {image_path}")

    if detector is None:
        detector = create_detector(descriptor)
    keypoints, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        xy = np.empty((0, 2), dtype=np.float32)
        desc = np.empty((0, detector.descriptorSize()), dtype=np.float32)
    else:
        xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        desc = desc.astype(np.float32)
        if descriptor == "rootsift":
            desc = rootsift_transform(desc)

    if not skip_cache_write:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(path, xy=xy, desc=desc)
        except OSError as exc:
            print(f"Could not write descriptor cache {path}: {exc}")
    return xy, desc


def descriptors_in_bbox(xy, desc, bbox):
    if bbox is None or len(desc) == 0:
        return desc
    x1, y1, x2, y2 = bbox
    keep = (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)
    return desc[keep]


def meanstd_from_desc(desc, dim):
    if desc is None or len(desc) == 0:
        return np.zeros(dim * 2, dtype=np.float32)
    desc = desc.astype(np.float32)
    return np.concatenate([desc.mean(axis=0), desc.std(axis=0)]).astype(np.float32)


def extract_or_load_meanstd(dataset, descriptor):
    out_dir = CACHE_DIR / "features" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"{descriptor}_meanstd_features.npy"
    ids_path = out_dir / f"{descriptor}_meanstd_ids.txt"
    if feat_path.exists() and ids_path.exists():
        return load_ids(ids_path), np.load(feat_path)

    detector = create_detector(descriptor)
    dim = detector.descriptorSize()
    ids = image_ids(dataset)
    feats = []
    for image_id in tqdm(ids, desc=f"Extracting {descriptor.upper()} mean/std for {dataset}"):
        _, desc = extract_local_descriptor(dataset, image_id, descriptor, detector)
        feats.append(meanstd_from_desc(desc, dim))
    X = np.vstack(feats).astype(np.float32)
    np.save(feat_path, X)
    save_ids(ids, ids_path)
    return ids, X


def normalize_features(X, normalization):
    X = X.astype(np.float32, copy=True)
    if normalization == "l1":
        denom = np.sum(np.abs(X), axis=1, keepdims=True) + 1e-12
        return X / denom
    if normalization == "l2":
        return l2_normalize(X)
    if normalization == "sqrt_l2":
        X = np.sqrt(np.maximum(X, 0.0))
        return l2_normalize(X)
    if normalization in ("", None, "none"):
        return X
    raise ValueError(f"Unknown normalization: {normalization}")


def rank_vector(query_vec, X, ids, metric, exclude_id):
    scores = score_vector(query_vec, X, metric)
    if metric in ("euclidean", "chisquare"):
        order = np.argsort(scores)
    else:
        order = np.argsort(-scores)
    return [ids[i] for i in order if ids[i] != exclude_id]


def score_vector(query_vec, X, metric):
    if np.isnan(query_vec).any() or np.isnan(X).any():
        raise ValueError("NaN found in feature matrix")
    q = query_vec.reshape(1, -1).astype(np.float32)
    if metric == "cosine":
        return l2_normalize(X) @ l2_normalize(q)[0]
    elif metric == "euclidean":
        return np.linalg.norm(X - q, axis=1)
    elif metric == "chisquare":
        denom = X + q + 1e-12
        return 0.5 * np.sum(((X - q) ** 2) / denom, axis=1)
    elif metric == "hist_intersection":
        return np.minimum(X, q).sum(axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def similarity_scores(query_vec, X, metric):
    scores = score_vector(query_vec, X, metric).astype(np.float32)
    if metric in ("euclidean", "chisquare"):
        return -scores
    return scores


def normalize_scores(scores, exclude_idx=None, method="minmax"):
    scores = scores.astype(np.float32, copy=True)
    valid = np.isfinite(scores)
    if exclude_idx is not None:
        valid[exclude_idx] = False
    if not valid.any():
        return np.zeros_like(scores, dtype=np.float32)
    out = np.zeros_like(scores, dtype=np.float32)
    vals = scores[valid]
    if method == "zscore":
        out[valid] = (vals - vals.mean()) / (vals.std() + 1e-12)
    else:
        lo, hi = vals.min(), vals.max()
        out[valid] = (vals - lo) / (hi - lo + 1e-12)
    if exclude_idx is not None:
        out[exclude_idx] = -np.inf
    return out


def ranking_from_scores(scores, ids, exclude_id):
    order = np.argsort(-scores)
    return [ids[i] for i in order if ids[i] != exclude_id]


def validate_rankings(dataset, exp_name, ids):
    expected_len = len(ids) - 1
    rankings_dir = RESULTS_DIR / "rankings" / dataset / exp_name
    for qf in list_query_files(DATASETS[dataset]["gt_dir"]):
        query_name, query_image_id = parse_query_file(qf)
        path = rankings_dir / f"{query_name}.txt"
        if not path.exists():
            raise AssertionError(f"Missing ranking file: {path}")
        with open(path, "r") as f:
            ranking = [line.strip() for line in f if line.strip()]
        if len(ranking) != expected_len:
            raise AssertionError(f"{path} has {len(ranking)} rows; expected {expected_len}")
        if query_image_id in ranking:
            raise AssertionError(f"{path} includes query image {query_image_id}")


def evaluate_and_log(cfg, ids, compute_ap_exe):
    validate_rankings(cfg.dataset, cfg.experiment, ids)
    ap_df = evaluate_experiment(cfg.dataset, cfg.experiment, compute_ap_exe)
    row = base_row(cfg)
    row["map"] = float(ap_df["ap"].mean()) if len(ap_df) else np.nan
    row["num_queries"] = int(len(ap_df))
    append_summary_row(row)
    return row


def run_meanstd_experiment(cfg, compute_ap_exe):
    ids, X = extract_or_load_meanstd(cfg.dataset, cfg.descriptor)
    if np.isnan(X).any():
        raise ValueError("NaN found in feature matrix")
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    detector = create_detector(cfg.descriptor)
    dim = detector.descriptorSize()
    out_dir = RESULTS_DIR / "rankings" / cfg.dataset / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    for qf in tqdm(list_query_files(DATASETS[cfg.dataset]["gt_dir"]), desc=f"Ranking {cfg.dataset}/{cfg.experiment}"):
        query_name, query_image_id, bbox = parse_query_file(qf, return_bbox=True)
        if query_image_id not in id_to_idx:
            continue
        xy, desc = extract_local_descriptor(cfg.dataset, query_image_id, cfg.descriptor, detector)
        qvec = meanstd_from_desc(descriptors_in_bbox(xy, desc, bbox), dim)
        ranking = rank_vector(qvec, X, ids, cfg.metric, query_image_id)
        write_ranking_file(ranking, out_dir / f"{query_name}.txt")

    return evaluate_and_log(cfg, ids, compute_ap_exe)


def require_sklearn():
    try:
        from joblib import dump, load
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise DependencyUnavailable("scikit-learn and joblib are required for BoVW/TF-IDF") from exc
    return MiniBatchKMeans, dump, load


def vocab_path(dataset, descriptor, k):
    return CACHE_DIR / "vocab" / dataset / f"{descriptor}_k{k}.joblib"


def train_or_load_vocab(dataset, descriptor, k, random_state=0, max_descriptors=200_000):
    MiniBatchKMeans, dump, load = require_sklearn()
    path = vocab_path(dataset, descriptor, k)
    if path.exists():
        return load(path)

    detector = create_detector(descriptor)
    rng = np.random.default_rng(random_state)
    sampled = []
    total = 0
    ids = image_ids(dataset)
    for image_id in tqdm(ids, desc=f"Sampling {descriptor.upper()} descriptors for {dataset}/k{k}"):
        _, desc = extract_local_descriptor(dataset, image_id, descriptor, detector)
        if len(desc) == 0:
            continue
        total += len(desc)
        if len(desc) > 300:
            desc = desc[rng.choice(len(desc), size=300, replace=False)]
        sampled.append(desc)
    if not sampled:
        raise ValueError("No descriptors available for vocabulary training")
    X = np.vstack(sampled)
    if len(X) > max_descriptors:
        X = X[rng.choice(len(X), size=max_descriptors, replace=False)]
    if len(X) < k:
        raise ValueError(f"Only {len(X)} sampled descriptors for k={k}")

    print(f"Training MiniBatchKMeans k={k} on {len(X)} sampled descriptors ({total} total seen)")
    vocab = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        verbose=0,
    )
    vocab.fit(X)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(vocab, path)
    return vocab


def hist_from_desc(desc, vocab, k):
    if desc is None or len(desc) == 0:
        return np.zeros(k, dtype=np.float32)
    words = vocab.predict(desc.astype(np.float32))
    return np.bincount(words, minlength=k).astype(np.float32)


def bovw_cache_paths(cfg):
    tag = f"{cfg.descriptor}_bovw_k{cfg.vocab_size}_{cfg.normalization}"
    if cfg.representation in ("bovw_tfidf", "tfidf_bovw"):
        tag = f"{cfg.descriptor}_bovw_tfidf_k{cfg.vocab_size}_{cfg.normalization}"
    out_dir = CACHE_DIR / "features" / cfg.dataset
    return out_dir / f"{tag}.npy", out_dir / f"{tag}_ids.txt", out_dir / f"{tag}_idf.npy"


def build_or_load_bovw_features(cfg, vocab):
    feat_path, ids_path, idf_path = bovw_cache_paths(cfg)
    if feat_path.exists() and ids_path.exists():
        X = np.load(feat_path)
        idf = np.load(idf_path) if idf_path.exists() else None
        return load_ids(ids_path), X, idf

    ids = image_ids(cfg.dataset)
    detector = create_detector(cfg.descriptor)
    raw = []
    for image_id in tqdm(ids, desc=f"Building {cfg.representation} histograms for {cfg.dataset}/k{cfg.vocab_size}"):
        _, desc = extract_local_descriptor(cfg.dataset, image_id, cfg.descriptor, detector)
        raw.append(hist_from_desc(desc, vocab, cfg.vocab_size))
    H = np.vstack(raw).astype(np.float32)

    idf = None
    X = H
    if cfg.representation in ("bovw_tfidf", "tfidf_bovw"):
        df = (H > 0).sum(axis=0)
        idf = (np.log((1 + len(H)) / (1 + df)) + 1).astype(np.float32)
        X = H * idf.reshape(1, -1)
    X = normalize_features(X, cfg.normalization)
    if np.isnan(X).any():
        raise ValueError("NaN found in BoVW feature matrix")

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, X)
    save_ids(ids, ids_path)
    if idf is not None:
        np.save(idf_path, idf)
    return ids, X, idf


def query_bovw_vector(dataset, query_image_id, bbox, vocab, k, normalization, idf=None, descriptor="sift"):
    xy, desc = extract_local_descriptor(dataset, query_image_id, descriptor, create_detector(descriptor))
    hist = hist_from_desc(descriptors_in_bbox(xy, desc, bbox), vocab, k).reshape(1, -1)
    if idf is not None:
        hist = hist * idf.reshape(1, -1)
    return normalize_features(hist, normalization)[0]


def read_image_bgr(dataset, image_id):
    image_path = get_image_path(dataset, image_id)
    if image_path is None:
        raise FileNotFoundError(f"Missing image: {dataset}/{image_id}")
    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def crop_bbox(image, bbox):
    if bbox is None:
        return image
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, round(x1))))
    x2 = int(max(0, min(w, round(x2))))
    y1 = int(max(0, min(h - 1, round(y1))))
    y2 = int(max(0, min(h, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def hog_descriptor(image, size=(128, 128)):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, size, interpolation=cv.INTER_AREA)
    hog = cv.HOGDescriptor(
        _winSize=size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    feat = hog.compute(gray)
    if feat is None:
        return np.zeros(8100, dtype=np.float32)
    return feat.reshape(-1).astype(np.float32)


def hsv_hist_descriptor(image, bins=(16, 8, 8)):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = hist.reshape(-1).astype(np.float32)
    denom = hist.sum() + 1e-12
    return hist / denom


def global_cache_paths(dataset, descriptor, normalization=""):
    tag = descriptor if not normalization else f"{descriptor}_{normalization}"
    out_dir = CACHE_DIR / "features" / dataset
    return out_dir / f"{tag}_features.npy", out_dir / f"{tag}_ids.txt"


def build_or_load_global_features(dataset, descriptor, normalization=""):
    feat_path, ids_path = global_cache_paths(dataset, descriptor, normalization)
    if feat_path.exists() and ids_path.exists():
        return load_ids(ids_path), np.load(feat_path)

    ids = image_ids(dataset)
    feats = []
    for image_id in tqdm(ids, desc=f"Extracting {descriptor.upper()} globals for {dataset}"):
        image = read_image_bgr(dataset, image_id)
        if descriptor == "hog":
            feats.append(hog_descriptor(image))
        elif descriptor == "hsv":
            feats.append(hsv_hist_descriptor(image))
        else:
            raise ValueError(f"Unknown global descriptor: {descriptor}")
    X = np.vstack(feats).astype(np.float32)
    X = normalize_features(X, normalization)
    if np.isnan(X).any():
        raise ValueError("NaN found in global feature matrix")

    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, X)
    save_ids(ids, ids_path)
    return ids, X


def query_global_vector(dataset, query_image_id, bbox, descriptor, normalization=""):
    image = crop_bbox(read_image_bgr(dataset, query_image_id), bbox)
    if descriptor == "hog":
        vec = hog_descriptor(image)
    elif descriptor == "hsv":
        vec = hsv_hist_descriptor(image)
    else:
        raise ValueError(f"Unknown global descriptor: {descriptor}")
    return normalize_features(vec.reshape(1, -1), normalization)[0]


def rerank_with_query_expansion(query_vec, X, ids, initial_ranking, metric, m, alpha=0.5,
                                normalization="l2"):
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    top_ids = [image_id for image_id in initial_ranking[:m] if image_id in id_to_idx]
    if not top_ids:
        return query_vec
    top = X[[id_to_idx[image_id] for image_id in top_ids]]
    expanded = alpha * query_vec.reshape(1, -1) + (1 - alpha) * top.mean(axis=0, keepdims=True)
    return normalize_features(expanded, normalization)[0]


def run_bovw_experiment(cfg, compute_ap_exe):
    vocab = train_or_load_vocab(cfg.dataset, cfg.descriptor, cfg.vocab_size)
    ids, X, idf = build_or_load_bovw_features(cfg, vocab)
    out_dir = RESULTS_DIR / "rankings" / cfg.dataset / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    qe_m = (cfg.extra_params or {}).get("query_expansion_m")
    qe_alpha = (cfg.extra_params or {}).get("query_expansion_alpha", 0.5)

    for qf in tqdm(list_query_files(DATASETS[cfg.dataset]["gt_dir"]), desc=f"Ranking {cfg.dataset}/{cfg.experiment}"):
        query_name, query_image_id, bbox = parse_query_file(qf, return_bbox=True)
        qvec = query_bovw_vector(
            cfg.dataset, query_image_id, bbox, vocab, cfg.vocab_size,
            cfg.normalization, idf, descriptor=cfg.descriptor
        )
        ranking = rank_vector(qvec, X, ids, cfg.metric, query_image_id)
        if qe_m:
            qvec = rerank_with_query_expansion(
                qvec, X, ids, ranking, cfg.metric, qe_m, alpha=qe_alpha,
                normalization=cfg.normalization,
            )
            ranking = rank_vector(qvec, X, ids, cfg.metric, query_image_id)
        write_ranking_file(ranking, out_dir / f"{query_name}.txt")

    return evaluate_and_log(cfg, ids, compute_ap_exe)


def run_global_experiment(cfg, compute_ap_exe):
    normalization = cfg.normalization or ("l2" if cfg.descriptor == "hog" else "l1")
    ids, X = build_or_load_global_features(cfg.dataset, cfg.descriptor, normalization)
    out_dir = RESULTS_DIR / "rankings" / cfg.dataset / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    for qf in tqdm(list_query_files(DATASETS[cfg.dataset]["gt_dir"]), desc=f"Ranking {cfg.dataset}/{cfg.experiment}"):
        query_name, query_image_id, bbox = parse_query_file(qf, return_bbox=True)
        qvec = query_global_vector(cfg.dataset, query_image_id, bbox, cfg.descriptor, normalization)
        ranking = rank_vector(qvec, X, ids, cfg.metric, query_image_id)
        write_ranking_file(ranking, out_dir / f"{query_name}.txt")

    return evaluate_and_log(cfg, ids, compute_ap_exe)


def base_bovw_qe_components(dataset):
    cfg = ExperimentConfig(
        dataset=dataset,
        experiment="sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25",
        descriptor="sift",
        representation="bovw",
        metric="cosine",
        normalization="l2",
        vocab_size=1024,
        extra_params={"random_state": 0, "query_expansion_m": 3, "query_expansion_alpha": 0.25},
    )
    vocab = train_or_load_vocab(cfg.dataset, cfg.descriptor, cfg.vocab_size)
    ids, X, idf = build_or_load_bovw_features(cfg, vocab)
    return cfg, vocab, ids, X, idf


def base_bovw_qe_scores(dataset, query_image_id, bbox, vocab, ids, X, idf):
    qvec = query_bovw_vector(dataset, query_image_id, bbox, vocab, 1024, "l2", idf, descriptor="sift")
    initial_ranking = rank_vector(qvec, X, ids, "cosine", query_image_id)
    qvec = rerank_with_query_expansion(
        qvec, X, ids, initial_ranking, "cosine", 3, alpha=0.25, normalization="l2"
    )
    return similarity_scores(qvec, X, "cosine")


def run_fusion_experiment(cfg, compute_ap_exe):
    _, vocab, ids, local_X, idf = base_bovw_qe_components(cfg.dataset)
    params = cfg.extra_params or {}
    global_specs = params.get("global_specs", [])
    weights = params.get("weights", [])
    score_norm = params.get("score_norm", "minmax")
    if len(global_specs) + 1 != len(weights):
        raise ValueError("Fusion weights must include local plus one weight per global descriptor")

    global_data = []
    for spec in global_specs:
        normalization = spec.get("normalization") or ("l2" if spec["descriptor"] == "hog" else "l1")
        global_ids, global_X = build_or_load_global_features(cfg.dataset, spec["descriptor"], normalization)
        if global_ids != ids:
            raise ValueError("Global feature ids do not match local feature ids")
        global_data.append((spec, normalization, global_X))

    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    out_dir = RESULTS_DIR / "rankings" / cfg.dataset / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    for qf in tqdm(list_query_files(DATASETS[cfg.dataset]["gt_dir"]), desc=f"Ranking {cfg.dataset}/{cfg.experiment}"):
        query_name, query_image_id, bbox = parse_query_file(qf, return_bbox=True)
        exclude_idx = id_to_idx.get(query_image_id)
        local_scores = base_bovw_qe_scores(cfg.dataset, query_image_id, bbox, vocab, ids, local_X, idf)
        final_scores = weights[0] * normalize_scores(local_scores, exclude_idx, score_norm)
        for weight, (spec, normalization, global_X) in zip(weights[1:], global_data):
            qvec = query_global_vector(cfg.dataset, query_image_id, bbox, spec["descriptor"], normalization)
            global_scores = similarity_scores(qvec, global_X, spec["metric"])
            final_scores += weight * normalize_scores(global_scores, exclude_idx, score_norm)
        if np.isnan(final_scores[np.isfinite(final_scores)]).any():
            raise ValueError("NaN found in fusion scores")
        ranking = ranking_from_scores(final_scores, ids, query_image_id)
        write_ranking_file(ranking, out_dir / f"{query_name}.txt")

    return evaluate_and_log(cfg, ids, compute_ap_exe)


def read_ranking(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def descriptor_from_experiment_name(exp_name):
    if str(exp_name).startswith("rootsift_"):
        return "rootsift"
    return "sift"


def vocab_size_from_experiment_name(exp_name, default=1024):
    match = re.search(r"_k(\d+)_", str(exp_name))
    return int(match.group(1)) if match else default


def base_bovw_config_from_name(dataset, exp_name):
    base = str(exp_name).split("_spatial_verify", 1)[0].split("_verified_qe", 1)[0]
    match = re.match(r"(?P<descriptor>sift|rootsift)_bovw(?P<tfidf>_tfidf)?_k(?P<k>\d+)_(?P<norm>[^_]+)_(?P<metric>.+)", base)
    if not match:
        raise ValueError(f"Cannot parse BoVW base experiment: {exp_name}")
    representation = "bovw_tfidf" if match.group("tfidf") else "bovw"
    metric = match.group("metric")
    extra = {"random_state": 0}
    qe_match = re.search(r"_qe_top(?P<m>\d+)_alpha(?P<alpha>[0-9p]+)", metric)
    if qe_match:
        metric = metric[:qe_match.start()]
        extra["query_expansion_m"] = int(qe_match.group("m"))
        extra["query_expansion_alpha"] = float(qe_match.group("alpha").replace("p", "."))
    return ExperimentConfig(
        dataset=dataset,
        experiment=base,
        descriptor=match.group("descriptor"),
        representation=representation,
        metric=metric,
        normalization=match.group("norm"),
        vocab_size=int(match.group("k")),
        extra_params=extra,
    )


def bovw_query_components(dataset, query_image_id, bbox, cfg, vocab, ids, X, idf):
    qvec = query_bovw_vector(
        dataset, query_image_id, bbox, vocab, cfg.vocab_size,
        cfg.normalization, idf, descriptor=cfg.descriptor
    )
    qe_m = (cfg.extra_params or {}).get("query_expansion_m")
    qe_alpha = (cfg.extra_params or {}).get("query_expansion_alpha", 0.5)
    if qe_m:
        ranking = rank_vector(qvec, X, ids, cfg.metric, query_image_id)
        qvec = rerank_with_query_expansion(
            qvec, X, ids, ranking, cfg.metric, qe_m, alpha=qe_alpha,
            normalization=cfg.normalization,
        )
    return qvec


def base_original_similarity_by_id(dataset, base_exp, query_image_id, bbox, ids):
    try:
        cfg = base_bovw_config_from_name(dataset, base_exp)
        vocab = train_or_load_vocab(cfg.dataset, cfg.descriptor, cfg.vocab_size)
        feature_ids, X, idf = build_or_load_bovw_features(cfg, vocab)
        if feature_ids != ids:
            return {}
        qvec = bovw_query_components(dataset, query_image_id, bbox, cfg, vocab, ids, X, idf)
        scores = similarity_scores(qvec, X, cfg.metric)
        return {image_id: float(score) for image_id, score in zip(ids, normalize_scores(scores))}
    except Exception as exc:
        print(f"Could not compute original similarities for {dataset}/{base_exp}: {exc}")
        return {}


def spatial_verification_score(q_xy, q_desc, db_xy, db_desc, ratio, scoring="inlier_count",
                               min_matches=0, min_inliers=0, original_score=0.0):
    if len(q_desc) < 4 or len(db_desc) < 4:
        return 0.0
    matcher = cv.BFMatcher(cv.NORM_L2)
    matches = matcher.knnMatch(q_desc.astype(np.float32), db_desc.astype(np.float32), k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < max(4, min_matches):
        return 0.0
    src = np.float32([q_xy[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([db_xy[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    if mask is None:
        return 0.0
    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return 0.0
    inlier_ratio = inliers / max(len(good), 1)
    if scoring == "inlier_ratio":
        return float(inlier_ratio)
    if scoring == "inliers_plus_ratio":
        return float(inliers + 10.0 * inlier_ratio)
    if scoring == "inliers_plus_original_score":
        return float(inliers + 0.01 * original_score)
    if scoring == "normalized_inliers":
        return float(inliers / max(math.log(2.0 + len(db_xy)), 1e-12))
    if scoring == "thresholded_inliers":
        return float(inliers if inliers >= 8 else 0.0)
    if scoring == "inlier_count":
        return float(inliers)
    raise ValueError(f"Unknown spatial verification scoring mode: {scoring}")


def run_spatial_verification_experiment(cfg, compute_ap_exe):
    params = cfg.extra_params or {}
    top_n = int(params.get("top_n", 50))
    ratio = float(params.get("ratio", 0.75))
    scoring = params.get("score", "inlier_count")
    min_matches = int(params.get("min_matches", 0) or 0)
    min_inliers = int(params.get("min_inliers", 0) or 0)
    base_exp = params.get("base_experiment", "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25")
    spatial_descriptor = params.get("spatial_descriptor") or descriptor_from_experiment_name(base_exp)
    ids = image_ids(cfg.dataset)
    base_dir = RESULTS_DIR / "rankings" / cfg.dataset / base_exp
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing base rankings: {base_dir}")

    detector = create_detector(spatial_descriptor)
    out_dir = RESULTS_DIR / "rankings" / cfg.dataset / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    for qf in tqdm(list_query_files(DATASETS[cfg.dataset]["gt_dir"]), desc=f"Spatial verify {cfg.dataset}/{cfg.experiment}"):
        query_name, query_image_id, bbox = parse_query_file(qf, return_bbox=True)
        base_path = base_dir / f"{query_name}.txt"
        original = read_ranking(base_path)
        try:
            q_xy_all, q_desc_all = extract_local_descriptor(cfg.dataset, query_image_id, spatial_descriptor, detector)
            if bbox is not None and len(q_desc_all):
                x1, y1, x2, y2 = bbox
                keep = (q_xy_all[:, 0] >= x1) & (q_xy_all[:, 0] <= x2) & (q_xy_all[:, 1] >= y1) & (q_xy_all[:, 1] <= y2)
                q_xy, q_desc = q_xy_all[keep], q_desc_all[keep]
            else:
                q_xy, q_desc = q_xy_all, q_desc_all

            reranked = []
            original_scores = {}
            if scoring == "inliers_plus_original_score":
                original_scores = base_original_similarity_by_id(cfg.dataset, base_exp, query_image_id, bbox, ids)
            for original_idx, cand_id in enumerate(original[:top_n]):
                db_xy, db_desc = extract_local_descriptor(cfg.dataset, cand_id, spatial_descriptor, detector)
                score = spatial_verification_score(
                    q_xy, q_desc, db_xy, db_desc, ratio,
                    scoring=scoring,
                    min_matches=min_matches,
                    min_inliers=min_inliers,
                    original_score=original_scores.get(cand_id, 0.0),
                )
                reranked.append((score, original_idx, cand_id))
            reranked.sort(key=lambda item: (-item[0], item[1]))
            ranking = [cand_id for _, _, cand_id in reranked] + original[top_n:]
        except Exception as exc:
            print(f"Spatial verification fallback for {cfg.dataset}/{query_name}: {exc}")
            ranking = original
        write_ranking_file(ranking, out_dir / f"{query_name}.txt")

    return evaluate_and_log(cfg, ids, compute_ap_exe)


def run_verified_qe_experiment(cfg, compute_ap_exe):
    params = cfg.extra_params or {}
    base_exp = params["base_experiment"]
    top_m = int(params.get("top_m", 3))
    alpha = float(params.get("alpha", 0.5))
    bovw_cfg = base_bovw_config_from_name(cfg.dataset, base_exp)
    vocab = train_or_load_vocab(bovw_cfg.dataset, bovw_cfg.descriptor, bovw_cfg.vocab_size)
    ids, X, idf = build_or_load_bovw_features(bovw_cfg, vocab)
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}
    base_dir = RESULTS_DIR / "rankings" / cfg.dataset / base_exp
    if not base_dir.exists():
        raise FileNotFoundError(f"Missing base rankings: {base_dir}")

    out_dir = RESULTS_DIR / "rankings" / cfg.dataset / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    for qf in tqdm(list_query_files(DATASETS[cfg.dataset]["gt_dir"]), desc=f"Verified QE {cfg.dataset}/{cfg.experiment}"):
        query_name, query_image_id, bbox = parse_query_file(qf, return_bbox=True)
        base_ranking = read_ranking(base_dir / f"{query_name}.txt")
        qvec = bovw_query_components(cfg.dataset, query_image_id, bbox, bovw_cfg, vocab, ids, X, idf)
        top_ids = [image_id for image_id in base_ranking[:top_m] if image_id in id_to_idx]
        if top_ids:
            top = X[[id_to_idx[image_id] for image_id in top_ids]]
            qvec = alpha * qvec.reshape(1, -1) + (1.0 - alpha) * top.mean(axis=0, keepdims=True)
            qvec = normalize_features(qvec, bovw_cfg.normalization)[0]
        if np.isnan(qvec).any():
            raise ValueError("NaN found in verified QE query vector")
        ranking = rank_vector(qvec, X, ids, bovw_cfg.metric, query_image_id)
        write_ranking_file(ranking, out_dir / f"{query_name}.txt")

    return evaluate_and_log(cfg, ids, compute_ap_exe)


def log_failure(cfg, error):
    row = base_row(cfg)
    row["status"] = "failed/unavailable" if isinstance(error, DependencyUnavailable) else "failed"
    row["error"] = str(error)
    append_summary_row(row)
    print(f"Skipped {cfg.dataset}/{cfg.experiment}: {error}")
    return row


def run_experiment(cfg, compute_ap_exe=None):
    print(f"Running {cfg.dataset}/{cfg.experiment}")
    compute_ap_exe = compute_ap_exe or compute_ap_executable()
    try:
        if successful_result_exists(cfg):
            row = base_row(cfg)
            row["status"] = "skipped/existing_success"
            print(f"Skipping existing successful result: {cfg.dataset}/{cfg.experiment}")
            return row
        if cfg.representation == "meanstd":
            return run_meanstd_experiment(cfg, compute_ap_exe)
        if cfg.representation == "global":
            return run_global_experiment(cfg, compute_ap_exe)
        if cfg.representation in ("bovw", "bovw_tfidf", "tfidf_bovw"):
            return run_bovw_experiment(cfg, compute_ap_exe)
        if cfg.representation == "late_fusion":
            return run_fusion_experiment(cfg, compute_ap_exe)
        if cfg.representation == "spatial_verification":
            return run_spatial_verification_experiment(cfg, compute_ap_exe)
        if cfg.representation == "verified_qe":
            return run_verified_qe_experiment(cfg, compute_ap_exe)
        raise ValueError(f"Unknown representation: {cfg.representation}")
    except Exception as exc:
        return log_failure(cfg, exc)


def surf_configs(datasets):
    for dataset in datasets:
        for metric in ["cosine", "euclidean"]:
            yield ExperimentConfig(
                dataset=dataset,
                experiment=f"surf_meanstd_bbox_{metric}",
                descriptor="surf",
                representation="meanstd",
                metric=metric,
            )


def bovw_configs(datasets, ks=(128, 256, 512, 1024), normalizations=("l1", "l2", "sqrt_l2"),
                 metrics=("cosine", "euclidean", "chisquare", "hist_intersection")):
    for dataset in datasets:
        for k in ks:
            for norm in normalizations:
                for metric in metrics:
                    yield ExperimentConfig(
                        dataset=dataset,
                        experiment=f"sift_bovw_k{k}_{norm}_{metric}",
                        descriptor="sift",
                        representation="bovw",
                        metric=metric,
                        normalization=norm,
                        vocab_size=k,
                        extra_params={"random_state": 0},
                    )


def tfidf_configs(datasets, ks=(256, 512, 1024), normalizations=("l2", "sqrt_l2"),
                  metrics=("cosine", "chisquare")):
    for dataset in datasets:
        for k in ks:
            for norm in normalizations:
                for metric in metrics:
                    yield ExperimentConfig(
                        dataset=dataset,
                        experiment=f"sift_bovw_tfidf_k{k}_{norm}_{metric}",
                        descriptor="sift",
                        representation="bovw_tfidf",
                        metric=metric,
                        normalization=norm,
                        vocab_size=k,
                        extra_params={"random_state": 0},
                    )


def global_configs(datasets):
    for dataset in datasets:
        yield ExperimentConfig(
            dataset=dataset,
            experiment="hog_bbox_cosine",
            descriptor="hog",
            representation="global",
            metric="cosine",
            normalization="l2",
            extra_params={"query": "bbox_crop", "image_size": [128, 128]},
        )
        for metric in ("cosine", "chisquare", "hist_intersection"):
            name_metric = "hist_intersection" if metric == "hist_intersection" else metric
            yield ExperimentConfig(
                dataset=dataset,
                experiment=f"hsv_bbox_{name_metric}",
                descriptor="hsv",
                representation="global",
                metric=metric,
                normalization="l1",
                extra_params={"query": "bbox_crop", "bins": [16, 8, 8]},
            )


def weight_tag(*weights):
    return "_".join(str(w).replace(".", "p") for w in weights)


def ratio_tag(value):
    return str(value).replace(".", "p")


def fusion_configs(datasets):
    base = "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25"
    for dataset in datasets:
        for wl, wg in ((0.95, 0.05), (0.90, 0.10), (0.80, 0.20)):
            yield ExperimentConfig(
                dataset=dataset,
                experiment=f"{base}_fusion_hog_cosine_w{weight_tag(wl, wg)}",
                descriptor="sift+hog",
                representation="late_fusion",
                metric="cosine",
                normalization="minmax",
                vocab_size=1024,
                extra_params={
                    "base_experiment": base,
                    "global_specs": [{"descriptor": "hog", "metric": "cosine", "normalization": "l2"}],
                    "weights": [wl, wg],
                    "score_norm": "minmax",
                },
            )
        for metric, metric_tag in (
            ("cosine", "cosine"),
            ("chisquare", "chisquare"),
            ("hist_intersection", "histint"),
        ):
            for wl, wg in ((0.95, 0.05), (0.90, 0.10), (0.80, 0.20)):
                yield ExperimentConfig(
                    dataset=dataset,
                    experiment=f"{base}_fusion_hsv_{metric_tag}_w{weight_tag(wl, wg)}",
                    descriptor="sift+hsv",
                    representation="late_fusion",
                    metric=metric,
                    normalization="minmax",
                    vocab_size=1024,
                    extra_params={
                        "base_experiment": base,
                        "global_specs": [{"descriptor": "hsv", "metric": metric, "normalization": "l1"}],
                        "weights": [wl, wg],
                        "score_norm": "minmax",
                    },
                )
        for wl, wh, wv in ((0.90, 0.05, 0.05), (0.80, 0.10, 0.10)):
            yield ExperimentConfig(
                dataset=dataset,
                experiment=f"{base}_fusion_hog_hsv_cosine_histint_w{weight_tag(wl, wh, wv)}",
                descriptor="sift+hog+hsv",
                representation="late_fusion",
                metric="cosine+hist_intersection",
                normalization="minmax",
                vocab_size=1024,
                extra_params={
                    "base_experiment": base,
                    "global_specs": [
                        {"descriptor": "hog", "metric": "cosine", "normalization": "l2"},
                        {"descriptor": "hsv", "metric": "hist_intersection", "normalization": "l1"},
                    ],
                    "weights": [wl, wh, wv],
                    "score_norm": "minmax",
                },
            )


def spatial_verification_configs(datasets, oxford_grid=True):
    base = "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25"
    for dataset in datasets:
        if dataset == "oxford" and oxford_grid:
            for top_n in (50, 100):
                for ratio in (0.70, 0.75, 0.80):
                    yield ExperimentConfig(
                        dataset=dataset,
                        experiment=f"{base}_spatial_verify_top{top_n}_ratio{str(ratio).replace('.', 'p')}_inliers",
                        descriptor="sift",
                        representation="spatial_verification",
                        metric="inlier_count",
                        normalization="",
                        vocab_size=1024,
                        extra_params={"base_experiment": base, "top_n": top_n, "ratio": ratio, "score": "inlier_count"},
                    )
        else:
            yield ExperimentConfig(
                dataset=dataset,
                experiment=f"{base}_spatial_verify_top50_ratio0p75_inliers",
                descriptor="sift",
                representation="spatial_verification",
                metric="inlier_count",
                normalization="",
                vocab_size=1024,
                extra_params={"base_experiment": base, "top_n": 50, "ratio": 0.75, "score": "inlier_count"},
            )


def spatial_score_tag(scoring):
    return {
        "inlier_count": "inliers",
        "inlier_ratio": "inlierratio",
        "inliers_plus_ratio": "inliersplusratio",
        "inliers_plus_original_score": "inliersplusorig",
        "normalized_inliers": "norminliers",
        "thresholded_inliers": "thresholdedinliers",
    }[scoring]


def spatial_threshold_tag(min_matches=0, min_inliers=0):
    if min_matches or min_inliers:
        return f"_minm{min_matches}_mini{min_inliers}"
    return ""


def spatial_experiment_name(base, top_n, ratio, scoring, min_matches=0, min_inliers=0):
    return (
        f"{base}_spatial_verify_top{top_n}_ratio{ratio_tag(ratio)}_"
        f"{spatial_score_tag(scoring)}{spatial_threshold_tag(min_matches, min_inliers)}"
    )


def make_spatial_config(dataset, base, top_n, ratio, scoring="inlier_count",
                        min_matches=0, min_inliers=0):
    descriptor = descriptor_from_experiment_name(base)
    return ExperimentConfig(
        dataset=dataset,
        experiment=spatial_experiment_name(base, top_n, ratio, scoring, min_matches, min_inliers),
        descriptor=descriptor,
        representation="spatial_verification",
        metric=scoring,
        normalization="",
        vocab_size=vocab_size_from_experiment_name(base),
        extra_params={
            "base_experiment": base,
            "top_n": top_n,
            "ratio": ratio,
            "score": scoring,
            "min_matches": min_matches,
            "min_inliers": min_inliers,
            "spatial_descriptor": descriptor,
        },
    )


def spatial_tuning_configs(dataset="oxford"):
    base = "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25"
    for top_n in (100, 150):
        for ratio in (0.60, 0.65, 0.70):
            for scoring in ("inlier_count", "inliers_plus_ratio"):
                for min_matches, min_inliers in ((0, 0), (8, 6)):
                    yield make_spatial_config(dataset, base, top_n, ratio, scoring, min_matches, min_inliers)


def alternative_spatial_base_configs(dataset, top_n, ratio, scoring, min_matches, min_inliers):
    bases = [
        "sift_bovw_k1024_l2_cosine",
        "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5",
        "sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p75",
        "sift_bovw_k256_l2_chisquare_qe_top3_alpha0p75",
    ]
    for base in bases:
        yield make_spatial_config(dataset, base, top_n, ratio, scoring, min_matches, min_inliers)


def phase5_bovw_configs(dataset="oxford"):
    for descriptor, ks in (("rootsift", (1024, 2048, 4096)), ("sift", (2048, 4096))):
        for k in ks:
            yield ExperimentConfig(
                dataset=dataset,
                experiment=f"{descriptor}_bovw_k{k}_l2_cosine",
                descriptor=descriptor,
                representation="bovw",
                metric="cosine",
                normalization="l2",
                vocab_size=k,
                extra_params={"random_state": 0, "phase": 5},
            )


def phase5_spatial_configs(dataset="oxford"):
    for cfg in phase5_bovw_configs(dataset):
        yield make_spatial_config(dataset, cfg.experiment, 150, 0.65, "inlier_count")


def alpha_tag(alpha):
    return str(alpha).replace(".", "p")


def make_verified_qe_config(dataset, base, top_m, alpha):
    bovw_cfg = base_bovw_config_from_name(dataset, base)
    return ExperimentConfig(
        dataset=dataset,
        experiment=f"{base}_verified_qe_top{top_m}_alpha{alpha_tag(alpha)}",
        descriptor=bovw_cfg.descriptor,
        representation="verified_qe",
        metric=bovw_cfg.metric,
        normalization=bovw_cfg.normalization,
        vocab_size=bovw_cfg.vocab_size,
        extra_params={
            "base_experiment": base,
            "top_m": top_m,
            "alpha": alpha,
            "phase": 5,
        },
    )


def verified_qe_config_from_name(dataset, exp_name):
    match = re.match(r"(?P<base>.+)_verified_qe_top(?P<m>\d+)_alpha(?P<alpha>[0-9p]+)$", exp_name)
    if not match:
        raise ValueError(f"Cannot parse verified QE experiment: {exp_name}")
    return make_verified_qe_config(
        dataset,
        match.group("base"),
        int(match.group("m")),
        float(match.group("alpha").replace("p", ".")),
    )


def phase5_verified_qe_configs(dataset="oxford"):
    bases = [
        "sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers",
        "rootsift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers",
        "rootsift_bovw_k2048_l2_cosine_spatial_verify_top150_ratio0p65_inliers",
        "sift_bovw_k2048_l2_cosine_spatial_verify_top150_ratio0p65_inliers",
    ]
    for base in bases:
        for top_m in (3, 5):
            for alpha in (0.5, 0.75):
                yield make_verified_qe_config(dataset, base, top_m, alpha)


def phase5_scoring_refinement_configs(dataset="oxford"):
    base = "sift_bovw_k1024_l2_cosine"
    for scoring in ("inliers_plus_original_score", "normalized_inliers", "thresholded_inliers"):
        yield make_spatial_config(dataset, base, 150, 0.65, scoring)


def query_expansion_configs(rows, top_ms=(5, 10), alphas=(0.5,), limit=3):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("representation") in ("bovw", "bovw_tfidf", "tfidf_bovw")]
    ok = sorted(ok, key=lambda r: r.get("map", -1), reverse=True)[:limit]
    for row in ok:
        for m in top_ms:
            for alpha in alphas:
                extra = json.loads(row["extra_params"]) if isinstance(row["extra_params"], str) else dict(row["extra_params"] or {})
                extra["query_expansion_m"] = m
                extra["query_expansion_alpha"] = alpha
                alpha_tag = str(alpha).replace(".", "p")
                yield ExperimentConfig(
                    dataset=row["dataset"],
                    experiment=f"{row['experiment']}_qe_top{m}_alpha{alpha_tag}",
                    descriptor=row["descriptor"],
                    representation=row["representation"],
                    metric=row["metric"],
                    normalization=row["normalization"],
                    vocab_size=None if pd.isna(row["vocab_size"]) else int(row["vocab_size"]),
                    extra_params=extra,
                )


def top_successful_configs(dataset="oxford", n=5):
    out = summary_path(True)
    if not out.exists():
        write_sorted_summary()
    df = pd.read_csv(out)
    df = df[
        df["status"].eq("ok")
        & df["dataset"].eq(dataset)
        & df["descriptor"].eq("sift")
        & df["representation"].isin(["bovw", "bovw_tfidf", "tfidf_bovw"])
    ].copy()
    if df.empty:
        return []
    df = df.sort_values("map", ascending=False)
    return df.head(n).to_dict("records")


def config_from_summary_row(row, dataset=None):
    return ExperimentConfig(
        dataset=dataset or row["dataset"],
        experiment=row["experiment"] if dataset is None else str(row["experiment"]).replace(f"{row['dataset']}/", ""),
        descriptor=row["descriptor"],
        representation=row["representation"],
        metric=row["metric"],
        normalization=row["normalization"],
        vocab_size=None if pd.isna(row["vocab_size"]) else int(row["vocab_size"]),
        extra_params=json.loads(row["extra_params"]) if isinstance(row["extra_params"], str) and row["extra_params"] else {},
    )


def equivalent_dataset_config(row, dataset):
    cfg = config_from_summary_row(row, dataset=dataset)
    if row["dataset"] != dataset:
        cfg.experiment = cfg.experiment
    return cfg


def extended_oxford_tfidf_configs():
    return list(tfidf_configs(
        ["oxford"],
        ks=(64, 128, 256, 384, 512, 768, 1024),
        normalizations=("l1", "l2", "sqrt_l2"),
        metrics=("chisquare", "cosine"),
    ))


def extended_oxford_plain_configs():
    return list(bovw_configs(
        ["oxford"],
        ks=(64, 384, 768, 1024),
        normalizations=("l2", "sqrt_l2"),
        metrics=("chisquare", "cosine"),
    ))


def run_configs(configs, compute_ap_exe=None):
    compute_ap_exe = compute_ap_exe or compute_ap_executable()
    rows = []
    started = time.perf_counter()
    for cfg in configs:
        rows.append(run_experiment(cfg, compute_ap_exe))
    elapsed = time.perf_counter() - started
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    skipped_count = sum(1 for row in rows if str(row.get("status", "")).startswith("skipped"))
    failed_count = len(rows) - ok_count - skipped_count
    print(
        f"Runtime summary: {len(rows)} configs, {ok_count} new successes, "
        f"{skipped_count} skipped, {failed_count} failed, {elapsed:.1f}s elapsed"
    )
    return pd.DataFrame(rows)


def run_safe_extended_sweep():
    before = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    compute_ap_exe = compute_ap_executable()
    rows = []

    print("Phase 1: Extended Oxford TF-IDF BoVW sweep")
    rows.extend(run_configs(extended_oxford_tfidf_configs(), compute_ap_exe).to_dict("records"))
    print("Phase 1: Extended Oxford plain BoVW new-K sweep")
    rows.extend(run_configs(extended_oxford_plain_configs(), compute_ap_exe).to_dict("records"))

    print("Phase 2: Query expansion on top 5 Oxford configs")
    top5 = top_successful_configs("oxford", 5)
    qe_cfgs = list(query_expansion_configs(top5, top_ms=(3, 5, 10), alphas=(0.25, 0.5, 0.75), limit=5))
    rows.extend(run_configs(qe_cfgs, compute_ap_exe).to_dict("records"))

    print("Phase 3: Paris validation for top 5 Oxford configs overall")
    top5_after_qe = top_successful_configs("oxford", 5)
    paris_cfgs = [equivalent_dataset_config(row, "paris") for row in top5_after_qe]
    rows.extend(run_configs(paris_cfgs, compute_ap_exe).to_dict("records"))

    write_sorted_summary()
    after = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    newly_added = max(0, len(after) - len(before))
    print(f"Newly added CSV rows: {newly_added}")
    print_final_report(newly_added)
    return pd.DataFrame(rows)


def run_global_fusion_spatial_phase():
    before = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    compute_ap_exe = compute_ap_executable()
    rows = []

    print("Phase 1: Global descriptor baselines")
    rows.extend(run_configs(global_configs(["oxford", "paris"]), compute_ap_exe).to_dict("records"))

    print("Phase 2: Late fusion with best SIFT BoVW/QE config")
    rows.extend(run_configs(fusion_configs(["oxford", "paris"]), compute_ap_exe).to_dict("records"))

    print("Phase 3: Spatial verification reranking on Oxford")
    rows.extend(run_configs(spatial_verification_configs(["oxford"]), compute_ap_exe).to_dict("records"))

    write_sorted_summary()
    df = pd.read_csv(summary_path(False))
    ox_sv = df[
        df["dataset"].eq("oxford")
        & df["status"].eq("ok")
        & df["representation"].eq("spatial_verification")
    ].copy()
    ox_sv_best = ox_sv["map"].max() if not ox_sv.empty else np.nan
    if pd.notna(ox_sv_best) and ox_sv_best > 0.218428:
        best_row = ox_sv.sort_values("map", ascending=False).iloc[0]
        print("Oxford spatial verification improved; running best setting on Paris")
        rows.extend(run_configs([equivalent_dataset_config(best_row, "paris")], compute_ap_exe).to_dict("records"))
    else:
        print("Oxford spatial verification did not improve over 0.218428; skipping Paris spatial verification")

    write_sorted_summary()
    after = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    newly_added = max(0, len(after) - len(before))
    print_final_report(newly_added)
    print_improvement_report()
    return pd.DataFrame(rows)


def best_result_for_experiments(dataset, experiments, representation="spatial_verification"):
    write_sorted_summary()
    df = pd.read_csv(summary_path(False))
    exp_set = set(experiments)
    ok = df[
        df["dataset"].eq(dataset)
        & df["status"].eq("ok")
        & df["representation"].eq(representation)
        & df["experiment"].isin(exp_set)
    ].copy()
    if ok.empty:
        return None
    return ok.sort_values("map", ascending=False).iloc[0].to_dict()


def spatial_params_from_row(row):
    params = json.loads(row["extra_params"]) if isinstance(row["extra_params"], str) else dict(row["extra_params"] or {})
    return {
        "base_experiment": params.get("base_experiment"),
        "top_n": int(params.get("top_n", 100)),
        "ratio": float(params.get("ratio", 0.7)),
        "score": params.get("score", row.get("metric", "inlier_count")),
        "min_matches": int(params.get("min_matches", 0) or 0),
        "min_inliers": int(params.get("min_inliers", 0) or 0),
    }


def maybe_run_fisher_smoke(compute_ap_exe):
    if "run_fisher_vector_experiment" not in globals():
        cfg = ExperimentConfig(
            dataset="oxford",
            experiment="sift_fisher_gmm16_diag_power_l2_cosine",
            descriptor="sift",
            representation="fisher_vector",
            metric="cosine",
            normalization="power_l2",
            vocab_size=16,
            extra_params={"components": 16, "covariance": "diag"},
        )
        return [log_failure(cfg, DependencyUnavailable("Fisher Vector implementation is unavailable"))]
    rows = []
    for components in (16, 32):
        cfg = ExperimentConfig(
            dataset="oxford",
            experiment=f"sift_fisher_gmm{components}_diag_power_l2_cosine",
            descriptor="sift",
            representation="fisher_vector",
            metric="cosine",
            normalization="power_l2",
            vocab_size=components,
            extra_params={"components": components, "covariance": "diag", "normalization": "power_l2"},
        )
        rows.append(run_experiment(cfg, compute_ap_exe))
    return rows


def run_spatial_tuning_phase(run_fisher=False):
    before = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    compute_ap_exe = compute_ap_executable()
    rows = []

    print("Phase 1: Controlled Oxford spatial verification tuning")
    phase1_cfgs = list(spatial_tuning_configs("oxford"))
    rows.extend(run_configs(phase1_cfgs, compute_ap_exe).to_dict("records"))
    best_phase1 = best_result_for_experiments("oxford", [cfg.experiment for cfg in phase1_cfgs])
    if best_phase1 is None:
        print("No successful Oxford spatial tuning rows found; skipping Paris and alternative bases")
        write_sorted_summary()
        print_final_report()
        return pd.DataFrame(rows)
    best_params = spatial_params_from_row(best_phase1)
    print(
        "Best Oxford spatial hyperparameters: "
        f"topN={best_params['top_n']}, ratio={best_params['ratio']}, "
        f"score={best_params['score']}, min_matches={best_params['min_matches']}, "
        f"min_inliers={best_params['min_inliers']}"
    )

    print("Phase 2: Apply best Oxford setting to Paris")
    paris_cfg = make_spatial_config(
        "paris",
        "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25",
        best_params["top_n"],
        best_params["ratio"],
        best_params["score"],
        best_params["min_matches"],
        best_params["min_inliers"],
    )
    rows.extend(run_configs([paris_cfg], compute_ap_exe).to_dict("records"))

    print("Phase 3: Apply best setting to alternative Oxford base configs")
    alt_cfgs = list(alternative_spatial_base_configs(
        "oxford",
        best_params["top_n"],
        best_params["ratio"],
        best_params["score"],
        best_params["min_matches"],
        best_params["min_inliers"],
    ))
    rows.extend(run_configs(alt_cfgs, compute_ap_exe).to_dict("records"))
    best_alt = best_result_for_experiments("oxford", [cfg.experiment for cfg in alt_cfgs])
    if best_alt is not None:
        best_alt_map = float(best_alt["map"])
        if best_alt_map > 0.274164 or best_alt_map >= 0.274164 * 0.99:
            alt_params = spatial_params_from_row(best_alt)
            alt_base = alt_params.get("base_experiment")
            print("Best alternative base is strong enough; trying equivalent on Paris")
            paris_alt_cfg = make_spatial_config(
                "paris",
                alt_base,
                alt_params["top_n"],
                alt_params["ratio"],
                alt_params["score"],
                alt_params["min_matches"],
                alt_params["min_inliers"],
            )
            rows.extend(run_configs([paris_alt_cfg], compute_ap_exe).to_dict("records"))
        else:
            print("No alternative Oxford base beat or came within 1% of the current best; skipping Paris alternative")
    else:
        print("No successful alternative Oxford spatial verification result")

    fisher_rows = []
    if run_fisher:
        print("Phase 4: Optional Fisher Vector smoke test")
        fisher_rows = maybe_run_fisher_smoke(compute_ap_exe)
        rows.extend(fisher_rows)
    else:
        print("Phase 4: Fisher Vector smoke test skipped (not requested by flag)")

    write_sorted_summary()
    after = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    newly_added = max(0, len(after) - len(before))
    print_final_report(newly_added)
    print_spatial_tuning_report(best_phase1, best_alt, fisher_rows, rows)
    return pd.DataFrame(rows)


def print_spatial_tuning_report(best_phase1, best_alt, fisher_rows, rows):
    if best_phase1 is not None:
        params = spatial_params_from_row(best_phase1)
        print(
            "Best spatial verification hyperparameters found: "
            f"{best_phase1['experiment']} mAP={float(best_phase1['map']):.6f}; "
            f"topN={params['top_n']}, ratio={params['ratio']}, score={params['score']}, "
            f"min_matches={params['min_matches']}, min_inliers={params['min_inliers']}"
        )
    if best_alt is not None:
        print(
            f"Best alternative base spatial result: {best_alt['experiment']} "
            f"mAP={float(best_alt['map']):.6f}; improved_current_best={float(best_alt['map']) > 0.274164}"
        )
    else:
        print("Best alternative base spatial result: none")
    if fisher_rows:
        for row in fisher_rows:
            print(
                f"Fisher smoke: {row.get('dataset')}/{row.get('experiment')} "
                f"status={row.get('status')} mAP={row.get('map')} error={row.get('error')}"
            )
    else:
        print("Fisher smoke: not run")
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    skipped_count = sum(1 for row in rows if str(row.get("status", "")).startswith("skipped"))
    failed = [row for row in rows if row.get("status") not in ("ok", "skipped/existing_success")]
    print(f"Phase rows: {ok_count} new successes, {skipped_count} skipped, {len(failed)} failed")
    if failed:
        for row in failed:
            print(f"Failed/skipped summary: {row.get('dataset')}/{row.get('experiment')}: {row.get('error')}")


def existing_ok_experiment(dataset, experiment):
    out = summary_path(False)
    if not out.exists():
        return False
    df = pd.read_csv(out)
    return (
        df["dataset"].fillna("").astype(str).eq(dataset)
        & df["experiment"].fillna("").astype(str).eq(experiment)
        & df["status"].fillna("").astype(str).eq("ok")
    ).any()


def run_dependency_for_base(dataset, base_exp, compute_ap_exe):
    rows = []
    if existing_ok_experiment(dataset, base_exp):
        return rows
    if "_verified_qe_" in base_exp and "_spatial_verify2_" not in base_exp:
        verified_cfg = verified_qe_config_from_name(dataset, base_exp)
        rows.extend(run_dependency_for_base(dataset, verified_cfg.extra_params["base_experiment"], compute_ap_exe))
        rows.append(run_experiment(verified_cfg, compute_ap_exe))
        return rows
    if "_spatial_verify" in base_exp:
        bovw_base = base_exp.split("_spatial_verify", 1)[0]
        rows.extend(run_dependency_for_base(dataset, bovw_base, compute_ap_exe))
        rows.append(run_experiment(make_spatial_config(dataset, bovw_base, 150, 0.65, "inlier_count"), compute_ap_exe))
        return rows
    try:
        rows.append(run_experiment(base_bovw_config_from_name(dataset, base_exp), compute_ap_exe))
    except Exception as exc:
        cfg = ExperimentConfig(
            dataset=dataset,
            experiment=base_exp,
            descriptor=descriptor_from_experiment_name(base_exp),
            representation="dependency",
            metric="",
            extra_params={"phase": 5},
        )
        rows.append(log_failure(cfg, exc))
    return rows


def equivalent_phase5_config(cfg, dataset):
    cfg = ExperimentConfig(**asdict(cfg))
    cfg.dataset = dataset
    return cfg


def phase5_all_known_configs(dataset="oxford"):
    cfgs = []
    cfgs.extend(list(phase5_bovw_configs(dataset)))
    cfgs.extend(list(phase5_spatial_configs(dataset)))
    cfgs.extend(list(phase5_verified_qe_configs(dataset)))
    cfgs.extend(list(phase5_scoring_refinement_configs(dataset)))
    return cfgs


def run_spatial_verify2_for_best_verified_qe(dataset, verified_rows, compute_ap_exe):
    ok = [row for row in verified_rows if row.get("status") == "ok"]
    if not ok:
        return []
    best = sorted(ok, key=lambda row: row.get("map", -1), reverse=True)[0]
    cfg = make_spatial_config(dataset, best["experiment"], 150, 0.65, "inlier_count")
    cfg.experiment = f"{best['experiment']}_spatial_verify2_top150_ratio0p65_inliers"
    cfg.extra_params["base_experiment"] = best["experiment"]
    cfg.extra_params["phase"] = 5
    return [run_experiment(cfg, compute_ap_exe)]


def make_spatial_verify2_config(dataset, verified_qe_exp):
    cfg = make_spatial_config(dataset, verified_qe_exp, 150, 0.65, "inlier_count")
    cfg.experiment = f"{verified_qe_exp}_spatial_verify2_top150_ratio0p65_inliers"
    cfg.extra_params["base_experiment"] = verified_qe_exp
    cfg.extra_params["phase"] = 5
    return cfg


def run_phase5():
    before = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    compute_ap_exe = compute_ap_executable()
    rows = []
    phase5_cfg_map = {}

    print("Phase 5A-C: RootSIFT and larger SIFT BoVW on Oxford")
    bovw_cfgs = list(phase5_bovw_configs("oxford"))
    for cfg in bovw_cfgs:
        phase5_cfg_map[cfg.experiment] = cfg
    rows.extend(run_configs(bovw_cfgs, compute_ap_exe).to_dict("records"))

    print("Phase 5B-C: Spatial verification for Phase 5 BoVW configs on Oxford")
    spatial_cfgs = list(phase5_spatial_configs("oxford"))
    for cfg in spatial_cfgs:
        phase5_cfg_map[cfg.experiment] = cfg
    rows.extend(run_configs(spatial_cfgs, compute_ap_exe).to_dict("records"))

    print("Phase 5D: Verified QE on Oxford")
    verified_cfgs = list(phase5_verified_qe_configs("oxford"))
    for cfg in verified_cfgs:
        phase5_cfg_map[cfg.experiment] = cfg
    verified_rows = run_configs(verified_cfgs, compute_ap_exe).to_dict("records")
    rows.extend(verified_rows)

    print("Phase 5D: Second spatial pass for best verified QE candidate")
    verify2_rows = run_spatial_verify2_for_best_verified_qe("oxford", verified_rows, compute_ap_exe)
    rows.extend(verify2_rows)
    for row in verify2_rows:
        if row.get("experiment"):
            phase5_cfg_map[row["experiment"]] = make_spatial_verify2_config(
                "oxford", row["experiment"].split("_spatial_verify2", 1)[0]
            )

    print("Phase 5E: Small spatial scoring refinements on Oxford")
    refine_cfgs = list(phase5_scoring_refinement_configs("oxford"))
    for cfg in refine_cfgs:
        phase5_cfg_map[cfg.experiment] = cfg
    rows.extend(run_configs(refine_cfgs, compute_ap_exe).to_dict("records"))

    write_sorted_summary()
    new_ok = [row for row in rows if row.get("status") == "ok" and row.get("dataset") == "oxford"]
    top3 = sorted(new_ok, key=lambda row: row.get("map", -1), reverse=True)[:3]
    print("Phase 5F: Paris validation for top 3 new Oxford experiments")
    for row in top3:
        base_cfg = phase5_cfg_map.get(row["experiment"])
        if base_cfg is None:
            continue
        paris_cfg = equivalent_phase5_config(base_cfg, "paris")
        base_exp = (paris_cfg.extra_params or {}).get("base_experiment")
        if base_exp:
            rows.extend(run_dependency_for_base("paris", base_exp, compute_ap_exe))
        rows.append(run_experiment(paris_cfg, compute_ap_exe))

    write_sorted_summary()
    after = pd.read_csv(summary_path(False)) if summary_path(False).exists() else pd.DataFrame(columns=SUMMARY_COLUMNS)
    newly_added = max(0, len(after) - len(before))
    print_final_report(newly_added)
    print_phase5_report(rows)
    return pd.DataFrame(rows)


def print_phase5_report(rows):
    ok = [row for row in rows if row.get("status") == "ok"]
    best_new = sorted(ok, key=lambda row: row.get("map", -1), reverse=True)[0] if ok else None
    if best_new:
        dataset = best_new["dataset"]
        prev = 0.299115 if dataset == "oxford" else 0.406194
        print(f"Best new Phase 5 experiment: {dataset}/{best_new['experiment']} mAP={best_new['map']:.6f}")
        print(f"Improvement over previous {dataset} best: {best_new['map'] - prev:.6f}")
    ox = [row for row in ok if row.get("dataset") == "oxford"]
    rootsift_best = max([row["map"] for row in ox if str(row.get("descriptor", "")).startswith("rootsift")], default=np.nan)
    sift_best = max([row["map"] for row in ox if str(row.get("descriptor", "")).startswith("sift")], default=np.nan)
    print(f"RootSIFT improved over SIFT in Phase 5: {pd.notna(rootsift_best) and pd.notna(sift_best) and rootsift_best > sift_best}")
    large_vocab_best = max([row["map"] for row in ox if row.get("vocab_size") in (2048, 4096)], default=np.nan)
    print(f"Larger vocabularies improved over k1024 Oxford best: {pd.notna(large_vocab_best) and large_vocab_best > 0.299115}")
    verified_best = max([row["map"] for row in ox if row.get("representation") == "verified_qe"], default=np.nan)
    print(f"Verified QE helped over previous Oxford best: {pd.notna(verified_best) and verified_best > 0.299115}")
    skipped = [row for row in rows if str(row.get("status", "")).startswith("skipped")]
    failed = [row for row in rows if row.get("status") not in ("ok", "skipped/existing_success")]
    print(f"Phase 5 failed/skipped experiments: {len(failed)} failed, {len(skipped)} skipped")
    for row in failed[:20]:
        print(f"Failed summary: {row.get('dataset')}/{row.get('experiment')}: {row.get('error')}")


def print_final_report(newly_added=None):
    write_sorted_summary()
    df = pd.read_csv(summary_path(True))
    ox = df[df["dataset"].eq("oxford") & df["status"].eq("ok")].copy()
    pa = df[df["dataset"].eq("paris") & df["status"].eq("ok")].copy()
    print("Top 15 Oxford experiments by mAP")
    if not ox.empty:
        print(ox[["experiment", "map", "num_queries"]].head(15).to_string(index=False))
    else:
        print("(none)")
    print("Top 10 Paris experiments by mAP")
    if not pa.empty:
        print(pa[["experiment", "map", "num_queries"]].head(10).to_string(index=False))
    else:
        print("(none)")
    ok = df[df["status"].eq("ok")].copy()
    if not ok.empty:
        best = ok.sort_values("map", ascending=False).iloc[0]
        print(f"Best experiment overall: {best['dataset']}/{best['experiment']} mAP={best['map']}")
    if newly_added is not None:
        print(f"Newly added experiment count: {newly_added}")
    failed = df[~df["status"].isin(["ok", "skipped/existing_success"])].copy()
    print(f"Failed experiment count: {len(failed)}")
    if not failed.empty:
        print(failed.groupby(["status", "error"]).size().reset_index(name="count").to_string(index=False))


def print_improvement_report():
    write_sorted_summary()
    df = pd.read_csv(summary_path(False))
    ok = df[df["status"].eq("ok")].copy()
    fusion = ok[ok["representation"].eq("late_fusion")]
    sv = ok[ok["representation"].eq("spatial_verification")]
    for dataset, previous_best in (("oxford", 0.218428), ("paris", 0.372444)):
        f = fusion[fusion["dataset"].eq(dataset)]
        s = sv[sv["dataset"].eq(dataset)]
        if not f.empty:
            best = f.sort_values("map", ascending=False).iloc[0]
            print(
                f"Best {dataset} fusion: {best['experiment']} mAP={best['map']:.6f}; "
                f"improved_previous_best={best['map'] > previous_best}"
            )
        else:
            print(f"Best {dataset} fusion: none")
        if not s.empty:
            best = s.sort_values("map", ascending=False).iloc[0]
            print(
                f"Best {dataset} spatial verification: {best['experiment']} mAP={best['map']:.6f}; "
                f"improved_previous_best={best['map'] > previous_best}"
            )
        else:
            print(f"Best {dataset} spatial verification: none")


def query_expansion_configs_old(rows):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("representation") in ("bovw", "bovw_tfidf", "tfidf_bovw")]
    ok = sorted(ok, key=lambda r: r.get("map", -1), reverse=True)[:3]
    for row in ok:
        for m in [5, 10]:
            extra = json.loads(row["extra_params"]) if isinstance(row["extra_params"], str) else dict(row["extra_params"] or {})
            extra["query_expansion_m"] = m
            yield ExperimentConfig(
                dataset=row["dataset"],
                experiment=f"{row['experiment']}_qe{m}",
                descriptor=row["descriptor"],
                representation=row["representation"],
                metric=row["metric"],
                normalization=row["normalization"],
                vocab_size=None if pd.isna(row["vocab_size"]) else int(row["vocab_size"]),
                extra_params=extra,
            )


def run_registry(datasets, groups, ks=None, normalizations=None, metrics=None):
    started = time.perf_counter()
    compute_ap_exe = compute_ap_executable()
    rows = []
    selected = []
    if "surf" in groups:
        selected.extend(surf_configs(datasets))
    if "bovw" in groups:
        selected.extend(bovw_configs(datasets, ks=ks or (128, 256, 512, 1024),
                                     normalizations=normalizations or ("l1", "l2", "sqrt_l2"),
                                     metrics=metrics or ("cosine", "euclidean", "chisquare", "hist_intersection")))
    if "tfidf" in groups:
        selected.extend(tfidf_configs(datasets, ks=ks or (256, 512, 1024),
                                      normalizations=normalizations or ("l2", "sqrt_l2"),
                                      metrics=metrics or ("cosine", "chisquare")))
    if "global" in groups:
        selected.extend(global_configs(datasets))
    if "fusion" in groups:
        selected.extend(fusion_configs(datasets))
    if "spatial" in groups:
        selected.extend(spatial_verification_configs(datasets))
    for cfg in selected:
        rows.append(run_experiment(cfg, compute_ap_exe))
    if "qe" in groups:
        for cfg in query_expansion_configs(rows):
            rows.append(run_experiment(cfg, compute_ap_exe))
    elapsed = time.perf_counter() - started
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Runtime summary: {len(rows)} experiments attempted, {ok_count} succeeded, {elapsed:.1f}s elapsed")
    return pd.DataFrame(rows)


def parse_csv_arg(value, cast=str):
    if value is None or value == "":
        return None
    return tuple(cast(v.strip()) for v in value.split(",") if v.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run advanced classical CV retrieval experiments.")
    parser.add_argument("--datasets", default="oxford,paris", help="Comma-separated datasets.")
    parser.add_argument("--groups", default="surf,bovw,tfidf,qe", help="Any of: surf,bovw,tfidf,qe,global,fusion,spatial")
    parser.add_argument("--ks", default="", help="Optional comma-separated vocabulary sizes.")
    parser.add_argument("--normalizations", default="", help="Optional comma-separated normalizations.")
    parser.add_argument("--metrics", default="", help="Optional comma-separated metrics.")
    parser.add_argument("--safe-extended", action="store_true", help="Run the staged safe extended sweep.")
    parser.add_argument("--global-fusion-spatial-phase", action="store_true", help="Run this phase's global, fusion, and spatial verification experiments.")
    parser.add_argument("--spatial-tuning-phase", action="store_true", help="Run spatial verification tuning and alternative-base phase.")
    parser.add_argument("--with-fisher-smoke", action="store_true", help="Also attempt the optional Oxford Fisher Vector smoke test.")
    parser.add_argument("--phase5", action="store_true", help="Run Phase 5 RootSIFT, larger vocabulary, verified QE, and spatial refinement experiments.")
    args = parser.parse_args(argv)

    if args.safe_extended:
        run_safe_extended_sweep()
        return
    if args.global_fusion_spatial_phase:
        run_global_fusion_spatial_phase()
        return
    if args.spatial_tuning_phase:
        run_spatial_tuning_phase(run_fisher=args.with_fisher_smoke)
        return
    if args.phase5:
        run_phase5()
        return

    datasets = parse_csv_arg(args.datasets)
    groups = parse_csv_arg(args.groups)
    ks = parse_csv_arg(args.ks, int)
    normalizations = parse_csv_arg(args.normalizations)
    metrics = parse_csv_arg(args.metrics)
    run_registry(datasets, groups, ks=ks, normalizations=normalizations, metrics=metrics)


if __name__ == "__main__":
    main()

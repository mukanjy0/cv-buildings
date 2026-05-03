import numpy as np
from pathlib import Path

from .paths import RESULTS_DIR


def l2_normalize(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def rank_cosine(query_vec, X, ids, exclude_id=None):
    Xn = l2_normalize(X)
    qn = l2_normalize(query_vec.reshape(1, -1))[0]

    scores = Xn @ qn
    order = np.argsort(-scores)

    ranking = []
    for idx in order:
        if ids[idx] != exclude_id:
            ranking.append(ids[idx])
    return ranking


def rank_euclidean(query_vec, X, ids, exclude_id=None):
    dists = np.linalg.norm(X - query_vec.reshape(1, -1), axis=1)
    order = np.argsort(dists)

    ranking = []
    for idx in order:
        if ids[idx] != exclude_id:
            ranking.append(ids[idx])
    return ranking


def write_ranking_file(ranking, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for image_id in ranking:
            f.write(image_id + "\n")
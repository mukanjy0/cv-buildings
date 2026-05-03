from tqdm import tqdm

from .paths import DATASETS, RESULTS_DIR
from .evaluation import list_query_files, parse_query_file
from .retrieval import rank_cosine, rank_euclidean, write_ranking_file
from .features import create_sift, sift_meanstd_from_bbox


def generate_rankings(dataset: str, ids, X, metric: str, use_query_bbox=True):
    qfiles = list_query_files(DATASETS[dataset]["gt_dir"])
    id_to_idx = {image_id: i for i, image_id in enumerate(ids)}

    suffix = "bbox" if use_query_bbox else "full"
    exp_name = f"sift_meanstd_{suffix}_{metric}"

    out_dir = RESULTS_DIR / "rankings" / dataset / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    detector = create_sift()

    for qf in tqdm(qfiles, desc=f"Ranking {dataset}/{metric}/{suffix}"):
        if use_query_bbox:
            query_name, query_image_id, bbox = parse_query_file(qf, return_bbox=True)
        else:
            query_name, query_image_id = parse_query_file(qf)
            bbox = None

        if query_image_id not in id_to_idx:
            missing.append((query_name, query_image_id))
            continue

        if use_query_bbox:
            qvec = sift_meanstd_from_bbox(
                dataset=dataset,
                image_id=query_image_id,
                bbox=bbox,
                detector=detector,
            )
        else:
            qidx = id_to_idx[query_image_id]
            qvec = X[qidx]

        if metric == "cosine":
            ranking = rank_cosine(qvec, X, ids, exclude_id=query_image_id)
        elif metric == "euclidean":
            ranking = rank_euclidean(qvec, X, ids, exclude_id=query_image_id)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        write_ranking_file(ranking, out_dir / f"{query_name}.txt")

    if missing:
        print("Missing query images:", len(missing))
        print(missing[:10])

    return exp_name, out_dir
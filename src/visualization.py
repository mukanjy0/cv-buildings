from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

from .paths import DATASETS, IMG_EXTS, RESULTS_DIR
from .evaluation import list_query_files, parse_query_file


def get_image_path(dataset: str, image_id: str):
    img_dir = DATASETS[dataset]["img_dir"]

    for ext in IMG_EXTS:
        p = img_dir / f"{image_id}{ext}"
        if p.exists():
            return p

    return None


def read_ranking_file(path: Path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_rgb(path: Path):
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def draw_bbox_rgb(img, bbox, color=(255, 0, 0), thickness=4):
    x1, y1, x2, y2 = map(int, bbox)
    out = img.copy()
    cv.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def save_top5_visualization(dataset: str, exp_name: str, query_name: str):
    qfile = DATASETS[dataset]["gt_dir"] / f"{query_name}_query.txt"
    query_name, query_image_id, bbox = parse_query_file(qfile, return_bbox=True)

    ranking_path = RESULTS_DIR / "rankings" / dataset / exp_name / f"{query_name}.txt"
    ranking = read_ranking_file(ranking_path)

    query_path = get_image_path(dataset, query_image_id)
    if query_path is None:
        raise FileNotFoundError(f"Missing query image: {query_image_id}")

    query_img = draw_bbox_rgb(load_rgb(query_path), bbox)

    top5_ids = ranking[:5]
    top5_imgs = []

    for image_id in top5_ids:
        p = get_image_path(dataset, image_id)
        if p is None:
            raise FileNotFoundError(f"Missing retrieved image: {image_id}")
        top5_imgs.append((image_id, load_rgb(p)))

    fig, axes = plt.subplots(1, 6, figsize=(22, 5))

    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\n{query_image_id}")
    axes[0].axis("off")

    for ax, (image_id, img) in zip(axes[1:], top5_imgs):
        ax.imshow(img)
        ax.set_title(image_id)
        ax.axis("off")

    fig.suptitle(f"{dataset} | {exp_name} | {query_name}", fontsize=14)
    plt.tight_layout()

    out_dir = RESULTS_DIR / "qualitative" / dataset / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{query_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def save_all_top5_visualizations(dataset: str, exp_name: str):
    out_paths = []

    for qfile in list_query_files(DATASETS[dataset]["gt_dir"]):
        query_name, _ = parse_query_file(qfile)
        out_paths.append(save_top5_visualization(dataset, exp_name, query_name))

    return out_paths


def select_success_failure_queries(ap_csv_path=None):
    if ap_csv_path is None:
        ap_csv_path = RESULTS_DIR / "ap_per_query.csv"

    ap_df = pd.read_csv(ap_csv_path)

    rows = []

    for (dataset, exp_name), group in ap_df.groupby(["dataset", "experiment"]):
        group = group.sort_values("ap").reset_index(drop=True)

        worst = group.iloc[0]
        best = group.iloc[-1]
        median = group.iloc[len(group) // 2]

        rows.extend([
            {
                "dataset": dataset,
                "experiment": exp_name,
                "case": "worst",
                "query": worst["query"],
                "ap": worst["ap"],
            },
            {
                "dataset": dataset,
                "experiment": exp_name,
                "case": "median",
                "query": median["query"],
                "ap": median["ap"],
            },
            {
                "dataset": dataset,
                "experiment": exp_name,
                "case": "best",
                "query": best["query"],
                "ap": best["ap"],
            },
        ])

    return pd.DataFrame(rows)


def save_success_failure_visualizations(ap_csv_path=None):
    cases_df = select_success_failure_queries(ap_csv_path)

    saved = []

    for _, row in cases_df.iterrows():
        out_path = save_top5_visualization(
            dataset=row["dataset"],
            exp_name=row["experiment"],
            query_name=row["query"],
        )
        saved.append({
            **row.to_dict(),
            "visualization": str(out_path),
        })

    saved_df = pd.DataFrame(saved)
    out_csv = RESULTS_DIR / "qualitative_cases.csv"
    saved_df.to_csv(out_csv, index=False)

    return saved_df
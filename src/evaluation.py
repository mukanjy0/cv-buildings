import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from .paths import ROOT, DATASETS, RESULTS_DIR


def list_query_files(gt_dir: Path):
    return sorted(gt_dir.glob("*_query.txt"))


def parse_query_file(query_file: Path):
    query_name = query_file.stem.replace("_query", "")

    with open(query_file, "r") as f:
        parts = f.read().strip().split()

    raw_img_name = parts[0]

    if raw_img_name.startswith("oxc1_"):
        raw_img_name = raw_img_name[len("oxc1_"):]

    image_id = Path(raw_img_name).stem
    return query_name, image_id


def compile_compute_ap():
    candidates = [
        ROOT / "compute_ap.cpp",
        ROOT / "oxford" / "compute_ap.cpp",
    ]

    cpp_path = next((p for p in candidates if p.exists()), None)

    if cpp_path is None:
        raise FileNotFoundError("Could not find compute_ap.cpp")

    exe_path = ROOT / "compute_ap"

    cmd = ["g++", "-O2", "-std=c++17", str(cpp_path), "-o", str(exe_path)]
    subprocess.run(cmd, check=True)

    return exe_path


def compute_ap_for_query(compute_ap_exe: Path, dataset: str, query_name: str, ranking_file: Path):
    gt_prefix = DATASETS[dataset]["gt_dir"] / query_name

    proc = subprocess.run(
        [str(compute_ap_exe), str(gt_prefix), str(ranking_file)],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(proc.stdout.strip())


def evaluate_experiment(dataset: str, exp_name: str, compute_ap_exe: Path):
    rankings_dir = RESULTS_DIR / "rankings" / dataset / exp_name
    qfiles = list_query_files(DATASETS[dataset]["gt_dir"])

    rows = []

    for qf in tqdm(qfiles, desc=f"Evaluating {dataset}/{exp_name}"):
        query_name, _ = parse_query_file(qf)
        ranking_file = rankings_dir / f"{query_name}.txt"

        if not ranking_file.exists():
            continue

        ap = compute_ap_for_query(compute_ap_exe, dataset, query_name, ranking_file)

        rows.append({
            "dataset": dataset,
            "experiment": exp_name,
            "query": query_name,
            "ap": ap,
        })

    return pd.DataFrame(rows)
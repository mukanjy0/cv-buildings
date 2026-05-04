from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.evaluation import list_query_files, parse_query_file  # noqa: E402
from src.features import get_image_path  # noqa: E402
from src.paths import DATASETS, RESULTS_DIR  # noqa: E402


try:
    from src.advanced_experiments import extract_local_descriptor
except Exception:
    extract_local_descriptor = None


OUT_ROOT = RESULTS_DIR / "final_artifacts"
QUANT_DIR = OUT_ROOT / "quantitative"
QUAL_DIR = OUT_ROOT / "qualitative"
SV_DIR = OUT_ROOT / "spatial_verification"
TABLE_DIR = OUT_ROOT / "tables"
SELECTED_DIR = OUT_ROOT / "selected_cases"
LOG_DIR = OUT_ROOT / "logs"

RANDOM_SEED = 13

ROUND_EXPERIMENTS = {
    "oxford": {
        "R1": "sift_bovw_tfidf_k256_l2_chisquare",
        "R2": "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25",
        "R3": "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers",
        "R4": "sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers",
        "R5": "sift_bovw_k2048_l2_cosine_spatial_verify_top150_ratio0p65_inliers_verified_qe_top3_alpha0p5_spatial_verify2_top150_ratio0p65_inliers",
        "R6": "multiscale_sift_bovw_k4096_l2_cosine_spatial_verify_top150_ratio0p65_inliers_verified_qe_top3_alpha0p5_spatial_verify2_top150_ratio0p65_inliers",
    },
    "paris": {
        "R1": "sift_bovw_tfidf_k256_l2_chisquare",
        "R2": "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25",
        "R3": "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers",
        "R4": "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers",
        "R5": "rootsift_bovw_k4096_l2_cosine_spatial_verify_top150_ratio0p65_inliers",
        "R6": "multiscale_sift_bovw_k4096_l2_cosine_spatial_verify_top150_ratio0p65_inliers_verified_qe_top3_alpha0p5_spatial_verify2_top150_ratio0p65_inliers",
    },
}

ROUND_NAMES = {
    "R1": "First SIFT BoVW baseline",
    "R2": "Expanded BoVW + QE",
    "R3": "Fusion + first SV",
    "R4": "Tuned SV",
    "R5": "Large vocab / RootSIFT / verified QE",
    "R6": "Multi-scale SIFT",
}

FINAL_EXP = {
    "oxford": ROUND_EXPERIMENTS["oxford"]["R6"],
    "paris": ROUND_EXPERIMENTS["paris"]["R6"],
}

EARLY_EXP = {
    "oxford": ROUND_EXPERIMENTS["oxford"]["R2"],
    "paris": ROUND_EXPERIMENTS["paris"]["R2"],
}

BORDER_COLORS = {
    "good": "#2ca02c",
    "ok": "#2ca02c",
    "junk": "#888888",
    "unknown": "#d62728",
    "query": "#1f77b4",
}


@dataclass
class Issue:
    step: str
    dataset: str
    experiment: str
    query_name: str
    message: str


class ArtifactBuilder:
    def __init__(self):
        self.issues: list[Issue] = []
        self.index_rows: list[dict] = []
        self.counts = {
            "quantitative_plots": 0,
            "top5_grids": 0,
            "progression_grids": 0,
            "before_after_grids": 0,
            "failure_cases": 0,
            "spatial_verification_figures": 0,
            "csv_tables": 0,
        }
        self.ap_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self.gt_cache: dict[tuple[str, str], dict[str, set[str]]] = {}

    def log_issue(self, step, dataset="", experiment="", query_name="", message=""):
        self.issues.append(Issue(step, dataset, experiment, query_name, message))
        print(f"[warn] {step}: {dataset}/{experiment}/{query_name}: {message}")

    def add_index(
        self,
        artifact_type,
        path,
        dataset="",
        round_label="",
        query_name="",
        presentation=False,
        report=True,
        notes="",
    ):
        self.index_rows.append(
            {
                "artifact_type": artifact_type,
                "dataset": dataset,
                "round": round_label,
                "query_name": query_name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "recommended_for_presentation": bool(presentation),
                "recommended_for_report": bool(report),
                "notes": notes,
            }
        )

    def prepare_dirs(self):
        if OUT_ROOT.exists():
            resolved = OUT_ROOT.resolve()
            if resolved != (RESULTS_DIR / "final_artifacts").resolve():
                raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
            shutil.rmtree(OUT_ROOT)
        for path in [QUANT_DIR, QUAL_DIR, SV_DIR, TABLE_DIR, SELECTED_DIR, LOG_DIR]:
            path.mkdir(parents=True, exist_ok=True)

    def ranking_dir(self, dataset, experiment):
        return RESULTS_DIR / "rankings" / dataset / experiment

    def ranking_path(self, dataset, experiment, query_name):
        return self.ranking_dir(dataset, experiment) / f"{query_name}.txt"

    def read_ranking(self, dataset, experiment, query_name):
        path = self.ranking_path(dataset, experiment, query_name)
        if not path.exists():
            raise FileNotFoundError(f"Missing ranking file: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def compute_ap_exe(self):
        exe = ROOT / "compute_ap.exe"
        if exe.exists():
            return exe
        exe = ROOT / "compute_ap"
        if exe.exists():
            return exe
        raise FileNotFoundError("compute_ap executable not found")

    def compute_ap_table(self, dataset, experiment):
        key = (dataset, experiment)
        if key in self.ap_cache:
            return self.ap_cache[key]
        ranking_dir = self.ranking_dir(dataset, experiment)
        if not ranking_dir.exists():
            self.log_issue("compute_ap", dataset, experiment, "", f"Missing rankings directory: {ranking_dir}")
            df = pd.DataFrame(columns=["dataset", "experiment", "query", "ap"])
            self.ap_cache[key] = df
            return df
        exe = self.compute_ap_exe()
        rows = []
        for qf in list_query_files(DATASETS[dataset]["gt_dir"]):
            query_name, _ = parse_query_file(qf)
            ranking = ranking_dir / f"{query_name}.txt"
            if not ranking.exists():
                self.log_issue("compute_ap", dataset, experiment, query_name, "Missing ranking file")
                continue
            try:
                proc = subprocess.run(
                    [str(exe), str(DATASETS[dataset]["gt_dir"] / query_name), str(ranking)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "experiment": experiment,
                        "query": query_name,
                        "ap": float(proc.stdout.strip()),
                    }
                )
            except Exception as exc:
                self.log_issue("compute_ap", dataset, experiment, query_name, str(exc))
        df = pd.DataFrame(rows)
        self.ap_cache[key] = df
        return df

    def get_gt_sets(self, dataset, query_name):
        key = (dataset, query_name)
        if key in self.gt_cache:
            return self.gt_cache[key]
        gt_dir = DATASETS[dataset]["gt_dir"]
        sets = {}
        for label in ["good", "ok", "junk"]:
            path = gt_dir / f"{query_name}_{label}.txt"
            if path.exists():
                sets[label] = {line.strip() for line in path.read_text().splitlines() if line.strip()}
            else:
                sets[label] = set()
        self.gt_cache[key] = sets
        return sets

    def relevance_label(self, dataset, query_name, image_id):
        sets = self.get_gt_sets(dataset, query_name)
        for label in ["good", "ok", "junk"]:
            if image_id in sets[label]:
                return label
        return "unknown"

    def imread_rgb(self, dataset, image_id):
        path = get_image_path(dataset, image_id)
        if path is None:
            raise FileNotFoundError(f"Missing image {dataset}/{image_id}")
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def draw_bbox(self, img, bbox):
        out = img.copy()
        h, w = out.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(max(0, min(w - 1, round(x1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y1 = int(max(0, min(h - 1, round(y1))))
        y2 = int(max(0, min(h - 1, round(y2))))
        cv.rectangle(out, (x1, y1), (x2, y2), (255, 48, 48), max(3, min(h, w) // 150))
        return out

    def set_border(self, ax, label):
        color = BORDER_COLORS.get(label, "#d62728")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(5)

    def short_exp(self, exp):
        exp = exp.replace("sift_bovw_", "sift ")
        exp = exp.replace("multiscale_sift_bovw_", "ms-sift ")
        exp = exp.replace("_spatial_verify_top150_ratio0p65_inliers", " + SV")
        exp = exp.replace("_verified_qe_top3_alpha0p5", " + vQE")
        exp = exp.replace("_spatial_verify2_top150_ratio0p65_inliers", " + SV2")
        exp = exp.replace("_", " ")
        return exp[:90]

    def save_top5_grid(self, dataset, experiment, query_name, out_path, round_label="", case_type="", ap=None):
        try:
            qfile = DATASETS[dataset]["gt_dir"] / f"{query_name}_query.txt"
            _, query_image_id, bbox = parse_query_file(qfile, return_bbox=True)
            ranking = self.read_ranking(dataset, experiment, query_name)
            top5 = ranking[:5]
            query_img = self.draw_bbox(self.imread_rgb(dataset, query_image_id), bbox)
            fig, axes = plt.subplots(1, 6, figsize=(22, 4.8))
            axes[0].imshow(query_img)
            axes[0].set_title(f"Query\n{query_image_id}", fontsize=11)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            self.set_border(axes[0], "query")
            rels = []
            for rank, (ax, image_id) in enumerate(zip(axes[1:], top5), start=1):
                label = self.relevance_label(dataset, query_name, image_id)
                rels.append(label)
                ax.imshow(self.imread_rgb(dataset, image_id))
                ax.set_title(f"#{rank} {label}\n{image_id}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                self.set_border(ax, label)
            title_ap = "" if ap is None or pd.isna(ap) else f" | AP={ap:.3f}"
            fig.suptitle(
                f"{dataset.upper()} {round_label} {case_type} | {query_name}{title_ap}\n{self.short_exp(experiment)}",
                fontsize=16,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.86])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            self.counts["top5_grids"] += 1
            return top5, rels
        except Exception as exc:
            self.log_issue("top5_grid", dataset, experiment, query_name, str(exc))
            return [], []

    def save_contact_sheet(self, image_paths, out_path, title, max_cols=3, thumb_w=520):
        image_paths = [Path(p) for p in image_paths if p and Path(p).exists()]
        if not image_paths:
            self.log_issue("contact_sheet", "", "", "", f"No source images for {out_path}")
            return
        thumbs = []
        for path in image_paths:
            img = cv.imread(str(path), cv.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = thumb_w / max(w, 1)
            thumb = cv.resize(img, (thumb_w, max(1, int(h * scale))), interpolation=cv.INTER_AREA)
            thumbs.append((path, thumb))
        if not thumbs:
            return
        cols = min(max_cols, len(thumbs))
        rows = math.ceil(len(thumbs) / cols)
        title_h = 70
        pad = 16
        cell_h = max(t.shape[0] for _, t in thumbs) + 44
        canvas = np.full((title_h + rows * cell_h + pad, cols * (thumb_w + pad) + pad, 3), 255, np.uint8)
        cv.putText(canvas, title, (pad, 45), cv.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv.LINE_AA)
        for i, (path, thumb) in enumerate(thumbs):
            r, c = divmod(i, cols)
            x = pad + c * (thumb_w + pad)
            y = title_h + r * cell_h
            canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
            cv.putText(
                canvas,
                path.stem[:42],
                (x, y + thumb.shape[0] + 28),
                cv.FONT_HERSHEY_SIMPLEX,
                0.58,
                (40, 40, 40),
                1,
                cv.LINE_AA,
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_path), canvas)

    def make_best_per_round(self):
        summary = pd.read_csv(RESULTS_DIR / "advanced_summary.csv")
        rows = []
        for dataset, rounds in ROUND_EXPERIMENTS.items():
            prev = None
            for round_label, exp in rounds.items():
                match = summary[
                    summary["dataset"].eq(dataset)
                    & summary["experiment"].eq(exp)
                    & summary["status"].eq("ok")
                ]
                map_value = float(match.iloc[-1]["map"]) if not match.empty else np.nan
                abs_imp = map_value - prev if prev is not None and pd.notna(map_value) else np.nan
                rel_imp = abs_imp / prev if prev not in (None, 0) and pd.notna(abs_imp) else np.nan
                rows.append(
                    {
                        "round": round_label,
                        "round_description": ROUND_NAMES[round_label],
                        "dataset": dataset,
                        "best_experiment": exp,
                        "mAP": map_value,
                        "absolute_improvement_vs_previous_round": abs_imp,
                        "relative_improvement_vs_previous_round": rel_imp,
                    }
                )
                if pd.notna(map_value):
                    prev = map_value
        df = pd.DataFrame(rows)
        out = TABLE_DIR / "best_per_round.csv"
        df.to_csv(out, index=False)
        self.counts["csv_tables"] += 1
        self.add_index("table", out, notes="Best representative experiment per round.")
        return df

    def plot_map_progress(self, df):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = {"oxford": "#1f77b4", "paris": "#ff7f0e"}
        for dataset, group in df.groupby("dataset"):
            group = group.sort_values("round")
            x = np.arange(len(group))
            y = group["mAP"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=3, markersize=10, label=dataset.title(), color=colors.get(dataset))
            for xi, yi in zip(x, y):
                if pd.notna(yi):
                    ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=12)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels([f"R{i}" for i in range(1, 7)], fontsize=14)
        ax.set_ylabel("mAP", fontsize=15)
        ax.set_xlabel("Experiment Round", fontsize=15)
        ax.set_title("Best mAP Progress by Round", fontsize=19, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=14, loc="lower right")
        ax.set_ylim(bottom=0.15, top=max(df["mAP"].dropna()) + 0.06)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            out = QUANT_DIR / f"map_progress_by_round.{ext}"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            self.add_index("plot", out, presentation=True, notes="Best mAP by round.")
        plt.close(fig)
        self.counts["quantitative_plots"] += 2

    def parameter_study(self):
        summary = pd.read_csv(RESULTS_DIR / "advanced_summary.csv")
        ok = summary[summary["status"].eq("ok")].copy()
        rows = []
        for _, row in ok.iterrows():
            exp = str(row["experiment"])
            dataset = row["dataset"]
            if dataset != "oxford":
                continue
            if re.fullmatch(r"sift_bovw_k\d+_l2_cosine", exp):
                rows.append({"study": "vocab_size_l2_cosine", "setting": int(row["vocab_size"]), "dataset": dataset, "experiment": exp, "mAP": row["map"]})
            m = re.search(r"spatial_verify_top(\d+)_ratio([0-9p]+)_inliers$", exp)
            if m and "k1024_l2_cosine_qe_top3_alpha0p25" in exp:
                rows.append({"study": "spatial_topN_ratio", "setting": f"top{m.group(1)} r{m.group(2).replace('p','.')}", "dataset": dataset, "experiment": exp, "mAP": row["map"]})
            m = re.search(r"_qe_top(\d+)_alpha([0-9p]+)$", exp)
            if m and exp.startswith("sift_bovw_k1024_l2_cosine_qe"):
                rows.append({"study": "query_expansion", "setting": f"top{m.group(1)} a{m.group(2).replace('p','.')}", "dataset": dataset, "experiment": exp, "mAP": row["map"]})
        df = pd.DataFrame(rows).sort_values(["study", "setting"])
        out = TABLE_DIR / "parameter_study.csv"
        df.to_csv(out, index=False)
        self.counts["csv_tables"] += 1
        self.add_index("table", out, notes="Parameter studies from logged Oxford experiments.")

        studies = [s for s in ["vocab_size_l2_cosine", "spatial_topN_ratio", "query_expansion"] if s in set(df["study"])]
        fig, axes = plt.subplots(len(studies), 1, figsize=(13, 4.2 * len(studies)))
        if len(studies) == 1:
            axes = [axes]
        for ax, study in zip(axes, studies):
            sub = df[df["study"].eq(study)].copy()
            if study == "vocab_size_l2_cosine":
                sub = sub.sort_values("setting")
                x = [str(int(v)) for v in sub["setting"]]
            else:
                sub = sub.sort_values("mAP", ascending=False).head(12).sort_values("mAP")
                x = sub["setting"].astype(str).tolist()
            ax.bar(x, sub["mAP"], color="#4c78a8")
            ax.set_title(study.replace("_", " ").title(), fontsize=15, fontweight="bold")
            ax.set_ylabel("mAP")
            ax.tick_params(axis="x", rotation=35, labelsize=10)
            for i, val in enumerate(sub["mAP"]):
                ax.text(i, val + 0.004, f"{val:.3f}", ha="center", fontsize=9)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            out_plot = QUANT_DIR / f"parameter_study.{ext}"
            fig.savefig(out_plot, dpi=220, bbox_inches="tight")
            self.add_index("plot", out_plot, presentation=ext == "png", notes="Parameter study summary.")
        plt.close(fig)
        self.counts["quantitative_plots"] += 2

    def method_ablation_summary(self):
        rows = [
            ["SIFT BoVW baseline", "", "sift_bovw_tfidf_k256_l2_chisquare", np.nan, 0.195169, np.nan, np.nan, "established baseline", "BoVW over local SIFT was a strong classical starting point."],
            ["TF-IDF", "sift_bovw_k256_l2_chisquare", "sift_bovw_tfidf_k256_l2_chisquare", 0.194956, 0.195169, np.nan, np.nan, "tiny gain", "TF-IDF helped only marginally in the best early Oxford setting."],
            ["Larger vocabulary", "sift_bovw_k512_l2_cosine", "sift_bovw_k1024_l2_cosine", 0.187848, 0.202552, np.nan, np.nan, "positive", "Higher k helped once cosine/L2 became the main setting."],
            ["Query expansion", "sift_bovw_k1024_l2_cosine", "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25", 0.202552, 0.218428, np.nan, 0.372444, "positive but drift-prone", "Small topM helped; larger topM tended to drift."],
            ["HOG/HSV fusion", "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25", "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p95_0p05", 0.218428, 0.220022, 0.372444, 0.374227, "weak gain", "Global descriptors added little compared with local landmark evidence."],
            ["Spatial verification", "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25", "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers", 0.218428, 0.274164, 0.372444, 0.392230, "large gain", "Geometry suppressed visually similar false positives."],
            ["Tuned spatial verification", "sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers", "sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers", 0.274164, 0.299115, 0.392230, 0.406194, "positive", "Top150 and ratio 0.65 were stronger than the first SV setting."],
            ["RootSIFT / large vocabulary", "sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers", "rootsift_bovw_k4096_l2_cosine_spatial_verify_top150_ratio0p65_inliers", 0.299115, 0.329886, 0.406194, 0.431121, "positive", "Large vocabularies helped; RootSIFT was especially useful on Paris."],
            ["Verified QE + second SV", "sift_bovw_k2048_l2_cosine_spatial_verify_top150_ratio0p65_inliers", "sift_bovw_k2048_l2_cosine_spatial_verify_top150_ratio0p65_inliers_verified_qe_top3_alpha0p5_spatial_verify2_top150_ratio0p65_inliers", 0.313679, 0.354805, 0.409938, 0.427590, "large gain", "QE alone could drift, but a second geometry pass recovered useful candidates."],
            ["Multi-scale SIFT", "sift_bovw_k2048_l2_cosine_spatial_verify_top150_ratio0p65_inliers_verified_qe_top3_alpha0p5_spatial_verify2_top150_ratio0p65_inliers", FINAL_EXP["oxford"], 0.354805, 0.363375, 0.431121, 0.445350, "best final", "Explicit scale pooling added useful local evidence."],
        ]
        columns = [
            "method_or_change",
            "representative_experiment_before",
            "representative_experiment_after",
            "oxford_before",
            "oxford_after",
            "paris_before",
            "paris_after",
            "effect",
            "interpretation",
        ]
        df = pd.DataFrame(rows, columns=columns)
        out = TABLE_DIR / "method_ablation_summary.csv"
        df.to_csv(out, index=False)
        self.counts["csv_tables"] += 1
        self.add_index("table", out, presentation=True, notes="Compact what worked / what did not summary.")

    def pipeline_diagram(self):
        steps = [
            "Query image\n+ bbox",
            "Multi-scale SIFT\n/ SIFT descriptors",
            "BoVW histogram\nk=4096",
            "L2 norm\n+ cosine",
            "Initial ranking",
            "Spatial verification\nLowe ratio + RANSAC",
            "Verified query\nexpansion",
            "Second spatial\nverification",
            "Final ranked list",
        ]
        fig, ax = plt.subplots(figsize=(17, 4.8))
        ax.axis("off")
        xs = np.linspace(0.05, 0.95, len(steps))
        for i, (x, step) in enumerate(zip(xs, steps)):
            ax.text(
                x,
                0.55,
                step,
                ha="center",
                va="center",
                fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#4c78a8", linewidth=2),
            )
            if i < len(steps) - 1:
                ax.annotate("", xy=(xs[i + 1] - 0.052, 0.55), xytext=(x + 0.052, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
        ax.set_title("Final Classical Image Retrieval Pipeline", fontsize=18, fontweight="bold", pad=20)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            out = QUANT_DIR / f"final_pipeline_diagram.{ext}"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            self.add_index("diagram", out, presentation=True, notes="Final method pipeline.")
        plt.close(fig)
        self.counts["quantitative_plots"] += 2

    def select_cases(self, dataset, experiment, n_each=2):
        ap = self.compute_ap_table(dataset, experiment)
        if ap.empty:
            return []
        ap = ap.sort_values(["ap", "query"]).reset_index(drop=True)
        low = ap.head(n_each).assign(case_type="failure")
        high = ap.tail(n_each).assign(case_type="success")
        median_ap = ap["ap"].median()
        mid = ap.assign(dist=(ap["ap"] - median_ap).abs()).sort_values(["dist", "query"]).head(n_each).assign(case_type="typical")
        cases = pd.concat([high, mid, low], ignore_index=True)
        return cases.drop_duplicates("query").to_dict("records")

    def qualitative_top5_pool(self):
        rows = []
        success_paths, failure_paths = [], []
        for round_label, exp in ROUND_EXPERIMENTS["oxford"].items():
            cases = self.select_cases("oxford", exp, n_each=2)
            for case in cases:
                out = QUAL_DIR / "oxford" / round_label / case["case_type"] / f"{case['query']}.png"
                top5, rels = self.save_top5_grid("oxford", exp, case["query"], out, round_label, case["case_type"], case["ap"])
                if top5:
                    rows.append(
                        {
                            "dataset": "oxford",
                            "round": round_label,
                            "experiment": exp,
                            "query_name": case["query"],
                            "case_type": case["case_type"],
                            "AP": case["ap"],
                            "top5_image_ids": "|".join(top5),
                            "top5_relevance_labels": "|".join(rels),
                            "artifact_path": str(out.relative_to(ROOT)).replace("\\", "/"),
                            "notes_auto": "selected by per-query AP quantile",
                        }
                    )
                    self.add_index("top5_grid", out, "oxford", round_label, case["query"], presentation=case["case_type"] == "success", notes=case["case_type"])
                    if case["case_type"] == "success":
                        success_paths.append(out)
                    if case["case_type"] == "failure":
                        failure_paths.append(out)
        df = pd.DataFrame(rows)
        out_csv = TABLE_DIR / "qualitative_cases.csv"
        df.to_csv(out_csv, index=False)
        self.counts["csv_tables"] += 1
        self.add_index("table", out_csv, notes="Top-5 qualitative case metadata.")
        return df, success_paths, failure_paths

    def progression_grids(self):
        final_ap = self.compute_ap_table("oxford", FINAL_EXP["oxford"]).rename(columns={"ap": "final_ap"})
        early_ap = self.compute_ap_table("oxford", ROUND_EXPERIMENTS["oxford"]["R1"]).rename(columns={"ap": "early_ap"})
        if early_ap.empty:
            early_ap = self.compute_ap_table("oxford", ROUND_EXPERIMENTS["oxford"]["R2"]).rename(columns={"ap": "early_ap"})
        merged = final_ap.merge(early_ap[["query", "early_ap"]], on="query", how="inner")
        if merged.empty:
            return []
        merged["delta"] = merged["final_ap"] - merged["early_ap"]
        improved = merged.sort_values(["delta", "final_ap"], ascending=False).head(2)
        failed = merged.sort_values(["final_ap", "query"]).head(2)
        mixed = merged.assign(abs_delta=merged["delta"].abs()).sort_values(["abs_delta", "query"]).head(2)
        selected = pd.concat([improved.assign(group="improved"), mixed.assign(group="mixed"), failed.assign(group="still_fails")]).drop_duplicates("query").head(6)
        paths = []
        for _, row in selected.iterrows():
            query = row["query"]
            out = SELECTED_DIR / f"progression_{query}.png"
            pdf = SELECTED_DIR / f"progression_{query}.pdf"
            if self.save_progression_grid(query, out, pdf):
                paths.append(out)
                self.add_index("progression_grid", out, "oxford", "R1-R6", query, presentation=len(paths) <= 2, notes=row["group"])
                self.add_index("progression_grid_pdf", pdf, "oxford", "R1-R6", query, report=True, notes=row["group"])
        self.save_contact_sheet(paths, SELECTED_DIR / "progression_contact_sheet.png", "Progression Grids", max_cols=2, thumb_w=650)
        self.add_index("contact_sheet", SELECTED_DIR / "progression_contact_sheet.png", "oxford", "R1-R6", presentation=True, notes="Progression contact sheet.")
        return paths

    def save_progression_grid(self, query_name, out_png, out_pdf):
        try:
            _, query_image_id, bbox = parse_query_file(DATASETS["oxford"]["gt_dir"] / f"{query_name}_query.txt", return_bbox=True)
            fig, axes = plt.subplots(6, 6, figsize=(20, 21))
            for r, (round_label, exp) in enumerate(ROUND_EXPERIMENTS["oxford"].items()):
                ranking = self.read_ranking("oxford", exp, query_name)[:5]
                qimg = self.draw_bbox(self.imread_rgb("oxford", query_image_id), bbox)
                cells = [("query", query_image_id, qimg)] + [(self.relevance_label("oxford", query_name, image_id), image_id, self.imread_rgb("oxford", image_id)) for image_id in ranking]
                ap_df = self.compute_ap_table("oxford", exp)
                ap_val = ap_df.loc[ap_df["query"].eq(query_name), "ap"]
                ap_text = f" AP={float(ap_val.iloc[0]):.3f}" if not ap_val.empty else ""
                for c, (label, image_id, img) in enumerate(cells):
                    ax = axes[r, c]
                    ax.imshow(img)
                    title = f"{round_label}{ap_text}\n{self.short_exp(exp)}" if c == 0 else f"#{c} {label}\n{image_id}"
                    ax.set_title(title, fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self.set_border(ax, label)
            fig.suptitle(f"Oxford Retrieval Progression: {query_name}", fontsize=20, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.965])
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            fig.savefig(out_pdf, bbox_inches="tight")
            plt.close(fig)
            self.counts["progression_grids"] += 1
            return True
        except Exception as exc:
            self.log_issue("progression_grid", "oxford", "R1-R6", query_name, str(exc))
            return False

    def before_after_grids(self):
        rows, paths = [], []
        for dataset in ["oxford", "paris"]:
            early = EARLY_EXP[dataset]
            final = FINAL_EXP[dataset]
            final_ap = self.compute_ap_table(dataset, final)
            early_ap = self.compute_ap_table(dataset, early)
            if final_ap.empty or early_ap.empty:
                continue
            merged = final_ap.rename(columns={"ap": "final_ap"}).merge(early_ap.rename(columns={"ap": "early_ap"})[["query", "early_ap"]], on="query")
            merged["delta"] = merged["final_ap"] - merged["early_ap"]
            seeded = pd.concat([
                merged.sort_values("delta", ascending=False).head(3),
                merged.sort_values("final_ap").head(2),
                merged.assign(abs_delta=merged["delta"].abs()).sort_values("abs_delta").head(1),
            ]).drop_duplicates("query")
            filler = merged[~merged["query"].isin(set(seeded["query"]))].sort_values(["delta", "query"], ascending=[False, True])
            chosen = pd.concat([seeded, filler]).drop_duplicates("query").head(6)
            for _, row in chosen.iterrows():
                out = SELECTED_DIR / f"before_after_{dataset}_{row['query']}.png"
                if self.save_before_after_grid(dataset, row["query"], early, final, row["early_ap"], row["final_ap"], out):
                    paths.append(out)
                    rows.append(
                        {
                            "dataset": dataset,
                            "query_name": row["query"],
                            "baseline_experiment": early,
                            "final_experiment": final,
                            "baseline_AP": row["early_ap"],
                            "final_AP": row["final_ap"],
                            "delta_AP": row["delta"],
                            "artifact_path": str(out.relative_to(ROOT)).replace("\\", "/"),
                        }
                    )
                    self.add_index("before_after", out, dataset, "early-vs-final", row["query"], presentation=len(paths) == 1, notes=f"delta={row['delta']:.3f}")
        out_csv = TABLE_DIR / "before_after_cases.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        self.counts["csv_tables"] += 1
        self.add_index("table", out_csv, notes="Before/after case metadata.")
        self.save_contact_sheet(paths, SELECTED_DIR / "contact_before_after.png", "Before / After Cases", max_cols=2, thumb_w=650)
        self.add_index("contact_sheet", SELECTED_DIR / "contact_before_after.png", presentation=True, notes="Before/after contact sheet.")
        return paths

    def save_before_after_grid(self, dataset, query_name, early_exp, final_exp, early_ap, final_ap, out_path):
        try:
            _, query_image_id, bbox = parse_query_file(DATASETS[dataset]["gt_dir"] / f"{query_name}_query.txt", return_bbox=True)
            fig, axes = plt.subplots(2, 6, figsize=(21, 8))
            for r, (label_row, exp, ap) in enumerate([("Baseline", early_exp, early_ap), ("Final", final_exp, final_ap)]):
                ranking = self.read_ranking(dataset, exp, query_name)[:5]
                cells = [("query", query_image_id, self.draw_bbox(self.imread_rgb(dataset, query_image_id), bbox))]
                cells += [(self.relevance_label(dataset, query_name, image_id), image_id, self.imread_rgb(dataset, image_id)) for image_id in ranking]
                for c, (rel, image_id, img) in enumerate(cells):
                    ax = axes[r, c]
                    ax.imshow(img)
                    title = f"{label_row}\nAP={ap:.3f}" if c == 0 else f"#{c} {rel}\n{image_id}"
                    ax.set_title(title, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    self.set_border(ax, rel)
            fig.suptitle(f"{dataset.title()} Before/After: {query_name}", fontsize=18, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            self.counts["before_after_grids"] += 1
            return True
        except Exception as exc:
            self.log_issue("before_after", dataset, final_exp, query_name, str(exc))
            return False

    def failure_case_pool(self):
        rows, paths = [], []
        for dataset, final_exp in FINAL_EXP.items():
            ap = self.compute_ap_table(dataset, final_exp)
            if ap.empty:
                continue
            for _, row in ap.sort_values(["ap", "query"]).head(10).iterrows():
                out = QUAL_DIR / "failures" / dataset / f"{row['query']}.png"
                top5, rels = self.save_top5_grid(dataset, final_exp, row["query"], out, "Final", "failure", row["ap"])
                if top5:
                    paths.append(out)
                    rows.append(
                        {
                            "dataset": dataset,
                            "query_name": row["query"],
                            "experiment": final_exp,
                            "AP": row["ap"],
                            "top5_image_ids": "|".join(top5),
                            "top5_relevance_labels": "|".join(rels),
                            "artifact_path": str(out.relative_to(ROOT)).replace("\\", "/"),
                            "failure_hint_auto": self.failure_hint(dataset, row["query"], top5, rels),
                        }
                    )
                    self.counts["failure_cases"] += 1
                    self.add_index("failure_case", out, dataset, "Final", row["query"], presentation=len(paths) == 1, notes="Low final AP")
        out_csv = TABLE_DIR / "failure_cases.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        self.counts["csv_tables"] += 1
        self.add_index("table", out_csv, notes="Lowest final-method AP cases.")
        self.save_contact_sheet(paths, SELECTED_DIR / "contact_failures.png", "Failure Case Pool", max_cols=3, thumb_w=520)
        self.add_index("contact_sheet", SELECTED_DIR / "contact_failures.png", presentation=True, notes="Failure case contact sheet.")
        return paths

    def failure_hint(self, dataset, query_name, top5, rels):
        if any(label in ("good", "ok") for label in rels):
            return "some relevant images present, but ranking depth/AP remains low; manual review recommended"
        try:
            _, _, bbox = parse_query_file(DATASETS[dataset]["gt_dir"] / f"{query_name}_query.txt", return_bbox=True)
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            if area < 15000:
                return "small or ambiguous query bbox; manual review recommended"
        except Exception:
            pass
        return "visually similar architecture or repeated facade likely; manual review recommended"

    def spatial_verification_examples(self):
        if extract_local_descriptor is None:
            self.log_issue("spatial_verification", "", "", "", "extract_local_descriptor unavailable")
            return []
        rows, paths = [], []
        for dataset in ["oxford", "paris"]:
            base_exp = "multiscale_sift_bovw_k4096_l2_cosine"
            sv_exp = "multiscale_sift_bovw_k4096_l2_cosine_spatial_verify_top150_ratio0p65_inliers"
            if not self.ranking_dir(dataset, base_exp).exists() or not self.ranking_dir(dataset, sv_exp).exists():
                continue
            candidates = self.find_sv_candidate_pairs(dataset, base_exp, sv_exp)
            for kind, query, cand in candidates[:12]:
                out = SV_DIR / f"sv_example_{dataset}_{query}_{cand}.png"
                stats = self.save_sv_match_figure(dataset, query, cand, out)
                if stats:
                    paths.append(out)
                    rows.append({"dataset": dataset, "query_name": query, "candidate_image_id": cand, "example_type": kind, **stats, "artifact_path": str(out.relative_to(ROOT)).replace("\\", "/")})
                    self.add_index("spatial_verification", out, dataset, "SV", query, presentation=len(paths) == 1, notes=kind)
                    self.counts["spatial_verification_figures"] += 1
                if len(paths) >= 12:
                    break
            if len(paths) >= 12:
                break
        out_csv = TABLE_DIR / "spatial_verification_examples.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        self.counts["csv_tables"] += 1
        self.add_index("table", out_csv, notes="Spatial verification example metadata.")
        self.save_contact_sheet(paths, SELECTED_DIR / "contact_spatial_verification.png", "Spatial Verification Examples", max_cols=2, thumb_w=660)
        self.add_index("contact_sheet", SELECTED_DIR / "contact_spatial_verification.png", presentation=True, notes="Spatial verification contact sheet.")
        return paths

    def find_sv_candidate_pairs(self, dataset, base_exp, sv_exp):
        positive, negative = [], []
        for qf in list_query_files(DATASETS[dataset]["gt_dir"]):
            query, _ = parse_query_file(qf)
            try:
                base = self.read_ranking(dataset, base_exp, query)
                sv = self.read_ranking(dataset, sv_exp, query)
            except Exception:
                continue
            base_top = base[:20]
            for cand in sv[:5]:
                rel = self.relevance_label(dataset, query, cand)
                if rel in ("good", "ok") and cand in base_top and base.index(cand) > sv.index(cand):
                    positive.append(("positive_promoted_relevant", query, cand))
                    break
            for cand in base[:5]:
                rel = self.relevance_label(dataset, query, cand)
                if rel == "unknown" and cand in sv and sv.index(cand) > 10:
                    negative.append(("negative_rejected_unknown", query, cand))
                    break
            for cand in sv[:5]:
                rel = self.relevance_label(dataset, query, cand)
                if rel == "unknown":
                    negative.append(("failure_unknown_still_high", query, cand))
                    break
        return positive[:6] + negative[:6]

    def save_sv_match_figure(self, dataset, query_name, candidate_id, out_path):
        try:
            qfile = DATASETS[dataset]["gt_dir"] / f"{query_name}_query.txt"
            _, query_image_id, bbox = parse_query_file(qfile, return_bbox=True)
            q_xy_all, q_desc_all = extract_local_descriptor(dataset, query_image_id, "multiscale_sift")
            db_xy, db_desc = extract_local_descriptor(dataset, candidate_id, "multiscale_sift")
            keep = (q_xy_all[:, 0] >= bbox[0]) & (q_xy_all[:, 0] <= bbox[2]) & (q_xy_all[:, 1] >= bbox[1]) & (q_xy_all[:, 1] <= bbox[3])
            q_xy, q_desc = q_xy_all[keep], q_desc_all[keep]
            if len(q_desc) < 4 or len(db_desc) < 4:
                raise ValueError("Not enough descriptors for matching")
            q_kp = [cv.KeyPoint(float(x), float(y), 3) for x, y in q_xy]
            db_kp = [cv.KeyPoint(float(x), float(y), 3) for x, y in db_xy]
            matcher = cv.BFMatcher(cv.NORM_L2)
            knn = matcher.knnMatch(q_desc.astype(np.float32), db_desc.astype(np.float32), k=2)
            ratio_matches = []
            for pair in knn:
                if len(pair) >= 2 and pair[0].distance < 0.65 * pair[1].distance:
                    ratio_matches.append(pair[0])
            inlier_matches = []
            if len(ratio_matches) >= 4:
                src = np.float32([q_xy[m.queryIdx] for m in ratio_matches]).reshape(-1, 1, 2)
                dst = np.float32([db_xy[m.trainIdx] for m in ratio_matches]).reshape(-1, 1, 2)
                _, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
                if mask is not None:
                    inlier_matches = [m for m, ok in zip(ratio_matches, mask.ravel()) if ok]
            q_bgr = cv.imread(str(get_image_path(dataset, query_image_id)), cv.IMREAD_COLOR)
            db_bgr = cv.imread(str(get_image_path(dataset, candidate_id)), cv.IMREAD_COLOR)
            q_bgr = cv.rectangle(q_bgr.copy(), (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 4)
            raw_draw = cv.drawMatches(q_bgr, q_kp, db_bgr, db_kp, sorted(ratio_matches, key=lambda m: m.distance)[:80], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            inlier_draw = cv.drawMatches(q_bgr, q_kp, db_bgr, db_kp, inlier_matches[:80], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            rel = self.relevance_label(dataset, query_name, candidate_id)
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            axes[0].imshow(cv.cvtColor(raw_draw, cv.COLOR_BGR2RGB))
            axes[0].set_title(f"Lowe-ratio matches: {len(ratio_matches)}", fontsize=14)
            axes[1].imshow(cv.cvtColor(inlier_draw, cv.COLOR_BGR2RGB))
            axes[1].set_title(f"RANSAC inliers: {len(inlier_matches)} | relevance={rel}", fontsize=14)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{dataset.title()} SV example: {query_name} vs {candidate_id}", fontsize=17, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(out_path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            return {
                "raw_knn_matches": len(knn),
                "lowe_ratio_matches": len(ratio_matches),
                "ransac_inliers": len(inlier_matches),
                "relevance": rel,
            }
        except Exception as exc:
            self.log_issue("spatial_verification_figure", dataset, "multiscale_sift", query_name, f"{candidate_id}: {exc}")
            return {}

    def make_readme(self):
        warnings = "None." if not self.issues else f"{len(self.issues)} issues logged in logs/issues.csv."
        readme = f"""# Final Artifacts

Generated report-ready and presentation-ready artifacts for the classical Oxford/Paris image retrieval project.

## Where To Look

- `quantitative/`: mAP progress, parameter study, and final pipeline diagram.
- `qualitative/`: top-5 retrieval grids and final-method failure pools.
- `spatial_verification/`: match/inlier figures explaining geometric reranking.
- `selected_cases/`: progression grids, before/after grids, and contact sheets.
- `tables/`: CSVs backing the report and figures.
- `logs/`: generation issues and run summary.

## Recommended For A 3-Minute Presentation

1. `quantitative/map_progress_by_round.png`
2. `quantitative/final_pipeline_diagram.png`
3. One strong `selected_cases/progression_*.png`
4. One `selected_cases/before_after_*.png`
5. One failure from `qualitative/failures/`
6. One `spatial_verification/sv_example_*.png`

## Recommended For The Report

- `tables/best_per_round.csv`
- `tables/parameter_study.csv`
- `tables/method_ablation_summary.csv`
- `tables/qualitative_cases.csv`
- `tables/before_after_cases.csv`
- `tables/failure_cases.csv`
- `tables/spatial_verification_examples.csv`
- All PDF versions of the quantitative figures.

## Best Final Experiments

- Oxford: `{FINAL_EXP['oxford']}`; mAP `0.363375`
- Paris: `{FINAL_EXP['paris']}`; mAP `0.445350`

## Warnings

{warnings}
"""
        out = OUT_ROOT / "README.md"
        out.write_text(readme, encoding="utf-8")
        self.add_index("readme", out, presentation=False, report=True, notes="Artifact guide.")

    def write_logs_and_index(self):
        self.counts["csv_tables"] = len(list(TABLE_DIR.glob("*.csv"))) + 3
        issues_path = LOG_DIR / "issues.csv"
        pd.DataFrame([issue.__dict__ for issue in self.issues]).to_csv(issues_path, index=False)
        summary_path = LOG_DIR / "run_summary.csv"
        pd.DataFrame([self.counts]).to_csv(summary_path, index=False)
        index_path = OUT_ROOT / "artifact_index.csv"
        pd.DataFrame(self.index_rows).to_csv(index_path, index=False)

    def run(self):
        self.prepare_dirs()
        best_round = self.make_best_per_round()
        self.plot_map_progress(best_round)
        self.parameter_study()
        self.method_ablation_summary()
        self.pipeline_diagram()
        _, success_paths, failure_paths = self.qualitative_top5_pool()
        progression_paths = self.progression_grids()
        before_after_paths = self.before_after_grids()
        final_failure_paths = self.failure_case_pool()
        sv_paths = self.spatial_verification_examples()
        self.save_contact_sheet(success_paths, SELECTED_DIR / "contact_successes.png", "Success Case Pool", max_cols=3, thumb_w=520)
        self.add_index("contact_sheet", SELECTED_DIR / "contact_successes.png", presentation=True, notes="Success contact sheet.")
        if not failure_paths and final_failure_paths:
            failure_paths = final_failure_paths
        self.make_readme()
        self.write_logs_and_index()
        print("Final artifact generation summary")
        for key, value in self.counts.items():
            print(f"- {key}: {value}")
        print(f"- issues logged: {len(self.issues)}")
        print(f"- progression grids: {len(progression_paths)}")
        print(f"- before/after grids: {len(before_after_paths)}")
        print(f"- spatial verification figures: {len(sv_paths)}")


def main():
    builder = ArtifactBuilder()
    builder.run()


if __name__ == "__main__":
    main()

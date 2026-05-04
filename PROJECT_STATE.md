# Project State

## Current Structure

- `src/`: project code
  - `paths.py`: dataset/cache/result paths
  - `features.py`: image discovery, SIFT mean/std baseline features, bbox SIFT mean/std query features
  - `retrieval.py`: cosine/Euclidean ranking helpers and ranking `.txt` writer
  - `evaluation.py`: query parsing, bbox parsing, `compute_ap` compilation/calls, mAP evaluation
  - `experiments.py`: original SIFT mean/std ranking generation
  - `visualization.py`: top-5 and success/failure visualizations
  - `advanced_experiments.py`: advanced classical experiments, registry, BoVW/TF-IDF/QE, CSV logging
- `local-desc.ipynb`: baseline notebook workflow
- `oxford/data/img/`, `oxford/data/gt/`, `oxford/compute_ap.cpp`
- `paris/data/img/`, `paris/data/gt/`
- `cache/`: generated descriptors, vocabularies, feature matrices
- `results/`: rankings, summaries, qualitative outputs

## Baseline Pipeline

- Baseline descriptor: SIFT local descriptors aggregated to fixed-length mean/std vectors.
- Database images use full-image SIFT mean/std features.
- Query images use bbox-filtered SIFT descriptors for query feature extraction.
- Ranking supports cosine and Euclidean distance/similarity.
- Baseline results and visualizations are generated from the notebook and original `src/experiments.py`.

## Evaluation Assumptions

- Do not modify mAP/evaluation semantics.
- Evaluation uses Oxford/Paris `*_query.txt` files from `data/gt`.
- AP is computed through `compute_ap`/`compute_ap.exe` against each query's ground-truth prefix.
- mAP is the mean of per-query AP over available ranking files.
- Query parsing strips the Oxford `oxc1_` prefix when present.

## Ranking Output Format

- Ranking files are plain text.
- One image id per line.
- No scores, paths, extensions, headers, or extra columns.
- Files are saved as `results/rankings/<dataset>/<experiment>/<query_name>.txt`.
- Query image id must be excluded from its own ranking.

## Bbox Query Handling

- Ground-truth query files include query image id and bbox coordinates.
- For local-descriptor query experiments, query descriptors/keypoints are filtered to keypoints inside the bbox.
- Database images always use full-image descriptors/features.

## Implemented Experiments

- Original baseline:
  - `sift_meanstd_bbox_cosine`
  - `sift_meanstd_bbox_euclidean`
- Advanced module:
  - SIFT BoVW histograms
  - SIFT BoVW + TF-IDF weighting
  - Metrics: cosine, Euclidean, chi-square, histogram intersection where configured
  - Normalizations: `l1`, `l2`, `sqrt_l2`
  - Query expansion:
    - `q_new = alpha * q + (1 - alpha) * mean(topM vectors)`
    - Experiment names include `qe_topM` and `alpha`
  - Global descriptors:
    - HOG with bbox-cropped query descriptors and full-image database descriptors
    - HSV 3D histograms with bbox-cropped query descriptors and full-image database descriptors
  - Late fusion:
    - Score-level fusion between the best SIFT BoVW/QE config and HOG/HSV globals
  - Spatial verification:
    - SIFT keypoint/descriptor matching with Lowe ratio test
    - Query descriptors restricted to bbox
    - Database descriptors use full images
    - Homography estimation with RANSAC
    - Top-N reranking with original tail preserved
    - Failed queries fall back to original ranking
- SURF mean/std support is implemented as a guarded optional path, but unavailable in the current OpenCV build.

## Best Results So Far

Round 4 spatial-verification tuning phase:

- Oxford best:
  - `sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers`
  - mAP = `0.299115`
- Paris best:
  - `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers`
  - mAP = `0.406194`

Best spatial verification hyperparameters found in the latest controlled sweep:

- `topN=150`
- Lowe ratio `0.65`
- scoring `inlier_count`
- no `min_matches` / `min_inliers` threshold

Round 4 notes:

- The best Oxford result came from applying spatial verification to the plain `sift_bovw_k1024_l2_cosine` base, not the query-expansion base.
- The best Paris result came from applying the best Round 4 spatial setting to the previous best query-expansion base.
- Alternative Oxford base configs with the same spatial setting also improved over the previous Oxford best:
  - `sift_bovw_k256_l2_chisquare_qe_top3_alpha0p75_spatial_verify_top150_ratio0p65_inliers`: mAP `0.292443`
  - `sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p75_spatial_verify_top150_ratio0p65_inliers`: mAP `0.292399`
  - `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5_spatial_verify_top150_ratio0p65_inliers`: mAP `0.285448`
- Paris alternative-base validation for `sift_bovw_k1024_l2_cosine` was skipped/failed cleanly because the base Paris ranking directory was missing.
- Fisher Vector smoke test was not run in Round 4 because the spatial sweep was already long-running.

Previous bests before spatial tuning:

- Oxford:
  - `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers`
  - mAP = `0.274164`
- Paris:
  - `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers`
  - mAP = `0.392230`

Earlier bests before spatial verification:

- Oxford best:
  - `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25`
  - mAP = `0.218428`
- Paris best:
  - `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25`
  - mAP = `0.372444`

Current top Oxford results:

```text
sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers                       0.299115
sift_bovw_k256_l2_chisquare_qe_top3_alpha0p75_spatial_verify_top150_ratio0p65_inliers   0.292443
sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p75_spatial_verify_top150_ratio0p65_inliers  0.292399
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers      0.287424
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5_spatial_verify_top150_ratio0p65_inliers       0.285448
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p7_inliers       0.284878
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p7_inliers_minm8_mini6  0.282550
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p6_inliers       0.282507
```

Current top Paris results:

```text
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers      0.406194
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers       0.392230
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_hsv_cosine_histint_w0p9_0p05_0p05 0.376876
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_histint_w0p95_0p05                0.374942
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_chisquare_w0p95_0p05              0.374334
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p95_0p05                 0.374227
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p9_0p1                   0.374082
sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25                                               0.372444
```

## Commands / Notebook Cells Used

Baseline notebook imports:

```python
from src.features import extract_or_load_sift_meanstd
from src.experiments import generate_rankings
from src.evaluation import compile_compute_ap, evaluate_experiment
```

Baseline feature extraction:

```python
oxford_ids, oxford_X = extract_or_load_sift_meanstd("oxford")
paris_ids, paris_X = extract_or_load_sift_meanstd("paris")
```

Baseline ranking/evaluation pattern:

```python
exp_name, out_dir = generate_rankings(dataset, ids, X, metric)
compute_ap_exe = compile_compute_ap()
df = evaluate_experiment(dataset, exp_name, compute_ap_exe)
```

Advanced smoke test:

```powershell
.\.venv\Scripts\python.exe -m src.advanced_experiments --datasets oxford --groups bovw --ks 128 --normalizations l2 --metrics cosine
```

Controlled Oxford BoVW sweep:

```powershell
.\.venv\Scripts\python.exe -m src.advanced_experiments --datasets oxford --groups bovw --ks 128,256,512 --normalizations l1,l2,sqrt_l2 --metrics cosine,chisquare,hist_intersection
```

Controlled Oxford TF-IDF sweep:

```powershell
.\.venv\Scripts\python.exe -m src.advanced_experiments --datasets oxford --groups tfidf --ks 128,256,512 --normalizations l2,sqrt_l2 --metrics cosine,chisquare
```

Safe extended sweep:

```powershell
.\.venv\Scripts\python.exe -m src.advanced_experiments --safe-extended
```

Round 3 global/fusion/spatial phase:

```powershell
.\.venv\Scripts\python.exe -m src.advanced_experiments --global-fusion-spatial-phase
```

Round 4 spatial tuning phase:

```powershell
.\.venv\Scripts\python.exe -m src.advanced_experiments --spatial-tuning-phase
```

## Constraints

- Do not modify evaluation semantics.
- Do not change ranking `.txt` format.
- Do not break the existing SIFT mean/std baseline.
- Keep generated features, vocabularies, descriptors, rankings, and summaries under `cache/` and `results/`.
- Do not use deep learning, pretrained CNNs, CLIP, ViT, or neural feature extractors.
- Do not run Fisher vectors unless explicitly requested later.
- Do not run more HOG/HSV fusion experiments unless explicitly requested later.
- Avoid duplicate successful result rows; existing `status=ok` experiments should be skipped unless forced.

## Known Issues

- SURF is unavailable in the current OpenCV build: `cv.xfeatures2d.SURF_create` is missing.
- Historical failed rows exist in `results/advanced_summary.csv` from early smoke runs:
  - SURF unavailable.
  - Missing `scikit-learn` before it was installed.
  - Interrupted/corrupt descriptor cache during a timed-out run.
- One generated descriptor cache file was observed locked/corrupt:
  - `cache/descriptors/oxford/sift/oxford_002056.npz`
  - Current runner handles this by recomputing that image in memory and continuing.
- Round 4 Paris alternative-base validation failed cleanly for:
  - `sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers`
  - Reason: missing base rankings directory `results/rankings/paris/sift_bovw_k1024_l2_cosine`
- `results/advanced_summary_sorted.csv` is the preferred file for reading current best results.

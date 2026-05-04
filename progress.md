# Experimentation Progress Log

This document records the iterative experimentation process for the classical image retrieval project on Oxford/Paris Buildings. The main metric is mAP. Unless otherwise stated, experiments use query bounding boxes for query-side local descriptors and full-image descriptors for database images.

# Round 1

## Goal

Establish a first serious BoVW baseline over SIFT and identify the most promising vocabulary size, normalization, and distance metric.

## Specs

- BoVW over SIFT (plain)
  - k ∈ [128, 256, 512]
  - normalization: L1, L2, SQRT(L2)
  - metric: cosine, chisquare, histogram intersection
- BoVW over SIFT (TF-IDF)
  - k ∈ [128, 256, 512]
  - normalization: L2, SQRT(L2)
  - metric: cosine, chisquare

## Top 10 experiments Oxford ranked by mAP

| experiment                        | mAP      |
| --------------------------------- | -------- |
| sift_bovw_tfidf_k256_l2_chisquare | 0.195169 |
| sift_bovw_k256_l2_chisquare       | 0.194956 |
| sift_bovw_k512_l2_chisquare       | 0.192117 |
| sift_bovw_tfidf_k512_l2_chisquare | 0.188687 |
| sift_bovw_k512_l2_cosine          | 0.187848 |
| sift_bovw_k512_l1_cosine          | 0.187848 |
| sift_bovw_k256_l2_cosine          | 0.183445 |
| sift_bovw_k256_l1_cosine          | 0.183445 |
| sift_bovw_tfidf_k256_l2_cosine    | 0.183042 |
| sift_bovw_tfidf_k512_l2_cosine    | 0.181397 |

## Summary

Plain BoVW: 27 experiments, 27 succeeded, 433.9s  
TF-IDF BoVW: 12 experiments, 12 succeeded, 243.7s  
Total: 39 requested experiments succeeded, ~677.6s

## Observations

- The strongest signal was that chi-square distance worked better than cosine for the initial BoVW histograms.
- k=256 was the best vocabulary size in this initial range.
- TF-IDF gave a very small improvement over plain BoVW in the best configuration:
  - TF-IDF k256 L2 chi-square: 0.195169
  - plain BoVW k256 L2 chi-square: 0.194956
- Since TF-IDF only helped slightly, the next step was not only to expand TF-IDF, but also to test larger vocabularies and query expansion.

# Round 2

Run on Oxford dataset, validated on Paris with top 5 configs for Oxford.

## Goal

Expand the BoVW search space and test query expansion on the strongest configurations from Oxford.

## Specs

- BoVW over SIFT (plain)
  - k ∈ [64, 128, 256, 384, 512, 768, 1024]
  - normalization: L1, L2, SQRT(L2)
  - metric: cosine, chisquare
- BoVW over SIFT (TF-IDF)
  - k ∈ [64, 384, 768, 1024]
  - normalization: L2, SQRT(L2)
  - metric: cosine, chisquare
- Query expansion for top 5 Oxford configs
  - topM ∈ [3, 5, 10]
  - alpha ∈ [0.25, 0.5, 0.75]
  - Process:
    1. Generate the original ranking.
    2. For each query, take topM retrieved database vectors.
    3. Compute:
       `q_new = alpha * q + (1 - alpha) * mean(topM vectors)`
    4. Normalize `q_new` with the same normalization convention as the original experiment.
    5. Rerank all database images.
    6. Exclude the query image.

## Top 15 experiments Oxford ranked by mAP

| experiment                                          | mAP      |
| --------------------------------------------------- | -------- |
| sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25         | 0.218428 |
| sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5          | 0.217551 |
| sift_bovw_k1024_l2_cosine_qe_top3_alpha0p75         | 0.210319 |
| sift_bovw_k1024_l2_cosine_qe_top5_alpha0p5          | 0.203859 |
| sift_bovw_k1024_l2_cosine_qe_top5_alpha0p75         | 0.202597 |
| sift_bovw_k1024_l2_cosine                           | 0.202552 |
| sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p75 | 0.200661 |
| sift_bovw_k256_l2_chisquare_qe_top3_alpha0p75       | 0.200062 |
| sift_bovw_k256_l2_chisquare_qe_top3_alpha0p5        | 0.199446 |
| sift_bovw_k1024_l2_cosine_qe_top5_alpha0p25         | 0.199041 |
| sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p5  | 0.198792 |
| sift_bovw_tfidf_k256_l2_chisquare_qe_top5_alpha0p75 | 0.197780 |
| sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p25 | 0.197365 |
| sift_bovw_k256_l2_chisquare_qe_top3_alpha0p25       | 0.196671 |
| sift_bovw_k1024_l2_cosine_qe_top10_alpha0p75        | 0.196428 |

## Top 5 experiments Paris ranked by mAP

| experiment                                  | mAP      |
| ------------------------------------------- | -------- |
| sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25 | 0.372444 |
| sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5  | 0.370516 |
| sift_bovw_k1024_l2_cosine_qe_top5_alpha0p5  | 0.368622 |
| sift_bovw_k1024_l2_cosine_qe_top3_alpha0p75 | 0.362382 |
| sift_bovw_k1024_l2_cosine_qe_top5_alpha0p75 | 0.360800 |

## Observations

- The best Oxford configuration changed from k=256 chi-square to k=1024 L2 cosine with query expansion.
- Query expansion improved the best Oxford result from 0.202552 to 0.218428.
- The best query expansion setting was top3 with alpha=0.25, meaning that a small number of high-confidence retrieved images helped refine the query.
- Larger topM values were generally weaker, suggesting query drift when too many retrieved images are averaged into the query.
- Paris showed substantially higher mAP than Oxford for the same best configuration, indicating that Oxford was the harder/noisier dataset under this pipeline.

# Round 3

Run on Oxford dataset, validated on Paris with the strongest configs.

## Goal

Test whether global descriptors and geometric reranking can improve over the strongest BoVW + query expansion configuration from Round 2.

Base local configuration:

- `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25`

## Specs

### Standalone global descriptors

- HOG
  - images resized to a fixed square size before descriptor extraction
  - query features computed from bbox crop when available
  - database features computed from full images
  - metric: cosine
- HSV color histogram
  - query features computed from bbox crop when available
  - database features computed from full images
  - histogram normalization
  - metrics: cosine, chisquare, histogram intersection

### Late fusion

Fusion was done at the score level instead of feature concatenation.

- Local score: best SIFT BoVW + query expansion score
- Global score: HOG or HSV score
- Per-query score normalization before combination
- Fusion formula:
  - `final_score = w_local * local_score + w_global * global_score`

Tested fusion variants:

- SIFT BoVW QE + HOG cosine
  - weights: 0.95/0.05, 0.90/0.10, 0.80/0.20
- SIFT BoVW QE + HSV
  - HSV metrics: cosine, chisquare, histogram intersection
  - weights: 0.95/0.05, 0.90/0.10, 0.80/0.20
- SIFT BoVW QE + HOG + HSV
  - weights: 0.90/0.05/0.05, 0.80/0.10/0.10

### Spatial verification reranking

Spatial verification was applied on top of the best local configuration from Round 2.

Process:

1. Start from the ranking produced by `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25`.
2. For each query, take the topN candidates.
3. Use SIFT descriptors restricted to the query bbox for the query image.
4. Use full-image SIFT descriptors for database candidates.
5. Match query descriptors against candidate descriptors.
6. Apply Lowe ratio test.
7. Estimate a homography with RANSAC.
8. Score candidates by number of RANSAC inliers.
9. Rerank only the topN candidates by geometric consistency.
10. Append the remaining original ranking unchanged.

Tested spatial verification parameters:

- topN ∈ [50, 100]
- Lowe ratio ∈ [0.7, 0.75, 0.8]
- score: RANSAC inlier count

## Top 15 experiments Oxford ranked by mAP

| mAP      | experiment                                                                               |
| -------- | ---------------------------------------------------------------------------------------- |
| 0.274164 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers       |
| 0.271842 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p75_inliers      |
| 0.270420 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p8_inliers       |
| 0.259873 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top50_ratio0p7_inliers        |
| 0.259101 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top50_ratio0p75_inliers       |
| 0.256345 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top50_ratio0p8_inliers        |
| 0.220022 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p95_0p05                 |
| 0.219756 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p9_0p1                   |
| 0.218428 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25                                              |
| 0.217551 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5                                               |
| 0.213698 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p8_0p2                   |
| 0.213245 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_hsv_cosine_histint_w0p9_0p05_0p05 |
| 0.211986 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_chisquare_w0p95_0p05              |
| 0.211024 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_histint_w0p95_0p05                |
| 0.210319 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p75                                              |

## Top 10 experiments Paris ranked by mAP

| mAP      | experiment                                                                               |
| -------- | ---------------------------------------------------------------------------------------- |
| 0.392230 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers       |
| 0.376876 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_hsv_cosine_histint_w0p9_0p05_0p05 |
| 0.374942 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_histint_w0p95_0p05                |
| 0.374334 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_chisquare_w0p95_0p05              |
| 0.374227 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p95_0p05                 |
| 0.374082 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p9_0p1                   |
| 0.372444 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25                                              |
| 0.372412 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_hsv_cosine_histint_w0p8_0p1_0p1   |
| 0.371421 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_cosine_w0p95_0p05                 |
| 0.370816 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_histint_w0p9_0p1                  |

## Summary

Best Round 2 configuration:

- Oxford: 0.218428
- Paris: 0.372444

Best Round 3 configuration:

- Oxford: 0.274164
- Paris: 0.392230

Absolute improvement over Round 2 best:

- Oxford: +0.055736 mAP
- Paris: +0.019786 mAP

Relative improvement over Round 2 best:

- Oxford: +25.5%
- Paris: +5.3%

## Observations

- Spatial verification produced the largest improvement so far, especially on Oxford.
- top100 consistently outperformed top50, suggesting that the correct matches often appear within a wider candidate set and can be promoted by geometric consistency.
- ratio=0.7 was best among the tested Lowe ratio thresholds. The stricter thresholds 0.75 and 0.8 were slightly worse.
- Fusion with HOG/HSV produced only marginal gains or even small regressions compared with the Round 2 best configuration.
- The best fusion result on Oxford was HOG cosine with local/global weight 0.95/0.05, improving only from 0.218428 to 0.220022.
- The best fusion result on Paris was HOG+HSV fusion, improving from 0.372444 to 0.376876.
- This suggests that global descriptors provide weak complementary information, but the retrieval task is dominated by local landmark correspondences.
- The next promising direction is to tune spatial verification further rather than continue broad fusion experiments.

# Round 4

Run on Oxford dataset, validated on Paris with the strongest spatial-verification configuration.

## Goal

Tune spatial verification after Round 3 and check whether the gain depends on the query-expanded base ranking or also appears on other strong BoVW/TF-IDF rankings.

Round 3 showed that spatial verification was the dominant improvement path, so Round 4 focused on a narrower but more careful sweep around the geometric reranking stage.

## Specs

### Main spatial verification tuning

The main sweep extended the Round 3 spatial-verification setup.

- topN candidates reranked by spatial verification:
  - tested top100 and top150 variants in the recorded top results
  - top150 became the most important candidate depth in this round
- Lowe ratio threshold:
  - tested ratio values around 0.6, 0.65, and 0.7
  - ratio=0.65 became the strongest setting
- spatial verification score variants:
  - `inliers`: rerank by RANSAC inlier count
  - `inliersplusratio`: combine absolute inlier count with a normalized inlier-ratio term
- threshold variants:
  - no explicit minimum threshold
  - `minm8_mini6`: require at least 8 matches and 6 inliers before trusting the geometric score

### Base rankings tested

Round 4 also tested whether spatial verification works better when applied to different first-stage rankings.

Main base ranking from Round 3:

- `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25`

Alternative base rankings that appeared in the Oxford top results:

- `sift_bovw_k1024_l2_cosine`
- `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5`
- `sift_bovw_k256_l2_chisquare_qe_top3_alpha0p75`
- `sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p75`

This was useful because the best final Oxford result came from applying spatial verification to the non-query-expanded `sift_bovw_k1024_l2_cosine` ranking, while the best Paris result still came from the query-expanded `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25` ranking.

## Top 15 experiments Oxford ranked by mAP

| mAP      | experiment                                                                                                   |
| -------- | -------------------------------------------------------------------------------------------------------- |
| 0.299115 | sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers                                        |
| 0.292443 | sift_bovw_k256_l2_chisquare_qe_top3_alpha0p75_spatial_verify_top150_ratio0p65_inliers                    |
| 0.292399 | sift_bovw_tfidf_k256_l2_chisquare_qe_top3_alpha0p75_spatial_verify_top150_ratio0p65_inliers              |
| 0.287424 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers                      |
| 0.285448 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p5_spatial_verify_top150_ratio0p65_inliers                       |
| 0.284878 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p7_inliers                       |
| 0.282550 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p7_inliers_minm8_mini6           |
| 0.282507 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p6_inliers                       |
| 0.279704 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers_minm8_mini6          |
| 0.278861 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p7_inliersplusratio_minm8_mini6  |
| 0.278731 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliersplusratio_minm8_mini6 |
| 0.278468 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p65_inliers                      |
| 0.274164 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers                       |
| 0.274095 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p6_inliersplusratio              |
| 0.272177 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p6_inliers                       |

## Top 10 experiments Paris ranked by mAP

| mAP      | experiment                                                                                   |
| -------- | ---------------------------------------------------------------------------------------- |
| 0.406194 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers      |
| 0.392230 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top100_ratio0p7_inliers       |
| 0.376876 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_hsv_cosine_histint_w0p9_0p05_0p05 |
| 0.374942 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_histint_w0p95_0p05                |
| 0.374334 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_chisquare_w0p95_0p05              |
| 0.374227 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p95_0p05                 |
| 0.374082 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_cosine_w0p9_0p1                   |
| 0.372444 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25                                              |
| 0.372412 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hog_hsv_cosine_histint_w0p8_0p1_0p1   |
| 0.371421 | sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_fusion_hsv_cosine_w0p95_0p05                 |

## Summary

Best Round 3 configuration:

- Oxford: 0.274164
- Paris: 0.392230

Best Round 4 configuration:

- Oxford: 0.299115
  - `sift_bovw_k1024_l2_cosine_spatial_verify_top150_ratio0p65_inliers`
- Paris: 0.406194
  - `sift_bovw_k1024_l2_cosine_qe_top3_alpha0p25_spatial_verify_top150_ratio0p65_inliers`

Absolute improvement over Round 3 best:

- Oxford: +0.024951 mAP
- Paris: +0.013964 mAP

Relative improvement over Round 3 best:

- Oxford: +9.1%
- Paris: +3.6%

Absolute improvement over Round 2 best:

- Oxford: +0.080687 mAP
- Paris: +0.033750 mAP

Relative improvement over Round 2 best:

- Oxford: +36.9%
- Paris: +9.1%

## Observations

- Increasing spatial verification depth from top100 to top150 was useful. The previous Round 3 best, `top100_ratio0p7_inliers`, reached 0.274164 on Oxford, while the best tuned top150 setting reached 0.299115.
- The best Oxford result did not use query expansion before spatial verification. This suggests that query expansion can improve the first-stage global ranking, but it may also introduce query drift before the geometric reranker.
- The best Paris result still used query expansion, so the effect is dataset-dependent. A reasonable final report framing is that query expansion is helpful, but spatial verification should be validated both with and without it.
- The best scoring mode remained plain RANSAC inlier count. Variants using `inliersplusratio` were consistently below the best `inliers` configurations in the top results.
- The `minm8_mini6` threshold did not improve the best configuration. It may have filtered out borderline but useful geometric evidence.
- `ratio=0.65` improved over the previous `ratio=0.7` setting, suggesting a slightly stricter Lowe ratio helped reduce false matches before RANSAC.
- Alternative base rankings became competitive after spatial verification. In particular, both k256 chi-square variants with query expansion reached approximately 0.2924 on Oxford after top150/ratio0.65 spatial verification.
- The project direction is now clear: the final system should be built around SIFT BoVW plus spatial verification, with query expansion treated as a dataset-dependent optional stage rather than an unconditional improvement.

# Overall Progress Summary

| Stage | Best Oxford mAP | Best Paris mAP | Best method |
| ----- | --------------- | -------------- | ----------- |
| Round 1 | 0.195169 | — | SIFT BoVW TF-IDF k256 L2 chi-square |
| Round 2 | 0.218428 | 0.372444 | SIFT BoVW k1024 L2 cosine + query expansion |
| Round 3 | 0.274164 | 0.392230 | SIFT BoVW k1024 L2 cosine + query expansion + spatial verification |
| Round 4 | 0.299115 | 0.406194 | SIFT BoVW + tuned spatial verification top150 ratio0.65 |

## Main conclusions so far

- BoVW over SIFT gives a strong classical retrieval baseline.
- Metric choice matters: chi-square was strongest for the initial histogram setup, but cosine became strongest with larger k and query expansion.
- Query expansion improved retrieval in Round 2, especially with top3 and alpha=0.25, but Round 4 showed that it is not always optimal before spatial verification.
- Global descriptor fusion with HOG/HSV provided only minor gains, suggesting limited complementarity for landmark retrieval.
- Spatial verification is the most important improvement because it directly addresses false positives caused by visually similar but geometrically inconsistent local features.
- The best Oxford score progressed from 0.195169 in Round 1 to 0.299115 in Round 4, an absolute gain of 0.103946 mAP.
- The best Paris score progressed from 0.372444 in Round 2 to 0.406194 in Round 4, an absolute gain of 0.033750 mAP.
- The strongest final direction is a classical SIFT BoVW retrieval system followed by tuned spatial verification. For the report, this gives a clean experimental story: baseline BoVW → larger vocabulary and query expansion → global fusion attempt → geometric reranking as the decisive improvement.

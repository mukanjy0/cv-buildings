# Round 1

## Specs

- BOVW over SIFT (plain)
  - k ∈ [128, 256, 512]
  - normalization: L1, L2, SQRT(L2)
  - metric: cosine, chisquare, histogram intersection
- BOVW over SIFT (TF-IDF)
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

# Round 2

Run on Oxford dataset, validated on Paris with top 5 configs (for Oxford).

## Specs

- BOVW over SIFT (plain)
  - k ∈ [64, 128, 256, 384, 512, 768, 1024]
  - normalization: L1, L2, SQRT(L2)
  - metric: cosine, chisquare
- BOVW over SIFT (TF-IDF)
  - k ∈ [64, 384, 768, 1024]
  - normalization: L2, SQRT(L2)
  - metric: cosine, chisquare
- Query expansion for Top 5 Oxford Configs
  - topM ∈ [3, 5, 10]
  - alpha ∈ [0.25, 0.5, 0.75]
  - Process:
    1. Generate the original ranking.
    2. For each query, take topM retrieved database vectors.
    3. Compute:
       q*new = alpha * q + (1 - alpha) \_ mean(topM vectors)
    4. Normalize q_new with the same normalization convention as the original experiment.
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

# Round 3

Run on Oxford dataset, validated on Paris with top 5 configs (for Oxford).

## Specs

- HOG
  - resizing images to fixed quadratic size
  - metric: cosine
- HSV
  - histogram normalization
  - metric: chisquare, histogram intersection
- Fusion
- Spatial verification reranking

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

# Round 4

## Top 15 experiments Oxford ranked by mAP

| Score    | Method                                                                                                   |
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

| Score    | Method                                                                                   |
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

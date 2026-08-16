# Supplementary Table S8. Calibrated held-out-state abstention and represented-state label transfer under the primary operational policy

Values are arithmetic means \(\pm\) sample standard deviations across five donor splits. The score cutoff is the method-specific 95th percentile estimated from the corresponding matched full-reference control. Held-out-state abstention and represented-state coverage are evaluated on disjoint cell sets and are therefore not complements; post-abstention macro-F1 is conditional on retained represented-state cells. Methods without an applicable operational abstention policy are omitted.

| Endpoint | Method | Held-out-state abstention rate | Represented-state coverage | Post-abstention macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| HLA-DRhi cDC2 | CoRe-OT | 0.684 \(\pm\) 0.094 | 0.921 \(\pm\) 0.029 | 0.961 \(\pm\) 0.010 |
| HLA-DRhi cDC2 | Uniform UOT | 0.249 \(\pm\) 0.042 | 0.956 \(\pm\) 0.005 | 0.961 \(\pm\) 0.009 |
| HLA-DRhi cDC2 | Seurat | 0.087 \(\pm\) 0.019 | 0.988 \(\pm\) 0.003 | 0.984 \(\pm\) 0.005 |
| HLA-DRhi cDC2 | SingleR | 0.022 \(\pm\) 0.005 | 0.993 \(\pm\) 0.003 | 0.947 \(\pm\) 0.008 |
| HLA-DRhi cDC2 | CellTypist | 0.000 \(\pm\) 0.000 | 1.000 \(\pm\) 0.000 | 0.930 \(\pm\) 0.009 |
| HLA-DRhi cDC2 | scmap-cell | 0.062 \(\pm\) 0.052 | 0.961 \(\pm\) 0.019 | 0.932 \(\pm\) 0.010 |
| HLA-DRhi cDC2 | scmap-cluster | 0.041 \(\pm\) 0.009 | 0.947 \(\pm\) 0.006 | 0.943 \(\pm\) 0.006 |
| HLA-DRhi cDC2 | CHETAH | 0.306 \(\pm\) 0.027 | 0.969 \(\pm\) 0.005 | 0.964 \(\pm\) 0.004 |
| ISG+ cDC2 | CoRe-OT | 0.649 \(\pm\) 0.025 | 0.952 \(\pm\) 0.008 | 0.947 \(\pm\) 0.008 |
| ISG+ cDC2 | Uniform UOT | 0.612 \(\pm\) 0.022 | 0.952 \(\pm\) 0.005 | 0.929 \(\pm\) 0.009 |
| ISG+ cDC2 | Seurat | 0.044 \(\pm\) 0.023 | 0.973 \(\pm\) 0.005 | 0.967 \(\pm\) 0.004 |
| ISG+ cDC2 | SingleR | 0.032 \(\pm\) 0.012 | 0.976 \(\pm\) 0.004 | 0.922 \(\pm\) 0.003 |
| ISG+ cDC2 | CellTypist | 0.000 \(\pm\) 0.000 | 1.000 \(\pm\) 0.000 | 0.913 \(\pm\) 0.005 |
| ISG+ cDC2 | scmap-cell | 0.016 \(\pm\) 0.018 | 0.948 \(\pm\) 0.020 | 0.901 \(\pm\) 0.007 |
| ISG+ cDC2 | scmap-cluster | 0.014 \(\pm\) 0.012 | 0.948 \(\pm\) 0.002 | 0.929 \(\pm\) 0.005 |
| ISG+ cDC2 | CHETAH | 0.054 \(\pm\) 0.020 | 0.952 \(\pm\) 0.003 | 0.929 \(\pm\) 0.004 |

_Sources: `units/hiha_parameter_and_calibration/data/comparison/compare_detection_summary.csv` and `units/hiha_parameter_and_calibration/data/comparison/compare_shared_label_transfer_summary.csv`._

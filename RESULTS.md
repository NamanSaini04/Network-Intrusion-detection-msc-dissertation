# Results

Macro F1 was used as the primary ranking metric because both intrusion-detection benchmarks are imbalanced and overall accuracy can conceal weak minority-class detection.

## CICIDS2017

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **0.969** | **0.899** | **0.910** | **0.892** | **0.970** |
| XGBoost | 0.929 | 0.867 | 0.889 | 0.833 | 0.937 |
| MLP Neural Network | 0.946 | 0.800 | 0.747 | 0.741 | 0.937 |
| SGD Linear SVM | 0.887 | 0.758 | 0.748 | 0.727 | 0.879 |
| SGD Logistic Regression | 0.818 | 0.714 | 0.704 | 0.676 | 0.809 |
| CNN | 0.812 | 0.746 | 0.663 | 0.663 | 0.801 |
| LSTM | 0.504 | 0.421 | 0.382 | 0.381 | 0.496 |

## UNSW-NB15

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | 0.670 | 0.456 | **0.527** | **0.448** | 0.721 |
| XGBoost | **0.757** | **0.525** | 0.438 | 0.424 | **0.756** |
| SGD Linear SVM | 0.659 | 0.372 | 0.526 | 0.378 | 0.704 |
| SGD Logistic Regression | 0.631 | 0.377 | 0.510 | 0.348 | 0.674 |
| MLP Neural Network | 0.328 | 0.215 | 0.226 | 0.152 | 0.291 |
| LSTM | 0.074 | 0.007 | 0.100 | 0.014 | 0.010 |
| CNN | 0.008 | 0.001 | 0.100 | 0.002 | 0.000 |

## Cross-dataset interpretation

Random Forest ranked first by Macro F1 on both benchmarks. This is a ranking comparison, not a direct numerical comparison of dataset difficulty.

The lower UNSW-NB15 scores are interpreted in the context of a different attack taxonomy, feature construction, class distribution and evaluation design. CICIDS2017 used an 80:20 split of a capped experimental sample, whereas UNSW-NB15 retained the full official test set.

### Rare-class caution

CICIDS2017 test support was extremely small for Heartbleed (2), SQL Injection (4) and Infiltration (7). UNSW-NB15 Worms had only 44 official test examples. Metrics for these classes should therefore not be treated as stable evidence of generalisable detection performance.

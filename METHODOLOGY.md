# Methodology

## Experimental design

The dissertation used a quantitative supervised multi-class classification design across two benchmark network-intrusion datasets.

### Shared design choices

- Random seed: **42**
- Feature selection: **ANOVA F-score**
- Selected features: **40**
- Primary ranking metric: **Macro F1**
- Additional evaluation: accuracy, macro precision, macro recall, weighted F1, class-specific reports and confusion matrices
- SMOTE: **not used in the final reported pipeline**

## CICIDS2017

- Cleaned dataset: **2,830,743 flows across 15 classes**
- Experimental capped sample: **31,193**
- Train/test design: **80:20 stratified split**
- Training records: **24,954**
- Test records: **6,239**
- Large classes were capped; very small classes were retained in full.
- Variance filtering, ANOVA selection and scaling were fitted after the split on training data.
- Median imputation occurred during cleaning before the split in the supplied implementation. This is reported as a limited preprocessing-leakage risk.

### CICIDS2017 model settings of note

- Random Forest: 200 trees, no maximum depth, minimum leaf size 2, balanced_subsample
- XGBoost: 70 trees, depth 4, learning rate 0.08; saved reported run used a **6,068-row stratified training subset**
- MLP: scikit-learn (96, 48), ReLU, Adam, early stopping
- CNN: 80 epochs, batch size 1024
- LSTM: 15 epochs, batch size 1024

## UNSW-NB15

- Official train/test design retained.
- Training data down-sampled proportionally to **60,000 records**.
- Full official test set retained: **82,332 records**.
- Categorical variables `proto`, `service` and `state` were one-hot encoded.
- Numerical imputation, encoding, scaling and feature selection were fitted on sampled training data and applied to the official test data.

### UNSW-NB15 model settings of note

- Random Forest: 80 trees, maximum depth 16, minimum leaf size 2, balanced_subsample
- XGBoost: 40 trees, depth 4, learning rate 0.08
- MLP/CNN/LSTM: PyTorch, weighted cross-entropy, **1 epoch**, batch size **8192**

## Class imbalance

CICIDS2017 used capped sampling and class weighting where supported. UNSW-NB15 used proportional training down-sampling and model-specific weighting where used.

The independent effect of class weighting or sampling was **not isolated through an ablation study**, so the dissertation does not claim that these procedures caused the observed performance.

## Interpretation boundary

The project is a comparative benchmark study, not a deployment-readiness evaluation. The results do not establish that any model is universally superior for operational intrusion detection.

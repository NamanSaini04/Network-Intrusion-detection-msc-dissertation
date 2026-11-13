# Dissertation

## Comparative Analysis of Machine Learning and Deep Learning Models for Network Intrusion Detection Using Feature Engineering and Class Imbalance Handling

**Author:** Naman Saini  
**Programme:** MSc Artificial Intelligence, University of Bath  
**Year:** 2026  
**Dissertation mark:** 72%

## Abstract

This dissertation compares machine learning and deep learning models for multi-class network intrusion detection using CICIDS2017 and UNSW-NB15. CICIDS2017 is used as the primary benchmark, while UNSW-NB15 provides a cross-dataset benchmark evaluation under a different feature structure, attack taxonomy, and test design. A reproducible Python pipeline was developed for data cleaning, identifier removal, categorical encoding, feature scaling, ANOVA F-score feature selection, class-imbalance handling, and model evaluation. The models evaluated were SGD Logistic Regression, SGD Linear SVM, Random Forest, XGBoost, MLP, one-dimensional CNN, and LSTM. Performance was assessed using accuracy, macro precision, macro recall, macro F1-score, weighted F1-score, confusion matrices, and class-specific reports.

On CICIDS2017, Random Forest achieved the strongest macro F1-score of 0.892. On UNSW-NB15, Random Forest achieved the strongest macro F1-score among the tested models, although its macro F1-score was lower. Because the datasets use different sampling and test designs, the numerical scores are not treated as directly comparable; instead, the cross-dataset analysis focuses on model ranking, minority-class behaviour and dataset dependence. The findings support tree-based ensembles as strong baselines for engineered tabular IDS data. The neural results are interpreted cautiously because the implementations, training budgets and flow-level representations were limited and were not designed as optimised deep-learning benchmarks.

## Public-repository note

This GitHub repository is the public portfolio/reproducibility edition of the dissertation. Student identifiers, assessment administration documents, raw datasets and similarity/AI-detection reports are deliberately excluded.

The exact assessed dissertation PDF is preserved separately. The research findings, methodology, reported result tables and reproducibility code are represented in this repository without removing the limitations stated in the submitted dissertation.

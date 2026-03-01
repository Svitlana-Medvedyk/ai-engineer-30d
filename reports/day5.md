# Day 5 — Cross-validation (stability)

## Setup
- Dataset: Titanic (Kaggle competition)
- Features (v2): Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone
- Preprocessing:
  - Numeric: median imputation
  - Categorical: most_frequent imputation + one-hot encoding
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Metrics: ROC-AUC, Accuracy

## Results (5-fold CV)

### LogisticRegression (v2)
- ROC-AUC scores: [0.87529644, 0.86189840, 0.83348930, 0.83028075, 0.87953730]
- ROC-AUC mean ± std: **0.85610044 ± 0.02063642**
- Accuracy scores: [0.78212291, 0.79775281, 0.79775281, 0.79213483, 0.83146067]
- Accuracy mean ± std: **0.80024481 ± 0.01661942**

### RandomForest (v2 features + same preprocessing)
- ROC-AUC scores: [0.90658762, 0.85574866, 0.83328877, 0.85574866, 0.89456189]
- ROC-AUC mean ± std: **0.86918712 ± 0.02717555**
- Accuracy scores: [0.82681564, 0.78089888, 0.79213483, 0.82584270, 0.82584270]
- Accuracy mean ± std: **0.81030695 ± 0.01975002**

## Interpretation
- RandomForest has better mean ROC-AUC and Accuracy than LogisticRegression on average.
- RandomForest is less stable (higher std), while LogisticRegression is slightly more stable and simpler to interpret/deploy.

## Next steps
1) Add a strong feature from Name (Title: Mr/Mrs/Miss/Master) and rerun CV.
2) Consider probability calibration and threshold selection based on the target objective (precision vs recall).
3) Try a lightweight boosted model (if available) or tune RF (max_depth, min_samples_leaf).

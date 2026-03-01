# Day 6 — Feature engineering: Title + Cross-validation

## Feature added
- Added `Title` extracted from `Name` (Mr/Mrs/Miss/Master/Other).
- Rationale: Title captures gender + age/social context and can improve predictive power.

## CV setup
- CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Preprocessing:
  - Numeric: median imputation
  - Categorical: most_frequent imputation + one-hot encoding
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone, Title

## Results (5-fold CV)

### LogisticRegression (v3 with Title)
- ROC-AUC mean ± std: **0.87188138 ± 0.01842227**
- Accuracy mean ± std: **0.82827192 ± 0.00784083**

### RandomForest (v3 with Title)
- ROC-AUC mean ± std: **0.87041399 ± 0.02621513**
- Accuracy mean ± std: **0.81031950 ± 0.01154687**

## Comparison vs Day 5 (without Title)
- LR v2: ROC-AUC **0.85610044 ± 0.02063642**, Acc **0.80024481 ± 0.01661942**
- RF v2: ROC-AUC **0.86918712 ± 0.02717555**, Acc **0.81030695 ± 0.01975002**

## Takeaways
- Title feature significantly improved LogisticRegression: 
  - ROC-AUC increased (~0.856 → ~0.872) and accuracy increased (~0.800 → ~0.828),
  - stability improved (lower std for both AUC and accuracy).
- RandomForest did not improve meaningfully with Title (accuracy stayed ~the same, AUC ~similar).
- Current best model (by CV): **LogisticRegression v3 with Title**.

## Next steps
1) Try more simple features (FareBin, IsChild) and rerun CV.
2) Threshold selection on validation folds (optimize recall vs precision for class 1).
3) Try a boosted model (if available) or tune RF depth/min_samples.

# Day 3 — Titanic baseline as code

## Model
- Pipeline: impute (median for numeric, most_frequent for categorical) + one-hot + LogisticRegression
- Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone
- Split: train_test_split(test_size=0.2, stratify=y, random_state=42)

## Results
- v1 (LogReg): Acc = 0.804, ROC-AUC = 0.84, CM = [[98, 12], [23, 46]]
- v2 (+FamilySize, IsAlone): Acc2 = 0.804, ROC-AUC2 = 0.85, CM2 = [[97, 13], [22, 47]]
- Delta: Acc +0.000, AUC +0.0066

## Interpretation
- Added features improved ranking quality (AUC), but accuracy at threshold=0.5 stayed the same.

## Next steps
1) Threshold tuning to improve recall for Survived=1 (precision/recall tradeoff).
2) Try RandomForest or GradientBoosting as baseline v3.
3) Feature engineering: Title from Name, FareBin, IsChild.

# Day 2 — Titanic EDA + Baseline

## Dataset
- Source: Kaggle Titanic competition
- Rows/Cols: 891 / 12
- Target: Survived (0/1)

## Data quality
- Missing values: Cabin=687, Age=177, Embarked=2

## Key findings (5)
- Sex: female survival ~0.742, male ~0.189
- Pclass: 1st ~0.630, 2nd ~0.473, 3rd ~0.242
- Age: children (0–12] ~0.580 survive more than older groups (50–80] ~0.344
- Embarked: C looks highest overall; driven partly by more 1st-class passengers (C has ~50.6% Pclass=1; Q mostly Pclass=3)
- Interaction: female+Pclass=1 ~0.968 vs male+Pclass=3 ~0.135

## Baselines
- v1 (LogReg): Acc=0.804, ROC-AUC=0.844
- v2 (+FamilySize, IsAlone): Acc=0.804, ROC-AUC=0.850 (ΔAUC ≈ +0.0066)

## Next steps
1) Tune threshold to improve recall for Survived=1 (tradeoff precision/recall).
2) Try a different baseline model (RandomForest / GradientBoosting).
3) Add simple feature engineering: Title from Name (Mr/Mrs/Miss/Master), IsChild, FareBin.

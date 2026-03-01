# Learning Log — AI Engineer 30 days

## Day 0
- Setup:
  - Kaggle notebook created + internet on
  - GitHub repo created: ai-engineer-30d
  - Notebook uploaded to GitHub: notebooks/00_kaggle_setup.ipynb
- Shipped:
  - Repo structure started
 
Train/Val/Test:
train — навчаємо, val — підбираємо налаштування, test — фінальна чесна оцінка.
Leakage:
коли модель “бачить” інформацію з майбутнього або з тесту під час навчання/підготовки даних.
Metrics:
для дисбалансу класів краще F1/ROC-AUC, а не тільки accuracy.

## Day 1
- Topics:
  - ML project structure: train/val/test, leakage, metrics
- Practice:
  - Create notebook template: EDA → baseline → eval → notes
- Shipped:
  - Notebook template + README goal & success criteria
 
## Day 2
- Topics:
  - Pandas basics (Kaggle Learn): reading/writing, selecting, filtering, groupby
  - EDA workflow: shape/info, missing values, target balance, hypotheses
- Practice:
  - Titanic EDA in Kaggle:
    - Checked dataset structure (shape/info)
    - Found missing values (Cabin, Age, Embarked)
    - Tested hypotheses for Survived vs Sex / Pclass / Embarked / AgeBin
    - Explored interactions (Sex + Pclass) and Embarked vs Pclass
- Shipped:
  - `reports/day2.md` with 5 findings + next steps
  - (Optional) Titanic notebook saved to GitHub in `notebooks/` (if uploaded)
 
## Day 3
baseline as code (train_titanic.py) + report shipped.

## Day 4
- Topics:
  - Threshold tuning: how changing the probability threshold shifts precision vs recall
  - Comparing different baselines (LogReg vs RandomForest)
- Practice:
  - LogisticRegression v2 threshold sweep:
    - t=0.3: precision=0.667, recall=0.812, f1=0.732, CM=[[82,28],[13,56]]
    - t=0.5: precision=0.783, recall=0.681, f1=0.729, CM=[[97,13],[22,47]]
    - Chosen threshold: t=0.3 (goal: reduce FN / catch more survivors)
  - RandomForest baseline:
    - t=0.5: Acc=0.8156, ROC-AUC=0.8332, CM=[[97,13],[20,49]]
    - Threshold sweep highlights:
      - t=0.4: precision=0.726, recall=0.768, f1=0.746, CM=[[90,20],[16,53]]
- Shipped:
  - `reports/day4.md` with threshold results + RF baseline comparison
 
  ## Day 5
- Topics:
  - Cross-validation (StratifiedKFold) for more reliable evaluation
  - Comparing models by mean ± std (stability)
- Practice:
  - 5-fold CV for LogisticRegression v2: ROC-AUC = 0.8561 ± 0.0206, Accuracy = 0.8002 ± 0.0166
  - 5-fold CV for RandomForest: ROC-AUC = 0.8692 ± 0.0272, Accuracy = 0.8103 ± 0.0198
- Shipped:
  - `reports/day5.md` with CV results and interpretation
 
  ## Day 6
- Topics:
  - Feature engineering from text columns (extracting Title from Name)
  - Validating improvements with 5-fold cross-validation (mean ± std)
- Practice:
  - Implemented Title feature (Mr/Mrs/Miss/Master/Other)
  - 5-fold CV results:
    - LR v3 (with Title): ROC-AUC = 0.8719 ± 0.0184, Accuracy = 0.8283 ± 0.0078
    - RF v3 (with Title): ROC-AUC = 0.8704 ± 0.0262, Accuracy = 0.8103 ± 0.0115
- Shipped:
  - `reports/day6.md` with CV results + comparison vs Day 5

  ## Day 7
- Topics:
  - Probability calibration (Platt scaling / sigmoid)
  - Production-style threshold selection based on objective (F1 vs Recall)
- Practice:
  - Ran calibration and compared probability samples:
    - Uncalibrated: [0.0714, 0.9589, 0.6063, 0.9466, 0.0733]
    - Calibrated:   [0.0785, 0.9468, 0.5940, 0.9349, 0.0828]
- Shipped:
  - `reports/day7.md` (calibration notes + threshold selection plan)

  



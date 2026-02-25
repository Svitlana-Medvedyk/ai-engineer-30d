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

  



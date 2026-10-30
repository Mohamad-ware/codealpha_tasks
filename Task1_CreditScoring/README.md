# Credit Scoring Project

Predicts whether a borrower will experience serious financial distress (90+ days past due) within
the next two years, using their credit history and financial information.

## Dataset

Kaggle's [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) dataset: 150,000
borrowers. The target is `SeriousDlqin2yrs` (1 = serious delinquency). Features include:

- `RevolvingUtilizationOfUnsecuredLines`, `DebtRatio`, `MonthlyIncome`
- `age`, `NumberOfDependents`
- `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines`
- Past-due counts: `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`

The data is not included in this repository. Download `cs-training.csv` and `cs-test.csv` from
Kaggle and put them in a `data/` folder:

```
CreditsScoringProject/
├── data/
│   ├── cs-training.csv
│   └── cs-test.csv
└── notebooks/
    └── CreditScoring.ipynb
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/CreditScoring.ipynb
```

## What the notebook does

1. **Exploration.** Data shape, types, missing values, descriptive statistics and class balance of the target.
2. **Cleaning.** Removes duplicate rows and fills missing `MonthlyIncome` and `NumberOfDependents` with the median.
3. **Visualisation.** Income and age distributions, a correlation heatmap and a look at the past-due counts.
4. **Feature engineering.** `TotalPayments` (all late payments combined) and `TotalLoans` (open credit lines plus real-estate loans).
5. **Modelling.** Stratified 80/20 train/validation split; trains Logistic Regression, a Decision Tree and a Random Forest.
6. **Evaluation.** Accuracy, precision, recall, F1, confusion matrix and ROC-AUC.
7. **Interpretation.** Random Forest feature importances.

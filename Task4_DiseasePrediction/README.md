# Task 4 – Disease Prediction from Medical Data

Predicts the likelihood of a disease from structured patient data (symptoms, age, blood tests,
clinical measurements). It compares four classifiers on three UCI medical datasets.

| Dataset | Patients | Features | Predicts |
|---------|---------:|---------:|----------|
| Heart Disease (Cleveland) | 303 | 13: age, sex, chest-pain type, blood pressure, cholesterol, max heart rate, ECG… | heart disease present |
| Pima Indians Diabetes | 768 | 8: glucose, insulin, BMI, blood pressure, age, pregnancies… | diabetes onset |
| Breast Cancer Wisconsin (Diagnostic) | 569 | 30: tumour cell-nuclei measurements | malignant tumour |

Models: **Logistic Regression**, **SVM**, **Random Forest**, **XGBoost**.

## Files

| File | Purpose |
|------|---------|
| `datasets.py` | Downloads and cleans each dataset (missing-value markers, binary target, numeric vs categorical columns). Also loads any custom CSV |
| `models.py` | Preprocessing pipeline (imputation, scaling, one-hot encoding) and the four models with their hyper-parameter grids |
| `train.py` | Trains, tunes and compares the models, then saves the best one together with its plots and results |
| `predict.py` | Scores new patients from a CSV or JSON input: probability, prediction and risk level |
| `examples/` | Sample patient files for trying `predict.py` |

## Setup

```bash
pip install -r requirements.txt
```

Datasets are downloaded into `data/` on first run.

## Usage

```bash
# Train and compare all models on all three datasets
python train.py --dataset all

# One dataset, or quick run without hyper-parameter tuning
python train.py --dataset heart
python train.py --dataset diabetes --no-tune

# Your own CSV with a 0/1 target column
python train.py --csv my_data.csv --target outcome

# Predict for new patients
python predict.py --model models/heart_best.joblib --input examples/heart_patients.csv
python predict.py --model models/diabetes_best.joblib --input examples/diabetes_patients.csv
python predict.py --model models/diabetes_best.joblib --json "{\"glucose\": 148, \"bmi\": 33.6, \"age\": 50}"
```

For each dataset, `train.py` writes to `outputs/`:
- `<dataset>_results.csv`: accuracy, precision, recall, F1, ROC-AUC (test set) and cross-validated ROC-AUC for every model
- `<dataset>_roc.png`: ROC curves of all models
- `<dataset>_confusion.png`: confusion matrix of the best model
- `<dataset>_importance.png`: the features that drive the best model's predictions

The best pipeline is saved as `models/<dataset>_best.joblib`.

## Approach

1. **Data cleaning.** Missing values are marked as missing: `?` in the heart data, and impossible zeros (glucose, BMI, insulin…) in the diabetes data. Each target becomes binary, 1 = disease.
2. **Preprocessing (in one scikit-learn `Pipeline`).** Median imputation and standardisation for numeric features; most-frequent imputation and one-hot encoding for categorical ones (chest-pain type, ECG result, thalassemia…). Keeping preprocessing inside the pipeline stops test-set information from leaking into training and lets the saved model accept raw patient data.
3. **Training.** Stratified 80/20 train/test split. Each model is tuned with 5-fold stratified cross-validated grid search, using ROC-AUC as the score. Class imbalance is handled with `class_weight="balanced"` (LR, SVM, RF) and `scale_pos_weight` (XGBoost).
4. **Evaluation.** Every model is scored on the untouched test set. Recall (sensitivity) is reported alongside accuracy, because missing a sick patient usually costs more than a false alarm.
5. **Interpretation.** Permutation importance shows which inputs matter most, for any model type.

Typical test ROC-AUC on these datasets is ~0.90 for heart disease, ~0.83 for diabetes and ~0.99 for breast cancer.

> **Disclaimer:** an educational project. The predictions are statistical estimates, not a medical diagnosis.

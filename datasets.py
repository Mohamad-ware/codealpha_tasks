"""Loaders for the three UCI medical datasets.

Each loader returns a `MedicalDataset` with a feature DataFrame, a binary target (1 = disease present)
and which columns are numeric vs categorical, so the preprocessing pipeline can treat them correctly.
Raw files are downloaded once and cached in `data/`.
"""
import os
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer as sk_breast_cancer

DATA_DIR = "data"

HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


@dataclass
class MedicalDataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    numeric: list
    categorical: list = field(default_factory=list)
    description: str = ""

    def summary(self):
        pos = int(self.y.sum())
        return (f"{self.name}: {len(self.y)} patients, {self.X.shape[1]} features "
                f"({len(self.numeric)} numeric, {len(self.categorical)} categorical), "
                f"{pos} positive ({pos / len(self.y):.1%}), "
                f"{int(self.X.isna().any(axis=1).sum())} rows with missing values")


def _download(url, filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_heart():
    """UCI Heart Disease (Cleveland): 303 patients, 13 clinical features."""
    columns = [
        "age",       # years
        "sex",       # 1 = male, 0 = female
        "cp",        # chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic
        "trestbps",  # resting blood pressure (mm Hg)
        "chol",      # serum cholesterol (mg/dl)
        "fbs",       # fasting blood sugar > 120 mg/dl
        "restecg",   # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
        "thalach",   # maximum heart rate achieved
        "exang",     # exercise-induced angina
        "oldpeak",   # ST depression induced by exercise
        "slope",     # slope of peak exercise ST segment
        "ca",        # number of major vessels coloured by fluoroscopy (0-3)
        "thal",      # 3 normal, 6 fixed defect, 7 reversible defect
        "target",    # 0 = no disease, 1-4 = disease severity
    ]
    df = pd.read_csv(_download(HEART_URL, "heart_cleveland.csv"), names=columns, na_values="?")
    y = (df.pop("target") > 0).astype(int).rename("heart_disease")
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    numeric = [c for c in df.columns if c not in categorical]
    return MedicalDataset("heart", df, y, numeric, categorical,
                          "Presence of heart disease from clinical and exercise-test measurements")


def load_diabetes():
    """Pima Indians Diabetes: 768 female patients, 8 features (originally from the UCI repository)."""
    columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
               "bmi", "diabetes_pedigree", "age", "outcome"]
    df = pd.read_csv(_download(DIABETES_URL, "pima_diabetes.csv"), names=columns)
    # A value of 0 is physically impossible for these measurements: it encodes "not recorded".
    for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df[col] = df[col].replace(0, np.nan)
    y = df.pop("outcome").astype(int).rename("diabetes")
    return MedicalDataset("diabetes", df, y, list(df.columns),
                          description="Onset of diabetes within 5 years from diagnostic measurements")


def load_breast_cancer():
    """Breast Cancer Wisconsin (Diagnostic): 569 tumours, 30 features computed from cell-nuclei images."""
    raw = sk_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    y = (1 - raw.target).rename("malignant")  # sklearn uses 0 = malignant; flip so 1 = disease
    return MedicalDataset("breast_cancer", X, y, list(X.columns),
                          description="Malignant vs benign tumour from fine-needle aspirate measurements")


def load_csv(path, target):
    """Any other tabular dataset: object/low-cardinality columns are treated as categorical."""
    df = pd.read_csv(path)
    y = df.pop(target).astype(int)
    categorical = [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 5]
    numeric = [c for c in df.columns if c not in categorical]
    name = os.path.splitext(os.path.basename(path))[0]
    return MedicalDataset(name, df, y, numeric, categorical, f"Custom dataset from {path}")


LOADERS = {"heart": load_heart, "diabetes": load_diabetes, "breast_cancer": load_breast_cancer}

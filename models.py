"""Preprocessing pipeline and the four classifiers with their hyper-parameter search spaces."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # keep the other three models usable without xgboost installed
    XGBClassifier = None

RANDOM_STATE = 42


def build_preprocessor(numeric, categorical):
    """Impute missing values, scale numeric features and one-hot encode categorical ones."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [("num", numeric_pipe, numeric)]
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(transformers)


def get_models(y_train):
    """Return {name: (estimator, param_grid)}. Grid keys are prefixed for use inside a Pipeline."""
    imbalance = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__kernel": ["rbf", "linear"], "model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
            {"model__n_estimators": [300], "model__max_depth": [None, 5, 10],
             "model__min_samples_leaf": [1, 3], "model__max_features": ["sqrt", 0.5]},
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=imbalance,
                          n_jobs=-1, random_state=RANDOM_STATE),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 5],
             "model__learning_rate": [0.03, 0.1], "model__subsample": [0.8, 1.0],
             "model__colsample_bytree": [0.8, 1.0]},
        )
    else:
        print("xgboost is not installed: skipping XGBoost (pip install xgboost)")
    return models


def make_pipeline(preprocessor, estimator):
    return Pipeline([("prep", preprocessor), ("model", estimator)])

"""Train and compare SVM, Logistic Regression, Random Forest and XGBoost on medical datasets.

For every dataset: stratified train/test split -> grid-searched cross-validation (ROC-AUC) per model
-> held-out test metrics -> comparison table, ROC curves, confusion matrix and feature importance
-> the best pipeline (preprocessing + model) is saved for `predict.py`.

Examples:
    python train.py --dataset heart
    python train.py --dataset all
    python train.py --csv my_data.csv --target outcome
"""
import argparse
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from datasets import LOADERS, load_csv
from models import RANDOM_STATE, build_preprocessor, get_models, make_pipeline

OUT_DIR, MODEL_DIR = "outputs", "models"


def evaluate(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),  # sensitivity: share of sick patients we catch
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def run_dataset(ds, cv_folds, test_size, tune):
    print(f"\n{'=' * 70}\n{ds.summary()}\n{ds.description}\n{'=' * 70}")
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=test_size, stratify=ds.y, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(ds.numeric, ds.categorical)
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, (estimator, grid) in get_models(y_train).items():
        t0 = time.time()
        pipe = make_pipeline(clone(preprocessor), estimator)
        search = GridSearchCV(pipe, grid if tune else {}, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        metrics = evaluate(best, X_test, y_test)
        rows.append({"model": name, "cv_roc_auc": search.best_score_, **metrics})
        fitted[name] = best
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"{name:<20} CV AUC {search.best_score_:.3f} | test AUC {metrics['roc_auc']:.3f} "
              f"acc {metrics['accuracy']:.3f} recall {metrics['recall']:.3f}  "
              f"({time.time() - t0:.1f}s) {params}")

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print(f"\n{results.round(3).to_string(index=False)}")
    results.to_csv(os.path.join(OUT_DIR, f"{ds.name}_results.csv"), index=False)

    best_name = results.loc[0, "model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_test, best_pipe.predict(X_test), target_names=["healthy", ds.y.name]))

    save_plots(ds, fitted, best_name, X_test, y_test)
    path = os.path.join(MODEL_DIR, f"{ds.name}_best.joblib")
    joblib.dump({
        "pipeline": best_pipe, "model_name": best_name, "dataset": ds.name, "target": ds.y.name,
        "features": list(ds.X.columns), "metrics": results.loc[0].to_dict(),
    }, path)
    print(f"Saved {path}; plots and results table in {OUT_DIR}/")
    return results.assign(dataset=ds.name)


def save_plots(ds, fitted, best_name, X_test, y_test):
    # ROC curves of all models on the same axes
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curves - {ds.name}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{ds.name}_roc.png"), dpi=120); plt.close(fig)

    best = fitted[best_name]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, display_labels=["healthy", ds.y.name],
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{best_name} - {ds.name}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{ds.name}_confusion.png"), dpi=120); plt.close(fig)

    # Permutation importance works on raw input columns, for any model type
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=20,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    order = imp.importances_mean.argsort()[::-1][:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(order) + 1.2))
    ax.barh(X_test.columns[order], imp.importances_mean[order], xerr=imp.importances_std[order], color="#4C72B0")
    ax.set_xlabel("drop in ROC-AUC when feature is shuffled")
    ax.set_title(f"Most important features - {ds.name}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{ds.name}_importance.png"), dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=[*LOADERS, "all"])
    ap.add_argument("--csv", help="train on your own CSV instead (needs --target)")
    ap.add_argument("--target", help="target column of --csv (0/1)")
    ap.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-tune", action="store_true", help="skip grid search, use default hyper-parameters")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if args.csv:
        if not args.target:
            ap.error("--csv requires --target")
        datasets = [load_csv(args.csv, args.target)]
    else:
        names = list(LOADERS) if args.dataset == "all" else [args.dataset]
        datasets = [LOADERS[n]() for n in names]

    summary = pd.concat([run_dataset(ds, args.cv, args.test_size, not args.no_tune) for ds in datasets])
    if len(datasets) > 1:
        table = summary.pivot(index="model", columns="dataset", values="roc_auc").round(3)
        print(f"\n{'=' * 70}\nTest ROC-AUC across datasets\n{table.to_string()}")
        summary.to_csv(os.path.join(OUT_DIR, "all_results.csv"), index=False)


if __name__ == "__main__":
    main()

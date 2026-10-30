"""Predict disease risk for new patients with a model saved by `train.py`.

Examples:
    python predict.py --model models/heart_best.joblib --input examples/heart_patients.csv
    python predict.py --model models/diabetes_best.joblib --json '{"glucose": 148, "bmi": 33.6, "age": 50}'

Missing features are allowed: they are filled in by the pipeline's imputer, with a warning.
"""
import argparse
import json

import joblib
import pandas as pd

RISK_LEVELS = [(0.3, "low"), (0.6, "moderate"), (1.01, "high")]


def risk_level(p):
    return next(label for limit, label in RISK_LEVELS if p < limit)


def predict(bundle, patients):
    missing = [c for c in bundle["features"] if c not in patients.columns]
    if missing:
        print(f"Warning: missing features {missing} will be imputed")
    X = patients.reindex(columns=bundle["features"])
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    out = patients.copy()
    out["probability"] = proba.round(3)
    out["prediction"] = (proba >= 0.5).astype(int)
    out["risk"] = [risk_level(p) for p in proba]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="CSV with one patient per row")
    src.add_argument("--json", help="one patient as a JSON object, or a list of them")
    ap.add_argument("--output", help="optional CSV to write the predictions to")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    if args.input:
        patients = pd.read_csv(args.input)
    else:
        data = json.loads(args.json)
        patients = pd.DataFrame(data if isinstance(data, list) else [data])

    m = bundle["metrics"]
    print(f"Model: {bundle['model_name']} trained on '{bundle['dataset']}' "
          f"(test ROC-AUC {m['roc_auc']:.3f}, recall {m['recall']:.3f})\n")
    result = predict(bundle, patients)
    for i, row in result.iterrows():
        verdict = bundle["target"].replace("_", " ") if row["prediction"] else "no " + bundle["target"].replace("_", " ")
        print(f"Patient {i + 1}: {verdict:<22} probability {row['probability']:.1%}  risk {row['risk']}")
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved {args.output}")
    print("\nThis is a statistical estimate for study purposes, not a medical diagnosis.")


if __name__ == "__main__":
    main()

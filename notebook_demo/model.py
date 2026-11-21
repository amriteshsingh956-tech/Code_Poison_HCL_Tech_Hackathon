"""Shared training and inference for the notebook's Linear Regression model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "notebook_demo"
NUMERIC = ["Age", "Annual_Income", "Website_Visits", "Previous_Purchases"]
CATEGORICAL = ["Membership_Type", "Preferred_Category", "City_Tier", "Discount_Used"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Amount_Spent"


def validate(frame):
    """Validate raw features, retaining missing values for fitted imputation."""
    missing = [name for name in FEATURES if name not in frame]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing) + ". Download the CSV template for the expected headers.")
    if frame.empty:
        raise ValueError("The CSV contains no customer rows.")
    result = frame[FEATURES].copy().replace(r"^\s*$", np.nan, regex=True)
    for name in NUMERIC:
        parsed = pd.to_numeric(result[name], errors="coerce")
        bad = (result[name].notna() & parsed.isna()) | (parsed.notna() & (~np.isfinite(parsed) | (parsed < 0)))
        if bad.any():
            rows = (np.flatnonzero(bad.to_numpy())[:5] + 1).tolist()
            raise ValueError(f"{name}: enter a finite, nonnegative number or leave blank. Invalid data rows: {rows}.")
        result[name] = parsed.astype(float)
    for name in CATEGORICAL:
        result[name] = result[name].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    return result


def predict(frame, bundle):
    """Append predictions without changing row order or identifier strings."""
    if "Predicted_Amount_Spent" in frame:
        raise ValueError("Remove the existing Predicted_Amount_Spent column before uploading.")
    result = frame.copy()
    result["Predicted_Amount_Spent"] = bundle["pipeline"].predict(validate(frame))
    return result


def train(source):
    """Refit the notebook winner on its seeded 80% split; save real evaluation."""
    frame = pd.read_csv(source)
    x = validate(frame)
    y = pd.to_numeric(frame[TARGET], errors="raise")
    if y.isna().any() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Targets must be present, finite and nonnegative.")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    pipeline = Pipeline([("preprocessing", preprocessing), ("model", LinearRegression())])
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    def scores(actual, estimates):
        return {"MAE": float(mean_absolute_error(actual, estimates)),
                "MSE": float(mean_squared_error(actual, estimates)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, estimates))),
                "R2": float(r2_score(actual, estimates))}
    metrics = {"model": "Linear Regression", "test": scores(y_test, predicted),
               "benchmark": scores(y_test, np.full(len(y_test), y_train.mean())),
               "train_r2": float(pipeline.score(x_train, y_train)),
               "train_rows": len(x_train), "test_rows": len(x_test), "seed": 42,
               "dataset_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
               "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib", "streamlit", "matplotlib"]}}
    schema = {"numeric": {name: {"default": float(x_train[name].median()),
                                     "min": float(x_train[name].min()), "max": float(x_train[name].max())} for name in NUMERIC},
              "categorical": {name: sorted(x_train[name].dropna().unique().tolist()) for name in CATEGORICAL}}
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipeline, "metrics": metrics, "schema": schema}
    joblib.dump(bundle, OUT / "model.pkl")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predicted,
                  "residual": y_test.to_numpy() - predicted}).to_csv(OUT / "evaluation.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].scatter(y_test, predicted, s=5, alpha=.18, color="#167d9a")
    bounds = [min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())]
    axes[0].plot(bounds, bounds, "--", color="#dc8549")
    axes[0].set(xlabel="Actual spend", ylabel="Predicted spend", title="Actual versus predicted")
    axes[1].scatter(predicted, y_test - predicted, s=5, alpha=.18, color="#167d9a")
    axes[1].axhline(0, color="#dc8549", linestyle="--")
    axes[1].set(xlabel="Predicted spend", ylabel="Actual minus predicted", title="Residuals: positive = underprediction")
    fig.savefig(OUT / "evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    train(parser.parse_args().source)

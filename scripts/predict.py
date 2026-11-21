"""Local inference for one ShopSmart profile or a CSV of profiles."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .audit_intake import PROFILE, ROOT

MODEL = ROOT / "artifacts/selected_pipeline.pkl"
BUNDLE = MODEL.parent / "spend_bundle.pkl"


def predict_outputs(frame):
    """Return consistent numeric estimates and bands from the complete saved bundle."""
    from .spend_bands import assign_bands
    bundle = joblib.load(BUNDLE)
    values = bundle["pipeline"].predict(inputs(frame))
    return pd.DataFrame({"Predicted_Amount_Spent": values,
                         "Predicted_Amount_Spent_Range": assign_bands(values,bundle["cutoffs"])},
                        index=frame.index)


def predict_customer(profile):
    """Return both website outputs for one profile."""
    return predict_outputs(pd.DataFrame([profile])).iloc[0].to_dict()


def inputs(frame):
    """Select raw fields, normalise blanks, and validate supplied numeric values."""
    absent = set(PROFILE) - set(frame.columns)
    if absent:
        raise ValueError(f"Required fields missing: {sorted(absent)}")
    data = frame[PROFILE].copy()
    for name in PROFILE:
        values = data[name].astype("string").str.strip().replace("", pd.NA)
        if name in ("Age", "Annual_Income"):
            numbers = pd.to_numeric(values, errors="coerce").astype(float)
            bad = values.notna() & numbers.isna()
            if bad.any():
                row = np.flatnonzero(bad)[0] + 1
                raise ValueError(f"Check {name} in data row {row}. Enter a number without commas or leave it blank.")
            if (numbers < 0).any() or np.isinf(numbers).any():
                raise ValueError(f"{name} requires finite, nonnegative values or a blank")
            data[name] = numbers
        else:
            data[name] = values.astype(object).where(values.notna(), np.nan)
    return data


def predict_one(profile):
    """Return predicted spending for one raw profile using the packaged pipeline."""
    return float(joblib.load(MODEL).predict(inputs(pd.DataFrame([profile])))[0])


def predict_csv(source, destination):
    """Write batch predictions alongside input records in their original order."""
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Choose a different output file to preserve input records")
    data = predict_batch(source)
    data.to_csv(destination, index=False)


def predict_batch(source, dual=False):
    """Score a local path or in-memory CSV while preserving identifier strings."""
    data = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if data.empty:
        raise ValueError("Add at least one customer row below the header.")
    if "predicted_spend" in data:
        raise ValueError("Rename or remove the existing predicted_spend column before scoring.")
    if dual:
        outputs = predict_outputs(data)
        if set(outputs.columns).intersection(data.columns):
            raise ValueError("Remove previous prediction columns before scoring.")
        return pd.concat([data,outputs],axis=1)
    data["predicted_spend"] = joblib.load(MODEL).predict(inputs(data))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--row", help="JSON profile")
    group.add_argument("--csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dual", action="store_true", help="Return both amount and range from the bundle")
    args = parser.parse_args()
    if args.csv:
        if not args.out:
            parser.error("--csv requires --out")
        if args.dual:
            if args.csv.resolve()==args.out.resolve():
                raise ValueError("Use a different output path")
            predict_batch(args.csv, dual=True).to_csv(args.out,index=False)
        else:
            predict_csv(args.csv, args.out)
    else:
        print(json.dumps(predict_customer(json.loads(args.row))) if args.dual else predict_one(json.loads(args.row)))

"""Audit the supplied ShopSmart CSV without changing customer records."""

import argparse
import csv
import hashlib
import io
import platform
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/shopsmart.csv"
PROFILE = ["Age", "Annual_Income", "Membership_Type", "City_Tier"]
REVIEW = ["Website_Visits", "Previous_Purchases", "Preferred_Category", "Discount_Used"]
TARGET = "Amount_Spent"
NUMERIC = ["Age", "Annual_Income", "Website_Visits", "Previous_Purchases", TARGET]
EXCLUDED = ["Amount_Spent_Range", "Age_Group"]
QUESTION = (
    "For each row, what customer or source observation and snapshot date does it "
    "represent, what dates and currency does Amount_Spent cover, and which inputs "
    "were known before that period began? Please specify the windows and formulas "
    "for visits, purchases, category preference and discounts, and whether any use "
    "transactions from the target period."
)


def load_snapshot(source):
    """Parse the exact supplied bytes and preserve a local copy without overwriting."""
    payload = source.read_bytes()
    header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))), [])
    expected = set(PROFILE + REVIEW + EXCLUDED + [TARGET])
    if len(header) != len(set(header)) or set(header) != expected:
        raise ValueError(f"Unexpected CSV schema. Expected {sorted(expected)}, found {header}")
    frame = pd.read_csv(io.BytesIO(payload), dtype="string", encoding="utf-8-sig")
    if frame.empty:
        raise ValueError("CSV contains no observations")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        with RAW.open("xb") as target:
            target.write(payload)
    if RAW.read_bytes() != payload:
        raise ValueError("Local snapshot differs from the supplied file. Original copy preserved.")
    return frame, hashlib.sha256(payload).hexdigest()


def audit(frame):
    """Print aggregate checks and reject malformed or infinite numeric inputs."""
    data = frame.apply(lambda col: col.str.strip().replace("", pd.NA))
    issues = []
    for name in NUMERIC:
        original = data[name]
        values = pd.to_numeric(original, errors="coerce")
        invalid = int((original.notna() & values.isna()).sum())
        infinite = int(np.isinf(values.dropna().astype(float)).sum())
        issues.append((name, invalid, infinite))
        data[name] = values
    checks = pd.DataFrame(issues, columns=["field", "parse_errors", "infinite"])
    print(checks.to_string(index=False))
    if checks[["parse_errors", "infinite"]].to_numpy().any():
        raise ValueError("Numeric validation failed. See field counts above.")
    print(f"Rows: {len(data):,} | Columns: {len(data.columns)}")
    print(pd.DataFrame({"type": data.dtypes, "missing": data.isna().sum(),
                        "distinct": data.nunique()}).to_string())
    print(data[NUMERIC].describe().T.round(3).to_string())
    print("Duplicate rows beyond first:", data.duplicated().sum())
    print("Repeated eight-field profiles beyond first:", data.duplicated(PROFILE + REVIEW).sum())
    print("Negative numeric values:", (data[NUMERIC] < 0).sum().to_dict())
    print("Zero spending:", data[TARGET].eq(0).sum())
    for name in ["Age", "Website_Visits", "Previous_Purchases"]:
        fraction = data[name].dropna()
        fraction = fraction[fraction % 1 != 0]
        print(f"{name}: {len(fraction)} fractional observations")
    for name in ["Amount_Spent_Range", "Age_Group"]:
        measure = TARGET if name == "Amount_Spent_Range" else "Age"
        print(data.groupby(name)[measure].agg(["size", "min", "max"]).sort_values("min").to_string())
    print("Target: Amount_Spent (spending prototype; currency and period unknown)")
    print("Candidate profile fields, subject to snapshot confirmation:", ", ".join(PROFILE))
    print("Hold until historical timing is confirmed:", ", ".join(REVIEW))
    print("Exclude Amount_Spent_Range (target leakage), Age_Group (duplicates age information), "
          "and Amount_Spent itself from predictors.")
    print("Provider question:", QUESTION)
    print("Repeat from repository root: python scripts/audit_intake.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW)
    args = parser.parse_args()
    records, digest = load_snapshot(args.source)
    print(f"Python {platform.python_version()} | pandas {pd.__version__} | NumPy {np.__version__}")
    print(f"Local snapshot: data/raw/shopsmart.csv | SHA-256: {digest}")
    audit(records)

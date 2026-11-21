"""Prepare and check the fingerprint-bound Step 2 development split."""
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, PredefinedSplit, train_test_split

from .audit_intake import RAW, ROOT, NUMERIC, PROFILE, REVIEW, TARGET, load_snapshot
from .preprocessing import build_preprocessor

SEED = 42
OUT = ROOT / "data/prepared"
FINGERPRINT = "00c1b3fdb65d5f7780ffa2ccc5a8f9aab32560001e1b4715aca26cd7bd50fdc1"


def eligible_rows(frame):
    """Apply fixed validity rules, keeping original zero-based CSV record positions."""
    clean = frame.apply(lambda col: col.str.strip().replace("", pd.NA))
    invalid = pd.Series(False, index=clean.index)
    for column in NUMERIC:
        original = clean[column]
        values = pd.to_numeric(original, errors="coerce").astype(float)
        invalid |= original.notna() & (~np.isfinite(values) | (values < 0))
        clean[column] = values
    missing = frame[TARGET].str.strip().replace("", pd.NA).isna()
    eligible = clean.loc[~(missing | invalid)]
    duplicates = eligible.duplicated()
    print(f"Missing targets: {missing.sum()}, invalid records: {invalid.sum()}, "
          f"duplicate records removed: {duplicates.sum()}")
    return eligible.loc[~duplicates]


def load_development():
    """Load development data and shared CV, aligned by saved source position."""
    import hashlib
    meta = json.loads((OUT / "split.json").read_text())
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Raw CSV fingerprint differs from split manifest")
    manifest = pd.read_csv(OUT / "manifest.csv", index_col="row_position")
    dev = pd.read_csv(OUT / "development.csv", index_col="row_position")
    folds = manifest.loc[manifest.partition == "development", "validation_fold"]
    if not dev.index.is_unique or not dev.index.sort_values().equals(folds.index.sort_values()):
        raise ValueError("Development records and fold assignments disagree")
    return dev, PredefinedSplit(folds.loc[dev.index].astype(int))


def main():
    """Write reusable assignments and exercise preprocessing on development data only."""
    frame, digest = load_snapshot(RAW)
    if digest != FINGERPRINT:
        raise ValueError("Input differs from the audited Step 1 CSV")
    clean = eligible_rows(frame)
    dev_ids, test_ids = train_test_split(clean.index.to_numpy(), test_size=.2, random_state=SEED)
    dev_ids.sort()
    assert not np.intersect1d(dev_ids, test_ids).size
    assert np.array_equal(np.union1d(dev_ids, test_ids), clean.index)
    manifest = pd.DataFrame(index=clean.index.rename("row_position"))
    manifest["partition"] = "test"
    manifest["validation_fold"] = pd.Series(pd.NA, index=manifest.index, dtype="Int64")
    manifest.loc[dev_ids, "partition"] = "development"
    for fold, (_, valid) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(dev_ids)):
        manifest.loc[dev_ids[valid], "validation_fold"] = fold
    assert manifest.loc[test_ids, "validation_fold"].isna().all()
    OUT.mkdir(exist_ok=True, parents=True)
    manifest.to_csv(OUT / "manifest.csv")
    clean.loc[dev_ids, PROFILE + REVIEW + [TARGET]].to_csv(OUT / "development.csv", index_label="row_position")
    meta = dict(sha256=digest, seed=SEED, development=len(dev_ids), test=len(test_ids),
                folds=5, python=platform.python_version(), pandas=pd.__version__,
                numpy=np.__version__, sklearn=sklearn.__version__)
    (OUT / "split.json").write_text(json.dumps(meta, indent=2) + "\n")
    dev, cv = load_development()
    seen = []
    for i, (train, valid) in enumerate(cv.split()):
        assert not np.intersect1d(train, valid).size
        seen.extend(valid)
        print(f"Fold {i}: train={len(train)}, validation={len(valid)}")
    assert np.array_equal(np.sort(seen), np.arange(len(dev)))
    train, valid = next(cv.split())
    for history in (False, True):
        fields = PROFILE + (REVIEW if history else [])
        for scale in (False, True):
            prep = build_preprocessor(scale_numeric=scale, include_confirmed_history=history)
            prep.fit(dev.iloc[train][fields])
            probe = dev.iloc[valid[:1]][fields].copy()
            probe["Annual_Income"] = np.nan
            probe["Membership_Type"] = "Previously unseen"
            probe["City_Tier"] = np.nan
            if history:
                probe["Discount_Used"] = np.nan
            values = prep.transform(probe)
            names = list(prep.get_feature_names_out())
            assert not any(TARGET in name for name in names)
            assert np.isfinite(values).all() and values[0, names.index("categories__City_Tier_Unknown")] == 1
            assert all(values[0, j] == 0 for j, name in enumerate(names) if "Membership_Type_" in name)
            if history:
                assert values[0, names.index("discount__Discount_Used")] == 2
    print(json.dumps(meta, indent=2))
    print("Checks passed: partition and fold coverage, disjoint positions, feature allowlist, "
          "missing numeric/category and unseen category transforms for all four configurations.")


if __name__ == "__main__":
    main()

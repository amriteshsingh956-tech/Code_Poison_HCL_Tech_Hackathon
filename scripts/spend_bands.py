"""Fold-local mapping between regression amounts and supplied ordered bands."""
import numpy as np
import pandas as pd

LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]


def infer_cutoffs(amounts, labels):
    """Infer midpoints from training rows, rejecting missing bands or overlap."""
    bands = pd.DataFrame({"amount":np.asarray(amounts), "band":np.asarray(labels)})
    if bands.isna().any().any() or not np.isfinite(bands.amount).all():
        raise ValueError("Band mapping requires finite training amounts and labels.")
    if set(bands.band) != set(LABELS):
        raise ValueError("Training labels must contain all five expected bands.")
    bounds = bands.groupby("band").amount.agg(["min","max"]).loc[LABELS]
    lower, upper = bounds["max"].to_numpy()[:-1], bounds["min"].to_numpy()[1:]
    if np.any(lower >= upper):
        raise ValueError("Conflicting overlapping labels require a separate range classifier.")
    return (lower + (upper-lower)/2).tolist()


def assign_bands(amounts, cutoffs):
    """Assign an exact cutoff to the higher band using four ascending boundaries."""
    values = np.asarray(amounts, dtype=float)
    edges = np.asarray(cutoffs,dtype=float)
    if edges.shape != (4,) or np.any(np.diff(edges)<=0) or not np.isfinite(values).all():
        raise ValueError("Require finite predictions and four ordered cutoffs.")
    return np.asarray(LABELS)[np.searchsorted(edges,values,side="right")]


def interval_label(band, cutoffs):
    """Describe the numeric interval with an inclusive lower and exclusive upper edge."""
    i = LABELS.index(band)
    if i == 0:
        return f"Below {cutoffs[0]:,.2f}"
    if i == 4:
        return f"{cutoffs[3]:,.2f} or higher"
    return f"{cutoffs[i-1]:,.2f} to below {cutoffs[i]:,.2f}"

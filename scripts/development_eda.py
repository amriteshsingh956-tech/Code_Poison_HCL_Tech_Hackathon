"""Run aggregate exploratory analysis on saved development rows."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

from .audit_intake import PROFILE, REVIEW, TARGET, ROOT
from .prepare_split import load_development
from .preprocessing import build_preprocessor

OUTPUT = ROOT / "reports/generated/eda"


def save_plot(fig, name):
    """Save a complete figure and release its memory."""
    fig.savefig(OUTPUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Summarise development rows and test simple reconstruction hypotheses."""
    data, folds = load_development()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spend = data[TARGET]
    missing = data.isna().agg(["sum", "mean"]).T
    missing.columns = ["missing_count", "missing_fraction"]
    missing.to_csv(OUTPUT / "missingness.csv")
    corr = data.select_dtypes("number").corr()
    numeric = data[corr.columns]
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    fences = 1.5 * (q3-q1)
    summary = numeric.describe().T.assign(skew=numeric.skew(),
        below_fence=(numeric < q1-fences).sum(), above_fence=(numeric > q3+fences).sum())
    summary.to_csv(OUTPUT / "numeric_summary.csv")
    corr.to_csv(OUTPUT / "correlations.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(spend, bins=48, color="#187780")
    ax.axvline(spend.median(), color="#912d16", label="Median")
    ax.legend()
    ax.set(title="Development spending distribution", xlabel="Amount spent (source units)", ylabel="Records")
    save_plot(fig, "spend.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i, j in np.ndindex(corr.shape):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.iloc[i,j])>.65 else "black")
    ax.set_title("Development Pearson correlations")
    fig.colorbar(image, ax=ax)
    save_plot(fig, "correlations.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained", sharex=True)
    tables = []
    for ax, field in zip(axes.flat, ["Membership_Type","City_Tier","Preferred_Category","Discount_Used"]):
        seg = data.groupby(field, dropna=False)[TARGET].agg(["count","mean","median"]).sort_values("mean")
        tables.append(seg.rename_axis("segment").reset_index().assign(field=field))
        labels = [f"{label} (n={n:,})" for label,n in seg["count"].items()]
        ax.barh(labels, seg["mean"], color="#187780")
        ax.set(title=field.replace("_"," "), xlabel="Mean spend (source units)")
    save_plot(fig,"segments.png")
    segments = pd.concat(tables)
    segments.to_csv(OUTPUT / "segments.csv", index=False)

    diagnostics = []
    for history in [False, True]:
        fields = PROFILE + (REVIEW if history else [])
        probe = make_pipeline(build_preprocessor(scale_numeric=True, include_confirmed_history=history),
                              LinearRegression())
        predictions = cross_val_predict(probe, data[fields], spend, cv=folds)
        diagnostics.append(dict(inputs="all eight" if history else "four profile fields",
            r2=r2_score(spend,predictions), rmse=root_mean_squared_error(spend,predictions),
            within_one_percent=np.mean(np.abs(predictions-spend)<=.01*np.abs(spend))))
    scores = pd.DataFrame(diagnostics)
    scores.to_csv(OUTPUT / "reconstruction.csv",index=False)
    train, valid = next(folds.split())
    for scale in (True, False):
        for engineered in (True, False):
            prep = build_preprocessor(scale_numeric=scale, engineered=engineered)
            prep.fit(data.iloc[train][PROFILE])
            sample = data.iloc[valid[:1]][PROFILE].copy()
            sample["Annual_Income"] = np.nan
            assert np.isfinite(prep.transform(sample)).all()
            assert any("Log_Income" in s for s in prep.get_feature_names_out()) == engineered
    print("Development rows:",len(data),"Missing cells:",int(missing.missing_count.sum()))
    print(summary.round(3).to_string(),corr.round(3).to_string(),segments.round(2).to_string(index=False),
          scores.round(5).to_string(index=False),sep="\n")
    print("Matplotlib",matplotlib.__version__,"| Feature pipeline checks passed")


if __name__ == "__main__":
    main()

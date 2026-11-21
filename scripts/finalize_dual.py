"""Report both targets from frozen OOF amounts and package the chosen regression."""
import hashlib
import json
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, r2_score, root_mean_squared_error)

from .audit_intake import ROOT, RAW, TARGET
from .prepare_split import load_development
from .spend_bands import LABELS, assign_bands, infer_cutoffs

OUT = ROOT / "reports/generated/dual"
PUBLIC = ROOT / "docs/results"


def score(actual, prediction, truth, bands):
    """Compute numeric and range metrics against original ground truth labels."""
    return dict(RMSE=root_mean_squared_error(actual,prediction),
                MAE=mean_absolute_error(actual,prediction), R2=r2_score(actual,prediction),
                range_accuracy=accuracy_score(truth,bands),
                range_macro_f1=f1_score(truth,bands,labels=LABELS,average="macro",zero_division=0))


def main():
    """Use training-only thresholds for OOF rows and development thresholds for test rows."""
    dev, cv = load_development()
    raw = pd.read_csv(RAW)
    dev_labels = raw.loc[dev.index,"Amount_Spent_Range"]
    bounds = raw.loc[dev.index].groupby("Amount_Spent_Range")[TARGET].agg(["count","min","max"]).loc[LABELS]
    cutoffs = infer_cutoffs(dev[TARGET],dev_labels)
    expected_age = pd.cut(raw.loc[dev.index,"Age"],[-np.inf,19,30,40,50,70,np.inf],
                          labels=["Below 20","20-30","30-40","40-50","50-70","70+"]).astype(str)
    assert np.array_equal(expected_age,raw.loc[dev.index,"Age_Group"])
    OUT.mkdir(parents=True,exist_ok=True)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    oof = pd.read_csv(ROOT/"reports/generated/model_comparison/oof_predictions.csv",index_col="row_position")
    tree = pd.read_csv(ROOT/"reports/generated/selection/tree_oof.csv",index_col="row_position")
    folds = pd.read_csv(ROOT/"reports/generated/selection/outer_folds.csv")
    all_rows=[]
    for row in folds.itertuples():
        train, valid = list(cv.split())[row.fold]
        local_cutoffs = infer_cutoffs(dev.iloc[train][TARGET],dev_labels.iloc[train])
        subset = dev.iloc[valid]
        source = oof if row.method=="step4" else tree
        column = f"{row.model}_original" if row.method=="step4" else f"{row.model}_{row.method}"
        predictions = source.loc[subset.index,column]
        bands = assign_bands(predictions,local_cutoffs)
        all_rows.append(dict(model=row.model,method=row.method,fold=row.fold,runtime=row.seconds,
                             **score(subset[TARGET],predictions,dev_labels.loc[subset.index],bands)))
    fold_results = pd.DataFrame(all_rows)
    mean = fold_results.groupby(["model","method"]).agg(
        RMSE=("RMSE","mean"),RMSE_sd=("RMSE","std"),MAE=("MAE","mean"),R2=("R2","mean"),
        range_accuracy=("range_accuracy","mean"),range_macro_f1=("range_macro_f1","mean"),
        runtime=("runtime","sum")).sort_values("RMSE")
    selection = json.loads((ROOT/"artifacts/configuration.json").read_text())
    assert mean.index[0] == (selection["model"],selection["method"])
    # The same frozen amount predictions from the final evaluation are reused here.
    held = pd.read_csv(ROOT/"reports/generated/holdout/predictions.csv",index_col="row_position")
    truth = raw.loc[held.index,"Amount_Spent_Range"]
    bands = assign_bands(held.prediction,cutoffs)
    final = score(held[TARGET],held.prediction,truth,bands)
    order={v:i for i,v in enumerate(LABELS)}
    final["miss_two_or_more_bands_percent"]=float(np.mean(np.abs(truth.map(order).to_numpy()-
                                                           pd.Series(bands).map(order).to_numpy())>=2)*100)
    final["test_rows"]=len(held)
    final["cutoffs"]=cutoffs
    report=pd.DataFrame(classification_report(truth,bands,labels=LABELS,output_dict=True,zero_division=0)).T
    matrix=confusion_matrix(truth,bands,labels=LABELS)
    fig,ax=plt.subplots(figsize=(8,6),layout="constrained")
    image=ax.imshow(matrix,cmap="Blues")
    ax.set_xticks(range(5),LABELS,rotation=30,ha="right")
    ax.set_yticks(range(5),LABELS)
    ax.set(xlabel="Predicted band",ylabel="Actual band",title="Final holdout spend-range confusion matrix")
    for i,j in np.ndindex(matrix.shape):
        ax.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>matrix.max()/2 else "black")
    fig.colorbar(image,ax=ax,label="Profiles")
    fig.savefig(PUBLIC/"range_confusion.png",dpi=150);plt.close(fig)
    for filename,table in [("comparison",mean),("cv_fold_scores",fold_results),
                           ("per_band",report),("band_bounds_development",bounds),
                           ("confusion",pd.DataFrame(matrix,index=LABELS,columns=LABELS))]:
        table.to_csv(PUBLIC/(filename+".csv"))
    (PUBLIC/"holdout_metrics.json").write_text(json.dumps(final,indent=2))
    for name in ["actual_vs_predicted","residuals"]:
        shutil.copyfile(ROOT/f"reports/generated/holdout/{name}.png",PUBLIC/f"{name}.png")
    for name in ["spend","correlations","segments"]:
        shutil.copyfile(ROOT/f"reports/generated/eda/{name}.png",PUBLIC/f"development_{name}.png")
    schema=json.loads((ROOT/"artifacts/input_schema.json").read_text())
    bundle=dict(pipeline=joblib.load(ROOT/"artifacts/selected_pipeline.pkl"),cutoffs=cutoffs,
                labels=LABELS,schema=schema,configuration=selection,metrics=final,
                versions=json.loads((ROOT/"artifacts/versions.json").read_text()),
                dataset_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest())
    joblib.dump(bundle,ROOT/"artifacts/spend_bundle.pkl")
    held["Predicted_Amount_Spent_Range"]=bands
    held["Actual_Amount_Spent_Range"]=truth
    held.to_csv(OUT/"holdout_predictions.csv")
    print(mean.round(5).to_string())
    print(json.dumps(final,indent=2))
    print(report.round(4).to_string())


if __name__=="__main__":
    main()

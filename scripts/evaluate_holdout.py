"""Evaluate the frozen ShopSmart pipeline and package its diagnostics locally."""
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .audit_intake import PROFILE, RAW, ROOT, TARGET
from .predict import MODEL, inputs
from .prepare_split import OUT as SPLIT, load_development

OUT = ROOT / "reports/generated/holdout"
EXAMPLE = dict(Age=35, Annual_Income=700000, Membership_Type="Silver", City_Tier="Tier 2")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def metrics(actual, predicted):
    """Return errors in original spending units and the coefficient of determination."""
    return dict(rmse=root_mean_squared_error(actual,predicted),
                mae=mean_absolute_error(actual,predicted), r2=r2_score(actual,predicted))


def verify_inference(model):
    """Compare fresh-process single and batch predictions against in-memory inference."""
    batch = pd.DataFrame([EXAMPLE, dict(EXAMPLE, Age=52, City_Tier="Tier 1"),
                          dict(EXAMPLE, Annual_Income=None, Membership_Type="New")])
    source, dest = OUT/"example_inputs.csv", OUT/"example_predictions.csv"
    batch.to_csv(source,index=False)
    expected = model.predict(inputs(batch))
    cmd = [sys.executable,"-m","scripts.predict"]
    single = float(subprocess.check_output(cmd+["--row",json.dumps(EXAMPLE)],cwd=ROOT,text=True))
    subprocess.run(cmd+["--csv",str(source),"--out",str(dest)],cwd=ROOT,check=True)
    actual = pd.read_csv(dest).predicted_spend
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0)
    np.testing.assert_allclose(single,expected[0],atol=1e-8,rtol=0)
    return dict(example=EXAMPLE, prediction=single, checked_batch_rows=len(batch),
                max_absolute_difference=float(max(abs(actual-expected).max(),abs(single-expected[0]))))


def diagnostics(rows):
    """Save paired prediction and residual plots, then summarise signed errors by segment."""
    for name, horizontal, vertical in [("actual_vs_predicted",TARGET,"prediction"),
                                       ("residuals","prediction","residual")]:
        fig, ax = plt.subplots(figsize=(7,5),layout="constrained")
        density = ax.hexbin(rows[horizontal],rows[vertical],gridsize=45,mincnt=1,cmap="viridis",bins="log")
        fig.colorbar(density,ax=ax,label="Records per bin")
        if name == "residuals":
            ax.axhline(0,color="#ad4124",lw=1.5)
        else:
            bounds = [rows[[horizontal,vertical]].min().min(),rows[[horizontal,vertical]].max().max()]
            ax.plot(bounds,bounds,color="#ad4124",lw=1.5)
        ax.set(xlabel=horizontal+" (spending units)",ylabel=vertical+" (spending units)",
               title="Final holdout: "+name.replace("_"," "))
        fig.savefig(OUT/(name+".png"),dpi=150)
        plt.close(fig)
    tables=[]
    for col in ["Membership_Type","City_Tier","spend_band"]:
        result=rows.groupby(col).agg(count=("residual","size"),mae=("absolute_error","mean"),
                                     mean_residual=("residual","mean"),underpredicted=("under","mean"))
        tables.append(result.rename_axis("segment").reset_index().assign(field=col))
    return pd.concat(tables,ignore_index=True)


def main():
    """Score frozen predictions once; reuse the saved scores for all holdout diagnostics."""
    OUT.mkdir(parents=True,exist_ok=True)
    dev,_ = load_development()
    config_path = ROOT/"reports/generated/selection/configuration.json"
    config=json.loads(config_path.read_text())
    assert config["model"]=="Linear" and PROFILE==config["predictors"] and not config["engineered"]
    model=joblib.load(MODEL)
    reload_check=verify_inference(model)
    manifest=pd.read_csv(SPLIT/"manifest.csv",index_col="row_position")
    ids=manifest.index[manifest.partition=="test"]
    assert ids.is_unique and len(ids)==config["split"]["test"] and dev.index.intersection(ids).empty
    assert hashlib.sha256(RAW.read_bytes()).hexdigest()==config["split"]["sha256"]
    rows=pd.read_csv(RAW,usecols=PROFILE+[TARGET]).loc[ids].copy()
    rows["prediction"]=model.predict(inputs(rows))
    rows["residual"]=rows[TARGET]-rows.prediction
    rows["absolute_error"]=rows.residual.abs()
    rows["under"]=rows.residual.gt(0)
    threshold=dev[TARGET].quantile(.9)
    rows["spend_band"]=np.where(rows[TARGET]>=threshold,"High","Other")
    benchmark=dev[TARGET].mean()
    summary=dict(test_rows=len(rows),linear=metrics(rows[TARGET],rows.prediction),
                 benchmark_mean=benchmark,benchmark=metrics(rows[TARGET],np.full(len(rows),benchmark)),
                 high_spend_threshold=threshold, residual_definition="actual minus predicted",
                 model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),reload_check=reload_check)
    rows.to_csv(OUT/"predictions.csv",index_label="row_position")
    groups=diagnostics(rows)
    groups.to_csv(OUT/"segments.csv",index=False)
    prep=model.named_steps["prepare"]
    slopes=pd.Series(model.named_steps["model"].coef_,index=prep.get_feature_names_out())
    scaler=prep.named_transformers_["numbers"].named_steps["scale"]
    effects=dict(income_per_100000=slopes["numbers__Annual_Income"]/scaler.scale_[1]*100000,
                 age_per_year=slopes["numbers__Age"]/scaler.scale_[0],
                 gold_minus_bronze=slopes["categories__Membership_Type_Gold"]-slopes["categories__Membership_Type_Bronze"],
                 tier1_minus_tier3=slopes["categories__City_Tier_Tier 1"]-slopes["categories__City_Tier_Tier 3"])
    summary["conditional_prediction_differences"]=effects
    write_json(OUT/"metrics.json",summary)
    package=MODEL.parent
    shutil.copyfile(config_path,package/"configuration.json")
    shutil.copyfile(OUT/"metrics.json",package/"metrics.json")
    write_json(package/"versions.json",dict(python=platform.python_version(),packages={
        lib:version(lib) for lib in ["pandas","numpy","scikit-learn","scipy","joblib","matplotlib","threadpoolctl"]}))
    write_json(package/"input_schema.json",dict(required=PROFILE,
        fields=dict(Age="numeric, years inferred",Annual_Income="numeric, currency unspecified",
                    Membership_Type=["Bronze","Silver","Gold"],City_Tier=["Tier 1","Tier 2","Tier 3"]),
        numeric_rules="Nonnegative finite values; blank cells use fitted development medians.",
        category_rules="Blank cells use Unknown; unseen labels use the fitted encoder's all-zero category block.",
        engineered_features=[],output="predicted_spend in original spending units",target_period="unconfirmed"))
    print(json.dumps(summary,indent=2))
    print(groups.to_string(index=False))


if __name__=="__main__":
    main()

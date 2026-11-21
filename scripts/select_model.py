"""Nested development-only tree searches and final ShopSmart pipeline selection."""
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .audit_intake import PROFILE, ROOT, TARGET
from .compare_models import OUTPUT as STEP4, candidates
from .prepare_split import OUT as SPLIT, SEED, load_development
from .preprocessing import build_preprocessor

OUT = ROOT / "reports/generated/selection"
ART = ROOT / "artifacts"
GRIDS = {
    "DecisionTree": dict(max_depth=[6, 10], min_samples_split=[2, 100], min_samples_leaf=[5, 30]),
    "RandomForest": dict(n_estimators=[24, 48], max_depth=[8, 14], min_samples_leaf=[5, 25],
                         max_features=[.5, 1.0]),
}


def tree_pipeline(family):
    """Build the default estimator with deterministic randomness and local CPU limits."""
    model = (DecisionTreeRegressor(random_state=SEED) if family == "DecisionTree"
             else RandomForestRegressor(random_state=SEED, n_jobs=2))
    return Pipeline([("prepare", build_preprocessor(scale_numeric=False)), ("model", model)])


def search(family):
    """Search full pipelines using shuffled inner folds within the caller's training rows."""
    grid = {"model__" + k: v for k, v in GRIDS[family].items()}
    return GridSearchCV(tree_pipeline(family), grid, scoring="neg_root_mean_squared_error",
                        cv=KFold(2, shuffle=True, random_state=SEED), n_jobs=1,
                        error_score="raise", refit=True)


def main():
    """Score nested searches on saved folds, select by mean RMSE and refit on development."""
    dev, cv = load_development()
    x, y = dev[PROFILE], dev[TARGET]
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(exist_ok=True)
    prior = pd.read_csv(STEP4 / "fold_scores.csv")
    prior = prior.loc[prior.features.eq("original"), ["model","fold","rmse","seconds"]].assign(method="step4")
    rows, searches = [], []
    oof = dev[[TARGET]].assign(fold=cv.test_fold)
    for fold, (tr, va) in enumerate(cv.split()):
        xt, yt = x.iloc[tr], y.iloc[tr]
        xv, yv = x.iloc[va], y.iloc[va]
        for family in GRIDS:
            for method, estimator in [("default",tree_pipeline(family)), ("tuned",search(family))]:
                started = perf_counter()
                estimator.fit(xt, yt)
                pred = estimator.predict(xv)
                seconds = perf_counter() - started
                oof.loc[xv.index, f"{family}_{method}"] = pred
                rows.append(dict(model=family, method=method, fold=fold,
                                 rmse=root_mean_squared_error(yv,pred), seconds=seconds))
                if method == "tuned":
                    trials = pd.DataFrame(estimator.cv_results_).assign(model=family, fold=fold)
                    trials["inner_rmse"] = -trials.mean_test_score
                    trials["selected"] = trials.index == estimator.best_index_
                    searches.append(trials)
        print(f"Completed outer fold {fold}", flush=True)
    assert oof.notna().all().all()
    folds = pd.concat([prior, pd.DataFrame(rows)], ignore_index=True)
    ranked = folds.groupby(["model","method"]).agg(
        mean_rmse=("rmse","mean"), sd_rmse=("rmse","std"), seconds=("seconds","sum")).sort_values("mean_rmse")
    family, method = ranked.index[0]
    if method == "tuned":
        final_search = search(family).fit(x,y)
        final = final_search.best_estimator_
        pd.DataFrame(final_search.cv_results_).to_csv(OUT/"final_inner_search.csv", index=False)
    elif method == "default":
        final = tree_pipeline(family).fit(x,y)
    else:
        final = Pipeline([("prepare",build_preprocessor(scale_numeric=family in ("Linear","ANN"))),
                          ("model",candidates()[family])]).fit(x,y)
    path = ART / "selected_pipeline.pkl"
    joblib.dump(final,path)
    np.testing.assert_allclose(joblib.load(path).predict(x.head(20)),final.predict(x.head(20)))
    config = dict(model=family, method=method, predictors=PROFILE, engineered=False, target=TARGET,
                  preprocessing=str(final.named_steps["prepare"]), parameters=final.named_steps["model"].get_params(),
                  split=json.loads((SPLIT/"split.json").read_text()), joblib=joblib.__version__,
                  grids=GRIDS, scoring="neg_root_mean_squared_error", inner_folds=2)
    (OUT/"configuration.json").write_text(json.dumps(config,indent=2,default=str))
    pd.concat(searches).to_csv(OUT/"inner_searches.csv",index=False)
    folds.to_csv(OUT/"outer_folds.csv",index=False)
    ranked.to_csv(OUT/"ranking.csv")
    oof.to_csv(OUT/"tree_oof.csv")
    print(ranked.to_string())
    print("Winner:",family,method,"development rows:",len(dev),"reload check passed")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

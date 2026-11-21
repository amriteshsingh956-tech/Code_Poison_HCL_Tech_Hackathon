"""Compare fixed regression candidates on the shared ShopSmart development folds."""
import json
import platform
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .audit_intake import PROFILE, ROOT, TARGET
from .prepare_split import SEED, load_development
from .preprocessing import build_preprocessor

OUTPUT = ROOT / "reports/generated/model_comparison"


def candidates():
    """Return fresh fixed estimators for a single training fold."""
    ann = MLPRegressor(hidden_layer_sizes=(16,), random_state=SEED, early_stopping=True,
                       max_iter=150, n_iter_no_change=10, validation_fraction=.1)
    return {
        "Mean": DummyRegressor(strategy="mean"),
        "Linear": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=50, min_samples_leaf=10,
                                             max_depth=14, random_state=SEED, n_jobs=2),
        "ANN": TransformedTargetRegressor(regressor=ann, transformer=StandardScaler()),
    }


def compare():
    """Generate one out-of-fold prediction per development row and configuration."""
    dev, cv = load_development()
    x, y = dev[PROFILE], dev[TARGET]
    predictions = dev[[TARGET]].copy()
    predictions["validation_fold"] = cv.test_fold
    scores = []
    for fold, (train, valid) in enumerate(cv.split()):
        for engineered in (False, True):
            variant = "log_income" if engineered else "original"
            for name, model in candidates().items():
                prep = build_preprocessor(scale_numeric=name in ("Linear", "ANN"), engineered=engineered)
                pipe = make_pipeline(prep, model)
                started = perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    pipe.fit(x.iloc[train], y.iloc[train])
                pred = pipe.predict(x.iloc[valid])
                elapsed = perf_counter() - started
                key = f"{name}_{variant}"
                if key not in predictions:
                    predictions[key] = np.nan
                ids = x.index[valid]
                assert predictions.loc[ids, key].isna().all() and np.isfinite(pred).all()
                predictions.loc[ids, key] = pred
                row = dict(model=name, features=variant, fold=fold, train_n=len(train), valid_n=len(valid),
                           rmse=root_mean_squared_error(y.iloc[valid], pred),
                           mae=mean_absolute_error(y.iloc[valid], pred), r2=r2_score(y.iloc[valid], pred),
                           seconds=elapsed)
                if name == "ANN":
                    net = model.regressor_
                    row.update(epochs=net.n_iter_, best_internal_r2=net.best_validation_score_,
                               final_loss=net.loss_, reached_cap=net.n_iter_ >= net.max_iter,
                               convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in caught))
                scores.append(row)
        print(f"Completed fold {fold}", flush=True)
    assert predictions.index.is_unique and predictions.notna().all().all()
    return pd.DataFrame(scores), predictions


def main():
    """Save ranked CV means, paired fold differences and local diagnostic data."""
    with threadpool_limits(limits=1):
        scores, predictions = compare()
    summary = scores.groupby(["model","features"]).agg(
        rmse=("rmse","mean"), rmse_sd=("rmse","std"), mae=("mae","mean"),
        r2=("r2","mean"), seconds=("seconds","sum")).sort_values("rmse")
    paired = scores.pivot(index=["model","fold"], columns="features", values="rmse")
    paired["log_minus_original"] = paired.log_income - paired.original
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, table in [("comparison",summary), ("feature_deltas",paired), ("oof_predictions",predictions)]:
        table.to_csv(OUTPUT / f"{name}.csv")
    scores.to_csv(OUTPUT / "fold_scores.csv", index=False)
    (OUTPUT / "versions.json").write_text(json.dumps(dict(python=platform.python_version(),
        pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__, seed=SEED), indent=2))
    print(summary.round(5).to_string())
    print(paired.groupby(level="model").log_minus_original.agg(["mean","min","max"]).round(4))
    print(scores.loc[scores.model=="ANN", ["epochs","best_internal_r2","reached_cap","convergence_warning"]])


if __name__ == "__main__":
    main()

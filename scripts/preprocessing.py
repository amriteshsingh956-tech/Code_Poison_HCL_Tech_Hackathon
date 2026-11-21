"""Shared preprocessing for ShopSmart regression pipelines."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler



DOMAINS = {
    "Membership_Type": ["Bronze", "Silver", "Gold", "Unknown"],
    "City_Tier": ["Tier 1", "Tier 2", "Tier 3", "Unknown"],
    "Preferred_Category": ["Beauty", "Electronics", "Fashion", "Home", "Unknown"],
}


def add_income_feature(frame):
    """Add log income from the current profile, preserving missing income."""
    return frame.assign(Log_Income=np.log1p(frame["Annual_Income"]))


def engineered_names(transformer, input_features):
    """Expose the appended feature name through the fitted pipeline."""
    return [*input_features, "Log_Income"]


def build_preprocessor(*, scale_numeric, include_confirmed_history=False, engineered=False):
    """Return a fresh transformer to place before the regressor inside CV."""
    numbers = ["Age", "Annual_Income"]
    categories = ["Membership_Type", "City_Tier"]
    if include_confirmed_history:
        numbers += ["Website_Visits", "Previous_Purchases"]
        categories += ["Preferred_Category"]
    numeric_steps = [("median", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    def fill_unknown():
        return SimpleImputer(strategy="constant", fill_value="Unknown", missing_values=np.nan)

    categorical = Pipeline([
        ("fill", fill_unknown()),
        ("encode", OneHotEncoder(categories=[DOMAINS[c] for c in categories],
                                 handle_unknown="ignore", sparse_output=False)),
    ])
    branches = [("numbers", Pipeline(numeric_steps), numbers),
                ("categories", categorical, categories)]
    if include_confirmed_history:
        discount = Pipeline([
            ("fill", fill_unknown()),
            ("binary", OrdinalEncoder(categories=[["No", "Yes", "Unknown"]],
                                      handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        branches.append(("discount", discount, ["Discount_Used"]))
    if engineered:
        numbers.append("Log_Income")
    columns = ColumnTransformer(branches, remainder="drop")
    if engineered:
        return Pipeline([
            ("features", FunctionTransformer(add_income_feature, feature_names_out=engineered_names)),
            ("columns", columns),
        ])
    return columns

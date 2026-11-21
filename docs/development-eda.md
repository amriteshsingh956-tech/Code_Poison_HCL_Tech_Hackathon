# Development data and feature choices

Run `python -m scripts.development_eda` from the repository root after Step 2. Install `requirements-eda.txt` in the project environment. Matplotlib renders the figures. The Step 2 loader hashes the raw file for identity and reads partition metadata. Only development records supply values to this analysis. Local plots and aggregate CSVs are in `reports/generated/eda/`.

The 40,000 records contain no missing cells in the nine exported fields. Spend has a median of 17,488.29 and skewness of 0.100. The 1.5-IQR rule flags 41 low and 139 high spend observations. Retain these records in source units. A statistical fence alone does not establish a recording error.

Annual income shows the strongest numeric association with spend (Pearson r = 0.735). Pairwise absolute correlations among the four numeric predictors are below 0.01. Keep the original profile version as the baseline. The optional version adds log(1 + annual income) to let a linear model represent a changing income slope. It is deliberately derived from income and adds no independent information. Its value must be tested later on the same CV folds using RMSE. Use the simpler version if improvement is negligible.

Membership means range from 16,154.65 for Bronze (17,913 rows) to 20,212.13 for Gold (7,237 rows). Keep membership categorical and verify snapshot timing. Group differences describe the supplied synthetic records.

A linear diagnostic predicts each development row from the other four folds using numeric predictors and encoded categories. Four profile fields yield R² 0.74862 and RMSE 1,882.45. Adding all four timing-uncertain fields yields R² 0.95008 and RMSE 838.83. Only 17.79% of these latter predictions fall within 1% of observed spending. The eight-field formula captures much of the synthetic variation but does not recover individual spending exactly. Neither result establishes a generating formula, feature timing, or accuracy on independent customers. Inspect generation rules before relying on the stronger association.

Keep Amount_Spent_Range excluded because it encodes the observed outcome. A new customer's band would require knowing their spending first. Age_Group repeats Age. There is no confirmed prior-period monetary total for historical basket value. Dividing the target by Previous_Purchases would leak the answer, and a zero purchase count would make the ratio undefined. No ratio is computed. If dated historical totals become available, return missing for zero denominators and fit imputation within folds.

For training and inference, use the fitted pipeline containing `build_preprocessor(scale_numeric=..., engineered=True)`. The default `engineered=False` retains Step 2 behavior. Both configurations use the same four primary inputs and fold-local processing. The optional feature uses no target, denominator, or fitted dataset statistic. History remains disabled by default.

# Final ShopSmart holdout evaluation

Run `python -m scripts.evaluate_holdout` from the repository root after model selection. The evaluator loads the fitted Step 5 pipeline, checks the raw fingerprint and disjoint holdout positions, and computes one prediction vector for all holdout diagnostics. Linear Regression uses the four original profile fields and its fitted preprocessing. Feature engineering is disabled in the selected configuration.

On 10,000 reserved synthetic profiles, RMSE was 1,862.11, MAE was 1,484.88 and R² was 0.75367. Predicting the development mean of 17,551.24 for every row gave RMSE 3,751.86, MAE 3,015.56 and R² −0.000006. Units and target period remain unconfirmed.

Residual is actual spending minus predicted spending. Among 1,014 test profiles above or equal to the development 90th percentile of spend (22,407.927), 82.45% had positive residuals averaging 1,562.76 across the whole high-spend group. This outcome-defined diagnostic describes misses among observed high spenders. That group is identifiable only after spending is observed. Membership and city-tier MAEs ranged from 1,464.34 to 1,507.96. See the segment table for counts and signed errors.

For this additive linear model, coefficient contrasts give direct changes in predicted spend holding other inputs fixed. Gold versus Bronze corresponds to +4,054.19, Tier 1 versus Tier 3 to +1,855.88, an additional 100,000 income units to +1,027.91, and one year of age to +34.48. These are predictive associations on different input scales. The analysis supports a synthetic spending prototype; future-month performance requires dated data and confirmed feature windows.

The local artifacts directory holds the fitted pipeline, configuration, input schema, versions and metrics. Detailed outputs under reports/generated/holdout include predictions, segments, figures, and synthetic inference examples. Fresh Python processes verified single and batch results against in-memory predictions with absolute tolerance 1e-8.

Use `from scripts.predict import predict_one, predict_csv` from the repository root. Pass a dictionary to predict_one, or input and output paths to predict_csv. The CLI accepts `python -m scripts.predict --csv reports/generated/holdout/example_inputs.csv --out reports/generated/holdout/batch_demo.csv`. A synthetic profile with age 35, annual income 700000, Silver membership and Tier 2 yields 17,102.72.

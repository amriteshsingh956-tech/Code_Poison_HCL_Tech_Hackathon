# Amount and range prediction

Both outputs are produced from one regression prediction. The original supplied labels remain ground truth. SyntheticDataEDA (1).ipynb shows pandas.qcut with five labels and pandas.cut for age groups. The notebook provides no official monetary cutoffs. Development labels form ordered non-overlapping intervals, and development Age_Group exactly matches its Age-derived bins. Age_Group is excluded as redundant.

Each outer fold infers four cutoffs from training rows alone, using the midpoint between the lower band's maximum and the next band's minimum. An exact boundary belongs to the higher band. Final development cutoffs are 14320.255000000001, 16532.045, 18478.89 and 20733.395. Calculations use full precision. The packaged schema, labels, boundaries, pipeline, configuration and versions reside in local artifacts/spend_bundle.pkl.

The ANN's CV macro-F1 of 0.53393 narrowly exceeds Linear Regression's 0.53391. Linear Regression wins under the agreed lowest RMSE criterion. Final holdout accuracy is 0.53 and macro-F1 is 0.53850077. The two-or-more-band miss rate is 6.68%. See results/comparison.csv, results/per_band.csv, results/confusion.csv and results/holdout_metrics.json for actual results. Confusion matrix rows are actual and columns predicted in label order Very Low, Low, Medium, High, Very High.

The labels were quantile-derived before splitting, so their original definition reflects the supplied population. Source independence, target timing and currency remain unknown. The middle bands have lower recall. Range reporting reuses the frozen holdout amount predictions and does not choose a model from test results.

A synthetic customer aged 35 with income 700000, Silver membership and Tier 2 yields Predicted_Amount_Spent=17102.7173817867 and Predicted_Amount_Spent_Range=Medium. Both clients call the same prediction function. CLI --dual produces both outputs while legacy numeric callers remain supported. Tests cover missing fields, unseen categories, row order, fresh-process reload and values immediately below, on and above all four boundaries.

The teammate notebooks remain preserved as source analysis. Their full-file imputation is historical context. The runnable pipeline fits transformations within training folds. Three CSVs already uploaded on the default branch are removed from tracking in this change and kept locally. Historical Git commits retain those earlier uploads.

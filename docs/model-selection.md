# Step 5 model selection

From the repository root, run `python -m scripts.select_model` after Steps 2 and 4. Step 4 found no improvement from log income, so all candidates use age, income, membership and city tier. The unit is a synthetic profile. Dates and customer IDs are unavailable, and the split assumes exchangeable rows. The loader checks the raw snapshot fingerprint. Development records alone enter fitting and scoring.

Each saved outer fold holds out 8,000 development records. Within the other 32,000, two shuffled inner folds with seed 42 select parameters by negative root mean squared error. Each trial fits the entire preprocessing and tree pipeline. GridSearchCV refits the best inner trial on all outer training rows before predicting the held-out outer fold. Report the mean of five positive outer RMSEs in spending units. RMSE gives larger misses more weight.

| Family | Grid parameter | Values | What changes |
|---|---|---|---|
| Both | max_depth | Tree 6, 10. Forest 8, 14 | Additional levels let the tree distinguish smaller groups of profiles. |
| Tree | min_samples_split | 2, 100 | A larger value requires more training rows in a node before splitting it. |
| Both | min_samples_leaf | Tree 5, 30. Forest 5, 25 | Larger leaves average spend over more customers, smoothing predictions. |
| Forest | n_estimators | 24, 48 | More trees yield a more stable average and take longer to train. |
| Forest | max_features | 0.5, 1.0 | Each split considers half or all of the transformed input features. Sampling fewer features encourages different trees. |

The tree performs 8 combinations × 2 inner folds × 5 outer folds plus 5 refits. The forest performs 16 × 2 × 5 plus 5 refits. Forest fitting uses two workers and grid searches run sequentially. Five library-default fits per family provide a separate reference from Step 4's fixed configurations. Time the searches, refits and predictions together. Step 4 timings come from its earlier run.

The local selection directory under `reports/generated/` contains the inner trial scores, per-fold selections, outer errors and timings, row-aligned tree predictions, ranking and final configuration. The lowest mean development RMSE determines the winner. If a tuned family wins, repeat its inner search on the full development partition. Save the refitted pipeline under `artifacts/selected_pipeline.pkl`.

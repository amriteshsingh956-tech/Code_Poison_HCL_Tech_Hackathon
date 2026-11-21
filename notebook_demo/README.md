# Interactive notebook model demo

This separate app implements the best model recorded in `Untitled5.ipynb`: Linear Regression. The existing four-feature project app remains available at the repository root. This version uses the notebook's eight predictors: Age, Annual_Income, Website_Visits, Previous_Purchases, Membership_Type, Preferred_Category, City_Tier and Discount_Used. Amount_Spent is the output; Amount_Spent_Range and Age_Group are excluded.

## Run locally

Run these commands from the repository root with the existing project environment (on this laptop, `E:/pro pro/.venv/Scripts/python.exe`):

```powershell
python -m pip install -r requirements.txt
python -m notebook_demo.model --source "C:/Users/gr251/OneDrive/Desktop/ShopSmart_synthetic_cleaned (1).csv"
python -m notebook_demo.verify
python -m streamlit run notebook_demo/app.py --server.address 127.0.0.1 --server.port 8530
```

Open http://127.0.0.1:8530. Adjust the eight fields and click **Estimate spend**, or download the CSV template and upload multiple customers. Uploaded CSVs stay in memory; predictions preserve row order and identifier strings. The evaluation tab displays metrics, the mean benchmark, actual-versus-predicted and residual plots, and downloadable metric metadata. R² is not classification accuracy. The app does not produce an individual confidence interval or an unsupported spending-band prediction.

## Actual reproduced results

Seed 42, shuffled 80/20 split: 40,000 training and 10,000 test rows. Linear Regression: RMSE 844.20355; MAE 648.29796; MSE 712679.63333; R² 0.94937054. Training R² is 0.95012027. Predicting the training mean gives test RMSE 3751.86144. These results match the notebook's saved winning scores to numerical tolerance. Other model searches are not rerun: their comparison values in the app are explicitly labeled notebook-recorded results.

The notebook already compared models against the test split. Its winner is used as requested, so this evaluation is a reproduction, not an independent estimate after model selection. It is separate from the earlier four-feature project's scores. Purchase history, visits, discount and preference timing are unconfirmed; their inclusion could encode target-period information. Source data is synthetic, and currency and target horizon are undocumented. Do not label the prediction next-month spend or interpret the higher R² as proof of future forecasting performance.

## Implementation

- pandas reads and validates inputs; NumPy validates finite values and computes RMSE.
- scikit-learn combines training-only numeric median imputation, categorical Unknown imputation, one-hot encoding with a dropped baseline, and Linear Regression. No scaling is needed to reproduce this notebook's linear fit. Unseen categories map to baseline encodings and are flagged in batch uploads.
- joblib stores the complete fitted pipeline locally. Streamlit serves the form, tables and downloads. Matplotlib saves evaluation plots.
- `model.py` shares validation and prediction between the app and verification command. The notebook's full-data `get_dummies` is replaced by a training-fitted encoder to support stable inference without fitting on test data.
- The original CSV is read without modification. Data, model, metrics and row-level results remain in ignored `artifacts/notebook_demo/`. The artifact records dataset SHA-256, dependency versions, seed, metrics and training-derived input ranges/categories.
- `verify.py` checks one-row/batch predictions, identifier preservation, missing values, unseen categories, invalid schemas/types/ranges, fresh-process artifact reload and real Streamlit form parity using AppTest.

Blank batch values use fitted imputers; invalid numbers are rejected. All feature headers are required. Single-input values outside training ranges trigger a warning. Negative model predictions are shown with a warning rather than silently clipped.

This app is local only; no customer data or model artifact should be committed.

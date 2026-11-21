# Step 1 — ShopSmart data intake

The current attachment is named `ShopSmart_synthetic_cleaned (1).csv`. Each row appears to describe one synthetic customer profile and a spending amount. Customer identity, generation method, independence, currency and observation periods are undocumented. The filename alone cannot establish how many independent source customers contributed. The data remains local.

## Reproduce the audit

From the repository root, using Python 3.12.14:

```powershell
python -m pip install -r requirements-intake.txt
python scripts/audit_intake.py --source "C:/Users/gr251/OneDrive/Desktop/ShopSmart_synthetic_cleaned (1).csv"
python scripts/audit_intake.py
```

The first run copies the exact input bytes to the ignored path `data/raw/shopsmart.csv`. Subsequent runs can read that snapshot directly. The script refuses to replace different contents. Pandas reads CSV fields as text before numeric parsing so malformed values can be distinguished from missing entries. NumPy checks for infinite values. The audit uses no random operations.

## Observed data and inferred dictionary

Types and ranges below come from the file. Business meanings are inferred from headers. The proposed prediction time is immediately before the spending period begins. No column has a timestamp proving it was available then.

| Field | Parsed type and observed values | Meaning, availability and decision |
|---|---|---|
| Age | Integer, 18–65 | Age in years inferred. Candidate if recorded by the cutoff. |
| Annual_Income | Numeric, 180,000–1,863,377 | Annual income, currency and reference year unknown. Candidate using the last value available by cutoff. |
| Website_Visits | Numeric, 2–20 | Likely visits in an unspecified window. Hold pending cutoff and definition. 403 fractional entries share the value 10.000686 to six decimals. Ask if this is imputation or an average before treating it as a count. |
| Previous_Purchases | Integer, 0–16 | Likely purchase count. Hold until the aggregation excludes all target-period purchases. |
| Membership_Type | Text, Bronze/Silver/Gold | Membership level. Candidate only for status as of cutoff. Spending during the target period could change membership. |
| Preferred_Category | Text, Beauty/Electronics/Fashion/Home | Stated preference or category derived from purchases. Hold until source and computation window are confirmed. |
| City_Tier | Text, Tier 1/2/3 | City classification. Candidate for residence at cutoff. Tier definitions unknown. |
| Discount_Used | Text, Yes/No | Discount usage flag. Hold until confirmed to refer to prior activity. Redemption during the predicted period would reveal future behavior. |
| Amount_Spent | Numeric, 6,000–34,139.26 | Regression target. Currency, gross/net treatment, and period unknown. Exclude from inputs. |
| Amount_Spent_Range | Text, five ordered spending bands | Exclude. Each band directly identifies an observed interval of the target. Its construction formula is undocumented. |
| Age_Group | Text, five age bands | Exclude as redundant with Age. Observed intervals are 18–19, 20–30, 31–40, 41–50 and 51–65. |

There are no separate historical purchase total, average order value or purchase frequency rate columns. If the provider supplies them later, require their formulas and windows to end before prediction. The current spending total serves as the outcome.

## Audit findings and README assumptions

- The supplied file has 50,000 rows and 11 columns. All fields have zero missing values. All five numeric fields parse with zero invalid or infinite values. No numeric value is negative.
- There are zero full-row duplicates. One additional row repeats an eight-field predictor profile. Without IDs this cannot establish shared customer identity.
- The target has 49,104 distinct values, mean 17,553.05 and median 17,498.34. No row has zero spend. Each spending band contains exactly 10,000 observations. Population prevalence of those bands is unknown.
- The earlier ShopSmart workflow anticipated missing values and eight source predictors. This attachment contains those eight fields, the numeric target and two additional derived-looking bands. It has no customer ID, transaction date, snapshot date or documented target horizon.
- Use the term **spending prediction prototype**. A next-month forecast requires dated evidence that predictors precede the target. The README's proposed random split assumes independent records. Before splitting synthetic records, establish their generation and shared-source lineage. When evaluating on original customer data, reserve independent original records before synthesis. The proposed model comparison and RMSE criterion remain design decisions for later steps.

**Question for the provider:** For each row, what customer or source observation and snapshot date does it represent, what dates and currency does Amount_Spent cover, and which inputs were known before that period began? Please specify the windows and formulas for visits, purchases, category preference and discounts, and whether any use transactions from the target period.

Also request the generator's source dataset and grouping identifiers, and ask how fractional visits and spending bands were constructed.

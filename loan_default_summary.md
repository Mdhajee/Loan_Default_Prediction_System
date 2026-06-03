# Loan Default Prediction System - One Page Summary

## Project goal
Build a classification model that predicts whether a loan applicant is likely to repay (`Y`) or default/not repay (`N`) using applicant, loan, credit history, and property-area information.

## Dataset
The workspace did not include an external CSV, so `loan_data.csv` was generated as a reproducible synthetic dataset with the required columns and realistic missing values. It contains 614 loan applications.

## Chosen model
Best model: **Decision Tree**

Final test accuracy: **0.740**

Final F1-score: **0.830**

## Model comparison
| Model | Accuracy | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.740 | 0.765 | 0.907 | 0.830 |
| Random Forest | 0.715 | 0.752 | 0.884 | 0.813 |
| Logistic Regression | 0.699 | 0.747 | 0.860 | 0.800 |

## EDA insights
- Credit history is the strongest signal: applicants with a recorded positive credit history had a much higher repayment rate than applicants with weak credit history.
- Semiurban and urban property areas showed stronger repayment rates than rural property areas in this dataset.
- Applicant income and coapplicant income are right-skewed, so preprocessing includes median imputation and numeric scaling.
- Loan amount has a moderate relationship with income, but credit history remains more predictive of loan status than income alone.

## Preprocessing summary
`Loan_ID` was removed, missing numeric values were filled with medians, missing categorical values were filled with modes, categorical variables were one-hot encoded, and numeric variables were standardized before model training.

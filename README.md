# Loan Default Prediction System

This project predicts whether a loan applicant is likely to repay or default using a classification workflow.

## Files

- `loan_data.csv` - reproducible synthetic dataset with the assignment schema
- `Loan_Default_Prediction_System.ipynb` - final notebook with preprocessing, EDA, modeling, and evaluation
- `index.html` - optional static HTML export with charts and model results
- `loan_default_summary.md` - one-page project summary
- `app.py` - Streamlit web app for live loan-status prediction
- `generate_project.py` - script used to regenerate all project artifacts

## How to run

Open `Loan_Default_Prediction_System.ipynb` in Jupyter and run all cells. The notebook uses `pandas` and `numpy`; charts are rendered as inline SVG/HTML, and the classifiers are implemented directly for offline compatibility.

## How to host on Streamlit

Push this repository to GitHub, then create a Streamlit Community Cloud app with the main file path set to `app.py`.

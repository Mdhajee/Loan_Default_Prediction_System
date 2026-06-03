from pathlib import Path

import pandas as pd
import streamlit as st

from generate_project import (
    DecisionTreeClassifierScratch,
    RandomForestClassifierScratch,
    classification_metrics,
    predict_logistic_regression,
    prepare_features,
    stratified_train_test_split,
    train_logistic_regression,
)


DATA_PATH = Path(__file__).with_name("loan_data.csv")
NUMERIC_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]


st.set_page_config(page_title="Loan Default Prediction", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_models(df):
    X, y, cleaned_df, scaler = prepare_features(df)
    X_train, X_test, y_train, y_test, feature_names = stratified_train_test_split(
        X, y, test_size=0.20
    )

    logistic_weights = train_logistic_regression(X_train, y_train)
    logistic_pred, _ = predict_logistic_regression(logistic_weights, X_test)

    decision_tree = DecisionTreeClassifierScratch(max_depth=4, min_samples_split=18)
    decision_tree.fit(X_train, y_train)
    tree_pred = decision_tree.predict(X_test)

    random_forest = RandomForestClassifierScratch(
        n_estimators=35, max_depth=5, min_samples_split=16
    )
    random_forest.fit(X_train, y_train)
    forest_pred = random_forest.predict(X_test)

    results = pd.DataFrame(
        [
            {"Model": "Logistic Regression", **classification_metrics(y_test, logistic_pred)},
            {"Model": "Decision Tree", **classification_metrics(y_test, tree_pred)},
            {"Model": "Random Forest", **classification_metrics(y_test, forest_pred)},
        ]
    ).sort_values(["F1-Score", "Accuracy"], ascending=False)

    return {
        "results": results,
        "best_model_name": results.iloc[0]["Model"],
        "models": {
            "Logistic Regression": logistic_weights,
            "Decision Tree": decision_tree,
            "Random Forest": random_forest,
        },
        "feature_names": feature_names,
        "scaler": scaler,
        "X": X,
        "y": y,
    }


def transform_applicant(applicant, feature_names, scaler):
    row = {feature: 0.0 for feature in feature_names}

    for column in NUMERIC_COLUMNS:
        mean = scaler[column]["mean"]
        std = scaler[column]["std"]
        row[column] = (float(applicant[column]) - mean) / std

    for column in CATEGORICAL_COLUMNS:
        encoded_column = f"{column}_{applicant[column]}"
        if encoded_column in row:
            row[encoded_column] = 1.0

    return pd.DataFrame([row], columns=feature_names).to_numpy(dtype=float)


def predict_repayment(model_name, model, applicant_matrix):
    if model_name == "Logistic Regression":
        _, probabilities = predict_logistic_regression(model, applicant_matrix)
        repayment_probability = float(probabilities[0])
    else:
        repayment_probability = float(model.predict_proba(applicant_matrix)[0])

    prediction = "Likely to Repay" if repayment_probability >= 0.5 else "Likely to Default"
    return prediction, repayment_probability


st.title("Loan Default Prediction System")
st.caption("Educational ML demo using the project dataset and the best trained model.")

if not DATA_PATH.exists():
    st.error("loan_data.csv was not found. Add the dataset to the project root.")
    st.stop()

df = load_data()
trained = train_models(df)
results = trained["results"].copy()

best_model_name = trained["best_model_name"]
best_model = trained["models"][best_model_name]
best_row = results.iloc[0]

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Best Model", best_model_name)
metric_col2.metric("Accuracy", f"{best_row['Accuracy']:.3f}")
metric_col3.metric("F1-Score", f"{best_row['F1-Score']:.3f}")

tab_predict, tab_data, tab_models = st.tabs(["Predict", "Dataset", "Model Comparison"])

with tab_predict:
    st.subheader("Applicant Details")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])

        with col2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            loan_term = st.number_input("Loan Amount Term", min_value=12, max_value=600, value=360, step=12)

        with col3:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=1500, step=500)
            loan_amount = st.number_input("Loan Amount", min_value=1, value=150, step=10)

        submitted = st.form_submit_button("Predict Loan Status")

    if submitted:
        applicant = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }
        applicant_matrix = transform_applicant(
            applicant, trained["feature_names"], trained["scaler"]
        )
        prediction, repayment_probability = predict_repayment(
            best_model_name, best_model, applicant_matrix
        )

        st.success(prediction)
        st.metric("Repayment Probability", f"{repayment_probability:.1%}")
        st.progress(repayment_probability)

with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)
    st.subheader("Loan Status Distribution")
    st.bar_chart(df["Loan_Status"].value_counts())

with tab_models:
    st.subheader("Evaluation Results")
    display_results = results[
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "TN", "FP", "FN", "TP"]
    ].copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_results[column] = display_results[column].round(3)
    st.dataframe(display_results, use_container_width=True)

    st.info(
        "This app is for academic demonstration. Validate with real institutional data "
        "before using any model for real lending decisions."
    )

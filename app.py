"""
Diabetes Readmission Predictor - Streamlit App
------------------------------------------------
Mirrors the cleaning / encoding / modeling pipeline from the diabetes.ipynb
notebook (Random Forest vs XGBoost, both trained on SMOTEENN-resampled data),
then exposes:
  1) Evaluation metrics comparison between the two models
  2) A form to enter your own patient data and get a prediction from each model
  3) A button to grab a random row from the dataset and compare predictions
     against the real outcome

Run with:  streamlit run app.py
Make sure "diabetic_data.csv" is in the same folder as this file.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN

# --------------------------------------------------------------------------
# Config / constants (same mappings used in the notebook)
# --------------------------------------------------------------------------
DATA_PATH = "diabetic_data.csv"
CLASS_NAMES = ["NO", ">30", "<30"]

AGE_MAPPING = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
}
A1C_MAPPING = {"None": 0, "Norm": 1, ">7": 2, ">8": 3, "Not Measured": 4}
GLU_MAPPING = {"None": 0, "Norm": 1, ">200": 2, ">300": 3, "Not Measured": 4}
CHANGE_MAPPING = {"No": 0, "Ch": 1}
DIABMED_MAPPING = {"No": 0, "Yes": 1}
MED_MAPPING = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}
MULTI_CLASS_MAPPING = {"NO": 0, ">30": 1, "<30": 2}

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

DISEASE_CATEGORIES = [
    "Diabetes", "Circulatory", "Respiratory", "Digestive",
    "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other",
]

NUMERIC_PASSTHROUGH = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def group_diagnosis(code):
    """Buckets an ICD-9 diagnosis code into one of 9 broad disease groups."""
    code = str(code).upper()
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        num = float(code)
        if 250 <= num < 251:
            return "Diabetes"
        elif (390 <= num <= 459) or num == 785:
            return "Circulatory"
        elif (460 <= num <= 519) or num == 786:
            return "Respiratory"
        elif (520 <= num <= 579) or num == 787:
            return "Digestive"
        elif 800 <= num <= 999:
            return "Injury"
        elif 710 <= num <= 739:
            return "Musculoskeletal"
        elif (580 <= num <= 629) or num == 788:
            return "Genitourinary"
        elif 140 <= num <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        return "Other"


# Raw columns the pipeline needs to see on ANY dataset (training or an
# uploaded file) before it can clean + encode it. 'readmitted' is not
# required -- if it's missing, the app just skips scoring and only predicts.
REQUIRED_RAW_COLUMNS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + MEDICATION_COLUMNS


def clean_and_encode(df_raw, top_10_specialties=None):
    """Runs the notebook's cleaning/encoding pipeline on any raw dataframe
    that has the diabetic_data.csv column layout.

    If top_10_specialties is None, it is computed from this dataframe
    (used once, on the training data). If it's provided, it's reused as-is
    so a newly uploaded dataset is bucketed the exact same way the model
    was trained on. Returns (encoded_df, categorical_options, top_10_specialties,
    n_rows_dropped).
    """
    df = df_raw.copy()
    n_before = len(df)

    df.drop(columns=["encounter_id", "patient_nbr", "weight", "examide", "citoglipton"],
             inplace=True, errors="ignore")
    df = df.replace("?", np.nan)
    df.dropna(subset=["race", "diag_1", "diag_2", "diag_3"], inplace=True)

    df["payer_code"] = df["payer_code"].fillna("not specified")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Measured")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not Measured")
    df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
    df.drop_duplicates(inplace=True)

    n_dropped = n_before - len(df)

    if top_10_specialties is None:
        top_10_specialties = df["medical_specialty"].value_counts().head(10).index.tolist()
    df.loc[~df["medical_specialty"].isin(top_10_specialties), "medical_specialty"] = "Other"

    # Save the raw category options here (before one-hot encoding) so the
    # "enter your own data" form can offer the exact same choices.
    categorical_options = {
        "race": sorted(df["race"].dropna().unique().tolist()),
        "gender": sorted(df["gender"].dropna().unique().tolist()),
        "payer_code": sorted(df["payer_code"].dropna().unique().tolist()),
        "medical_specialty": sorted(df["medical_specialty"].dropna().unique().tolist()),
    }

    for col in MEDICATION_COLUMNS:
        df[col] = df[col].map(MED_MAPPING)

    df["age"] = df["age"].map(AGE_MAPPING)
    df["A1Cresult"] = df["A1Cresult"].map(A1C_MAPPING)
    df["max_glu_serum"] = df["max_glu_serum"].map(GLU_MAPPING)
    df["change"] = df["change"].map(CHANGE_MAPPING)
    df["diabetesMed"] = df["diabetesMed"].map(DIABMED_MAPPING)

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(group_diagnosis)

    if "readmitted" in df.columns:
        df["readmitted"] = df["readmitted"].map(MULTI_CLASS_MAPPING)

    nominal_columns = ["race", "gender", "payer_code", "medical_specialty",
                        "diag_1", "diag_2", "diag_3"]
    df = pd.get_dummies(df, columns=nominal_columns, dtype=int)

    for disease in DISEASE_CATEGORIES:
        cols_to_check = [f"diag_1_{disease}", f"diag_2_{disease}", f"diag_3_{disease}"]
        valid_cols = [c for c in cols_to_check if c in df.columns]
        if valid_cols:
            df[f"has_{disease}"] = df[valid_cols].max(axis=1)
            df.drop(columns=valid_cols, inplace=True)

    return df, categorical_options, top_10_specialties, n_dropped


# --------------------------------------------------------------------------
# Data loading + cleaning (same steps as the notebook)
# --------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df_raw = pd.read_csv(DATA_PATH)
    df, categorical_options, top_10_specialties, _ = clean_and_encode(df_raw)

    numeric_bounds = {
        col: (int(df[col].min()), int(df[col].max()), int(df[col].median()))
        for col in NUMERIC_PASSTHROUGH
    }

    return df, categorical_options, numeric_bounds, top_10_specialties


# --------------------------------------------------------------------------
# Model training (cached as a resource so it only runs once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def train_models(df):
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smoteenn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smoteenn.fit_resample(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_res, y_train_res)

    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train_res, y_train_res)

    rf_pred = rf_model.predict(X_test_scaled)
    xgb_pred = xgb_model.predict(X_test_scaled)

    metrics = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "Accuracy": [accuracy_score(y_test, rf_pred), accuracy_score(y_test, xgb_pred)],
        "F1 (macro)": [f1_score(y_test, rf_pred, average="macro"),
                        f1_score(y_test, xgb_pred, average="macro")],
        "Precision (macro)": [precision_score(y_test, rf_pred, average="macro"),
                               precision_score(y_test, xgb_pred, average="macro")],
        "Recall (macro)": [recall_score(y_test, rf_pred, average="macro"),
                            recall_score(y_test, xgb_pred, average="macro")],
    })

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "scaler": scaler,
        "feature_columns": X.columns.tolist(),
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# Helpers to turn a raw feature row into predictions
# --------------------------------------------------------------------------
def predict_row(raw_row_df, artifacts):
    """raw_row_df: single-row DataFrame with the same (unscaled) columns as
    the training features. Returns predicted class + probabilities for both
    models."""
    row = raw_row_df.reindex(columns=artifacts["feature_columns"], fill_value=0)
    scaled = artifacts["scaler"].transform(row)

    rf_proba = artifacts["rf_model"].predict_proba(scaled)[0]
    xgb_proba = artifacts["xgb_model"].predict_proba(scaled)[0]

    return {
        "rf_class": int(np.argmax(rf_proba)),
        "rf_proba": rf_proba,
        "xgb_class": int(np.argmax(xgb_proba)),
        "xgb_proba": xgb_proba,
    }


def show_prediction_result(result):
    col1, col2 = st.columns(2)
    for col, name, cls_key, proba_key in [
        (col1, "Random Forest", "rf_class", "rf_proba"),
        (col2, "XGBoost", "xgb_class", "xgb_proba"),
    ]:
        with col:
            st.metric(f"{name} prediction", CLASS_NAMES[result[cls_key]])
            proba_df = pd.DataFrame({
                "Outcome": CLASS_NAMES,
                "Probability": result[proba_key],
            }).set_index("Outcome")
            st.bar_chart(proba_df)


# --------------------------------------------------------------------------
# Streamlit layout
# --------------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Readmission Predictor", layout="wide")
st.title("🏥 Diabetes Readmission Predictor")
st.caption("Random Forest vs XGBoost, trained on SMOTEENN-resampled data")

try:
    df, categorical_options, numeric_bounds, top_10_specialties = load_and_clean_data()
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Put the diabetic_data.csv file in the "
        "same folder as app.py and rerun."
    )
    st.stop()

with st.spinner("Training models (only happens once per session)..."):
    artifacts = train_models(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison", "📝 Enter Your Own Data", "🎲 Random Sample",
    "📁 Upload a Dataset",
])

# --------------------------------------------------------------------------
# Tab 1: Model comparison
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Evaluation Metrics")
    st.dataframe(
        artifacts["metrics"].set_index("Model").style.format("{:.3f}"),
        use_container_width=True,
    )

    st.bar_chart(artifacts["metrics"].set_index("Model")[["Accuracy", "F1 (macro)"]])

    st.subheader("Confusion Matrices")
    c1, c2 = st.columns(2)
    for col, name, preds, cmap in [
        (c1, "Random Forest", artifacts["rf_pred"], "Blues"),
        (c2, "XGBoost", artifacts["xgb_pred"], "Purples"),
    ]:
        with col:
            cm = confusion_matrix(artifacts["y_test"], preds)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(name)
            st.pyplot(fig)

# --------------------------------------------------------------------------
# Tab 2: Custom input form
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Enter Patient Data")
    st.caption(
        "Core fields are below. Diagnosis is simplified to a single primary "
        "category, and medications default to 'No' unless expanded."
    )

    with st.form("custom_input_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            race = st.selectbox("Race", categorical_options["race"])
            gender = st.selectbox("Gender", categorical_options["gender"])
            age_bracket = st.selectbox("Age group", list(AGE_MAPPING.keys()), index=6)
            payer_code = st.selectbox("Payer code", categorical_options["payer_code"])
            medical_specialty = st.selectbox(
                "Medical specialty", categorical_options["medical_specialty"]
            )

        with c2:
            time_in_hospital = st.number_input(
                "Time in hospital (days)", *numeric_bounds["time_in_hospital"][:2],
                value=numeric_bounds["time_in_hospital"][2]
            )
            num_lab_procedures = st.number_input(
                "Number of lab procedures", *numeric_bounds["num_lab_procedures"][:2],
                value=numeric_bounds["num_lab_procedures"][2]
            )
            num_procedures = st.number_input(
                "Number of procedures", *numeric_bounds["num_procedures"][:2],
                value=numeric_bounds["num_procedures"][2]
            )
            num_medications = st.number_input(
                "Number of medications", *numeric_bounds["num_medications"][:2],
                value=numeric_bounds["num_medications"][2]
            )
            number_diagnoses = st.number_input(
                "Number of diagnoses", *numeric_bounds["number_diagnoses"][:2],
                value=numeric_bounds["number_diagnoses"][2]
            )

        with c3:
            number_outpatient = st.number_input(
                "Outpatient visits (prior yr)", *numeric_bounds["number_outpatient"][:2],
                value=numeric_bounds["number_outpatient"][2]
            )
            number_emergency = st.number_input(
                "Emergency visits (prior yr)", *numeric_bounds["number_emergency"][:2],
                value=numeric_bounds["number_emergency"][2]
            )
            number_inpatient = st.number_input(
                "Inpatient visits (prior yr)", *numeric_bounds["number_inpatient"][:2],
                value=numeric_bounds["number_inpatient"][2]
            )
            max_glu_serum = st.selectbox("Max glucose serum", list(GLU_MAPPING.keys()), index=0)
            a1c_result = st.selectbox("A1C result", list(A1C_MAPPING.keys()), index=0)

        c4, c5, c6 = st.columns(3)
        with c4:
            change = st.selectbox("Medication changed?", ["No", "Ch"])
        with c5:
            diabetes_med = st.selectbox("On diabetes medication?", ["No", "Yes"])
        with c6:
            primary_diagnosis = st.selectbox("Primary diagnosis category", DISEASE_CATEGORIES)

        with st.expander("Admission details (hospital codes)"):
            a1, a2, a3 = st.columns(3)
            with a1:
                admission_type_id = st.number_input(
                    "Admission type ID", *numeric_bounds["admission_type_id"][:2],
                    value=numeric_bounds["admission_type_id"][2]
                )
            with a2:
                discharge_disposition_id = st.number_input(
                    "Discharge disposition ID", *numeric_bounds["discharge_disposition_id"][:2],
                    value=numeric_bounds["discharge_disposition_id"][2]
                )
            with a3:
                admission_source_id = st.number_input(
                    "Admission source ID", *numeric_bounds["admission_source_id"][:2],
                    value=numeric_bounds["admission_source_id"][2]
                )

        with st.expander("Medication details (optional, defaults to 'No')"):
            med_values = {}
            med_cols = st.columns(3)
            for i, med in enumerate(MEDICATION_COLUMNS):
                with med_cols[i % 3]:
                    med_values[med] = st.selectbox(
                        med, list(MED_MAPPING.keys()), index=0, key=f"med_{med}"
                    )

        submitted = st.form_submit_button("Predict Readmission")

    if submitted:
        row = {col: 0 for col in artifacts["feature_columns"]}
        row["age"] = AGE_MAPPING[age_bracket]
        row["time_in_hospital"] = time_in_hospital
        row["num_lab_procedures"] = num_lab_procedures
        row["num_procedures"] = num_procedures
        row["num_medications"] = num_medications
        row["number_diagnoses"] = number_diagnoses
        row["number_outpatient"] = number_outpatient
        row["number_emergency"] = number_emergency
        row["number_inpatient"] = number_inpatient
        row["admission_type_id"] = admission_type_id
        row["discharge_disposition_id"] = discharge_disposition_id
        row["admission_source_id"] = admission_source_id
        row["max_glu_serum"] = GLU_MAPPING[max_glu_serum]
        row["A1Cresult"] = A1C_MAPPING[a1c_result]
        row["change"] = CHANGE_MAPPING[change]
        row["diabetesMed"] = DIABMED_MAPPING[diabetes_med]
        row[f"has_{primary_diagnosis}"] = 1

        for med, val in med_values.items():
            row[med] = MED_MAPPING[val]

        for prefix, val in [
            ("race", race), ("gender", gender),
            ("payer_code", payer_code), ("medical_specialty", medical_specialty),
        ]:
            colname = f"{prefix}_{val}"
            if colname in row:
                row[colname] = 1

        raw_row_df = pd.DataFrame([row])
        result = predict_row(raw_row_df, artifacts)
        st.divider()
        show_prediction_result(result)

# --------------------------------------------------------------------------
# Tab 3: Random sample from the dataset
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Test a Random Patient From the Dataset")
    if st.button("🎲 Pick a random row"):
        idx = np.random.choice(artifacts["X_test"].index)
        raw_row_df = artifacts["X_test"].loc[[idx]]
        actual = artifacts["y_test"].loc[idx]

        result = predict_row(raw_row_df, artifacts)

        st.write(f"**Actual outcome:** {CLASS_NAMES[actual]}")

        st.divider()
        show_prediction_result(result)


# --------------------------------------------------------------------------
# Tab 4: Upload a whole dataset for batch predictions
# --------------------------------------------------------------------------
def predict_batch(encoded_features_df, artifacts):
    X_batch = encoded_features_df.reindex(columns=artifacts["feature_columns"], fill_value=0)
    scaled = artifacts["scaler"].transform(X_batch)
    rf_pred = artifacts["rf_model"].predict(scaled)
    xgb_pred = artifacts["xgb_model"].predict(scaled)
    rf_proba = artifacts["rf_model"].predict_proba(scaled)
    xgb_proba = artifacts["xgb_model"].predict_proba(scaled)
    return rf_pred, xgb_pred, rf_proba, xgb_proba


with tab4:
    st.subheader("Upload a Dataset for Batch Predictions")
    st.caption(
        "Upload a CSV with the same raw columns as diabetic_data.csv "
        "(same format the models were trained on). If your file also has a "
        "'readmitted' column, the app will score accuracy against it too."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            upload_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that file as a CSV: {e}")
            upload_raw = None

        if upload_raw is not None:
            missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in upload_raw.columns]
            if missing_cols:
                st.error(
                    "This file is missing columns the model needs: "
                    + ", ".join(missing_cols)
                )
            else:
                id_cols = [c for c in ["encounter_id", "patient_nbr"] if c in upload_raw.columns]

                with st.spinner("Cleaning data and generating predictions..."):
                    encoded_df, _, _, n_dropped = clean_and_encode(
                        upload_raw, top_10_specialties=top_10_specialties
                    )
                    has_actual = "readmitted" in encoded_df.columns
                    actual = encoded_df["readmitted"] if has_actual else None
                    features_only = encoded_df.drop(columns=["readmitted"], errors="ignore")

                    rf_pred, xgb_pred, rf_proba, xgb_proba = predict_batch(features_only, artifacts)

                if n_dropped:
                    st.warning(
                        f"{n_dropped} row(s) were dropped during cleaning "
                        "(missing race/diagnosis info, or exact duplicates)."
                    )

                results = pd.DataFrame(index=encoded_df.index)
                for id_col in id_cols:
                    results[id_col] = upload_raw.loc[encoded_df.index, id_col].values
                results["RF Prediction"] = [CLASS_NAMES[p] for p in rf_pred]
                results["XGBoost Prediction"] = [CLASS_NAMES[p] for p in xgb_pred]
                results["RF Confidence"] = rf_proba.max(axis=1).round(3)
                results["XGBoost Confidence"] = xgb_proba.max(axis=1).round(3)
                if has_actual:
                    results["Actual"] = [CLASS_NAMES[a] for a in actual]

                st.success(f"Generated predictions for {len(results)} row(s).")
                st.dataframe(results, use_container_width=True)

                if has_actual:
                    st.subheader("Accuracy on This File")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Random Forest accuracy", f"{accuracy_score(actual, rf_pred):.3f}")
                        st.metric("Random Forest macro F1", f"{f1_score(actual, rf_pred, average='macro'):.3f}")
                    with m2:
                        st.metric("XGBoost accuracy", f"{accuracy_score(actual, xgb_pred):.3f}")
                        st.metric("XGBoost macro F1", f"{f1_score(actual, xgb_pred, average='macro'):.3f}")

                st.subheader("Prediction Distribution")
                label_map = dict(enumerate(CLASS_NAMES))
                dist_df = pd.DataFrame({
                    "Random Forest": pd.Series(rf_pred).map(label_map).value_counts(),
                    "XGBoost": pd.Series(xgb_pred).map(label_map).value_counts(),
                }).reindex(CLASS_NAMES).fillna(0).astype(int)
                st.bar_chart(dist_df)

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=csv_bytes,
                    file_name="readmission_predictions.csv",
                    mime="text/csv",
                )

# ============================================================
# HOW TO RUN THIS PROJECT (Windows / PowerShell)
# ============================================================
# 1. Check Python is installed:
#    python --version
#
# 2. Navigate to the project folder:
#    cd D:\training\project_nti
#
# 3. Install required libraries:
#    pip install -r requirements.txt
#
# 4. Run the Streamlit app:
#    python -m streamlit run app.py
#
# 5. The app will open automatically in your browser at:
#    http://localhost:8501
#    (If it doesn't, open that link manually)
#
# Note: diabetic_data.csv must be in the same folder as app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

st.set_page_config(page_title="Diabetes Readmission Prediction", layout="wide")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LOCAL_CSV = Path(__file__).parent / "diabetic_data.csv"


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def get_raw_data():
    if LOCAL_CSV.exists():
        return load_data(LOCAL_CSV)
    uploaded = st.sidebar.file_uploader("Upload diabetic_data.csv", type="csv")
    if uploaded is not None:
        return load_data(uploaded)
    return None


# ---------------------------------------------------------------------------
# Pipeline: cleaning, encoding, mapping (mirrors the notebook exactly)
# ---------------------------------------------------------------------------

@st.cache_data
def clean_and_encode(df_raw):
    df = df_raw.copy()

    # Drop irrelevant / near-empty / zero-variance columns
    df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'examide', 'citoglipton'], inplace=True)

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    # Drop rows with missing values in these key columns
    df.dropna(subset=['race', 'diag_1', 'diag_2', 'diag_3'], inplace=True)

    # Fill remaining missing values
    df['payer_code'] = df['payer_code'].fillna('not specified')
    df['max_glu_serum'] = df['max_glu_serum'].fillna('Not Measured')
    df['A1Cresult'] = df['A1Cresult'].fillna('Not Measured')
    df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    df.drop_duplicates(inplace=True)

    # Keep top 10 medical specialties, bucket the rest as 'Other'
    top_10_specialties = df['medical_specialty'].value_counts().head(10).index
    df.loc[~df['medical_specialty'].isin(top_10_specialties), 'medical_specialty'] = 'Other'

    # Medication columns mapping
    medication_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone'
    ]
    med_mapping = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}
    for col in medication_columns:
        df[col] = df[col].map(med_mapping)

    # Age mapping
    age_mapping = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
        '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }
    df['age'] = df['age'].map(age_mapping)

    # Lab result mappings
    a1c_mapping = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    glu_mapping = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    df['A1Cresult'] = df['A1Cresult'].map(a1c_mapping)
    df['max_glu_serum'] = df['max_glu_serum'].map(glu_mapping)

    # Change / diabetesMed mapping
    df['change'] = df['change'].map({'No': 0, 'Ch': 1})
    df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1})

    # Diagnosis grouping (ICD-9 buckets)
    def group_diagnosis(code):
        code = str(code).upper()
        if code.startswith('V') or code.startswith('E'):
            return 'Other'
        try:
            num = float(code)
            if 250 <= num < 251:
                return 'Diabetes'
            elif (390 <= num <= 459) or num == 785:
                return 'Circulatory'
            elif (460 <= num <= 519) or num == 786:
                return 'Respiratory'
            elif (520 <= num <= 579) or num == 787:
                return 'Digestive'
            elif 800 <= num <= 999:
                return 'Injury'
            elif 710 <= num <= 739:
                return 'Musculoskeletal'
            elif (580 <= num <= 629) or num == 788:
                return 'Genitourinary'
            elif 140 <= num <= 239:
                return 'Neoplasms'
            else:
                return 'Other'
        except ValueError:
            return 'Other'

    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(group_diagnosis)

    # Target mapping
    df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})

    # One-hot encode nominal columns
    nominal_columns = ['race', 'gender', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
    df = pd.get_dummies(df, columns=nominal_columns, dtype=int)

    # Collapse the 27 diag_*_disease columns into 9 has_disease columns
    disease_categories = ['Diabetes', 'Circulatory', 'Respiratory', 'Digestive',
                           'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
    for disease in disease_categories:
        cols_to_check = [f'diag_1_{disease}', f'diag_2_{disease}', f'diag_3_{disease}']
        valid_cols = [c for c in cols_to_check if c in df.columns]
        if valid_cols:
            df[f'has_{disease}'] = df[valid_cols].max(axis=1)
            df.drop(columns=valid_cols, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Model training (cached so it only runs once per data/session)
# ---------------------------------------------------------------------------

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Random Forest ---
    scaler_rf = MinMaxScaler()
    X_train_scaled = scaler_rf.fit_transform(X_train)
    X_test_scaled = scaler_rf.transform(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    # --- Neural Network ---
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', max_iter=150,
        random_state=42, early_stopping=True
    )
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    scaler_nn = MinMaxScaler()
    X_train_scaled_nn = scaler_nn.fit_transform(X_train_imputed)
    X_test_scaled_nn = scaler_nn.transform(X_test_imputed)

    nn_model.fit(X_train_scaled_nn, y_train)
    nn_pred = nn_model.predict(X_test_scaled_nn)

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "rf_model": rf_model, "rf_pred": rf_pred,
        "nn_model": nn_model, "nn_pred": nn_pred,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Diabetes Readmission Prediction")
st.caption("Random Forest vs. Neural Network on the UCI Diabetes 130-US Hospitals dataset")

raw_df = get_raw_data()

if raw_df is None:
    st.info("Upload `diabetic_data.csv` in the sidebar to run the pipeline "
            "(or place the file next to app.py before deploying).")
    st.stop()

section = st.sidebar.radio(
    "Section",
    [
        "Data & Cleaning",
        "Encoding",
        "Visualization",
        "Train / Test Split",
        "Random Forest",
        "Neural Network",
        "Model Comparison",
    ],
)

df = clean_and_encode(raw_df)
target_names = ['NO', '>30', '<30']

# ---------------------------------------------------------------------------
if section == "Data & Cleaning":
    st.header("Read Data and Handle Missing Values")
    st.write("Raw data preview:")
    st.dataframe(raw_df.head())
    st.write(f"Raw shape: {raw_df.shape}")

    st.write("After dropping irrelevant columns, replacing '?' with NaN, dropping rows "
             "with missing `race`/`diag_1`/`diag_2`/`diag_3`, filling remaining missing "
             "values, and removing duplicates:")
    st.write(f"Cleaned shape (before encoding): "
             f"{raw_df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'examide', 'citoglipton']).replace('?', np.nan).dropna(subset=['race','diag_1','diag_2','diag_3']).drop_duplicates().shape}")

# ---------------------------------------------------------------------------
elif section == "Encoding":
    st.header("Cleaning + Encoding")
    st.write("Medication columns, age, A1C/glucose, change/diabetesMed, and diagnosis "
             "codes are mapped to numeric values; nominal columns are one-hot encoded; "
             "diagnosis dummy columns are collapsed into 9 `has_<disease>` columns.")
    st.dataframe(df.head())
    st.write(f"Final encoded shape: {df.shape}")

# ---------------------------------------------------------------------------
elif section == "Visualization":
    st.header("Visualization")

    st.subheader("Distribution of Readmission Classes")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='readmitted', data=df, order=[0, 1, 2], palette='viridis', ax=ax)
    ax.set_xticklabels(target_names)
    ax.set_title('Distribution of Readmission Classes')
    ax.set_xlabel('Readmitted')
    ax.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Age Group Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x='age', data=df, palette='mako', ax=ax)
    ax.set_title('Age Group Distribution (encoded 0=[0-10) ... 9=[90-100))')
    ax.set_xlabel('Age group (encoded)')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Time in Hospital by Readmission Class")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='readmitted', y='time_in_hospital', data=df, palette='Set2', ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(target_names)
    ax.set_title('Time in Hospital by Readmission Class')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Number of Medications and Diagnoses")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['num_medications'], bins=30, ax=axes[0], color='steelblue')
    axes[0].set_title('Number of Medications')
    sns.histplot(df['number_diagnoses'], bins=16, ax=axes[1], color='indianred')
    axes[1].set_title('Number of Diagnoses')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                     'num_medications', 'number_outpatient', 'number_emergency',
                     'number_inpatient', 'number_diagnoses', 'readmitted']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap (numeric features)')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Readmission Rate by Medication Change")
    ct = pd.crosstab(df['change'], df['readmitted'], normalize='index')
    ct.columns = target_names
    ct.index = ['No Change', 'Changed']
    fig, ax = plt.subplots(figsize=(6, 4))
    ct.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    ax.set_title('Readmission Rate by Medication Change')
    ax.set_ylabel('Proportion')
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
elif section == "Train / Test Split":
    st.header("Train / Test Split")
    results = train_models(df)
    st.write(f"Train shape: {results['X_train'].shape}")
    st.write(f"Test shape: {results['X_test'].shape}")
    st.write("Train class proportions:")
    st.dataframe(results['y_train'].value_counts(normalize=True))

# ---------------------------------------------------------------------------
elif section == "Random Forest":
    st.header("Random Forest")
    results = train_models(df)
    rf_pred, y_test = results['rf_pred'], results['y_test']

    st.metric("Accuracy", f"{accuracy_score(y_test, rf_pred):.4f}")
    st.metric("F1 (macro)", f"{f1_score(y_test, rf_pred, average='macro'):.4f}")

    st.text("Classification Report")
    st.text(classification_report(y_test, rf_pred, target_names=target_names))

    st.subheader("Confusion Matrix")
    cm_rf = confusion_matrix(y_test, rf_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title('Random Forest - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Top 15 Feature Importances")
    importances = pd.Series(
        results['rf_model'].feature_importances_, index=results['X'].columns
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(15).plot(kind='barh', ax=ax)
    ax.invert_yaxis()
    ax.set_title('Top 15 Feature Importances - Random Forest')
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
elif section == "Neural Network":
    st.header("Neural Network")
    results = train_models(df)
    nn_pred, y_test = results['nn_pred'], results['y_test']

    st.metric("Accuracy", f"{accuracy_score(y_test, nn_pred):.4f}")
    st.metric("F1 (macro)", f"{f1_score(y_test, nn_pred, average='macro'):.4f}")

    st.text("Classification Report")
    st.text(classification_report(y_test, nn_pred, target_names=target_names))

    st.subheader("Confusion Matrix")
    cm_nn = confusion_matrix(y_test, nn_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Purples',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title('Neural Network - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Training Loss Curve")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results['nn_model'].loss_curve_)
    ax.set_title('Neural Network Training Loss Curve')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
elif section == "Model Comparison":
    st.header("Compare Results Of Random Forest & Neural Network")
    results = train_models(df)
    rf_pred, nn_pred, y_test = results['rf_pred'], results['nn_pred'], results['y_test']

    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'Neural Network'],
        'Accuracy': [accuracy_score(y_test, rf_pred), accuracy_score(y_test, nn_pred)],
        'F1 (macro)': [f1_score(y_test, rf_pred, average='macro'), f1_score(y_test, nn_pred, average='macro')],
        'Precision (macro)': [precision_score(y_test, rf_pred, average='macro'), precision_score(y_test, nn_pred, average='macro')],
        'Recall (macro)': [recall_score(y_test, rf_pred, average='macro'), recall_score(y_test, nn_pred, average='macro')],
    })
    st.dataframe(results_df)

    fig, ax = plt.subplots(figsize=(7, 4))
    results_df.set_index('Model')[['Accuracy', 'F1 (macro)']].plot(kind='bar', colormap='Set2', ax=ax)
    ax.set_title('Random Forest vs Neural Network')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

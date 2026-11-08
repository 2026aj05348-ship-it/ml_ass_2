import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# Page configuration
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Machine Learning Classification Models - Evaluation")
st.write("Upload your test data (test_data.csv) to evaluate the 5 trained models.")

# Mapping of model names to file paths
model_files = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "SVM": "model/SVM.pkl"
}

# Load models and scaler (cached for performance)
@st.cache_resource
def load_models():
    loaded_models = {}
    for name, path in model_files.items():
        loaded_models[name] = joblib.load(path)
    scaler = joblib.load("model/scaler.pkl")
    return loaded_models, scaler

models, scaler = load_models()

# File uploader
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())
    
    target_col = "Recommendation"
    
    if target_col not in df.columns:
        st.error(f"Error: Target column '{target_col}' not found in the uploaded CSV.")
    else:
        X = df.drop(columns=[target_col])
        y_true = df[target_col]
        
        # Model selection dropdown
        selected_model_name = st.selectbox("Select a Model for Evaluation", list(models.keys()))
        model = models[selected_model_name]
        
        # Apply scaling for LR, KNN, SVM (the data in test_data.csv is already label-encoded)
        if selected_model_name in ["Logistic Regression", "KNN", "SVM"]:
            X_input = scaler.transform(X)
        else:
            X_input = X
            
        # Predictions
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]
        
        # Metrics calculation
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Display metrics
        st.write(f"### Evaluation Metrics: {selected_model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
        # Classification Report
        st.write("### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap="Blues"))
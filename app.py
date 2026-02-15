import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

import xgboost as xgb


# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------
def preprocess_data(data):
    X = data.drop("y", axis=1)
    y = data["y"].map({"yes": 1, "no": 0})

    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    return X, y


# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }


# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.set_page_config(page_title="Term Deposit Prediction Dashboard", layout="wide")

st.title("📊 Term Deposit Prediction Dashboard")
st.markdown("Train and evaluate machine learning models.")

# ----------------------------------------------------
# Upload + Download Side-by-Side
# ----------------------------------------------------
col_upload, col_download = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader(
        "📂 Upload Training Dataset (with 'y')",
        type=["csv"]
    )

with col_download:
    st.markdown("Download sample dataset")
    if os.path.exists("term-deposit.csv"):
        with open("term-deposit.csv", "rb") as f:
            st.download_button(
                label="⬇️ Download Test CSV",
                data=f,
                file_name="term-deposit.csv",
                mime="text/csv"
            )
    else:
        st.info("sample_test.csv not found in repo.")


# ----------------------------------------------------
# Model Selection
# ----------------------------------------------------
model_option = st.selectbox(
    "⚙️ Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

process_button = st.button("🚀 Train & Evaluate")


# ----------------------------------------------------
# Main Logic
# ----------------------------------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file, sep=";")

    # ------------------------------------------------
    # Dataset Preview
    # ------------------------------------------------
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head())

    if process_button:

        X, y = preprocess_data(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Model Selection
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_option == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_option == "Naive Bayes":
            model = GaussianNB()
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_option == "XGBoost":
            model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------
        st.subheader("📈 Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col1.metric("Precision", f"{metrics['Precision']:.4f}")

        col2.metric("Recall", f"{metrics['Recall']:.4f}")
        col2.metric("F1 Score", f"{metrics['F1']:.4f}")

        col3.metric("AUC", f"{metrics['AUC']:.4f}" if metrics['AUC'] else "N/A")
        col3.metric("MCC", f"{metrics['MCC']:.4f}")

        # ------------------------------------------------
        # Confusion Matrix (Half Width)
        # ------------------------------------------------
        st.subheader("Confusion Matrix")

        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            fig, ax = plt.subplots(figsize=(3, 2.5))

            sns.heatmap(
                metrics["Confusion Matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                square=True,
                cbar=True,
                annot_kws={"size": 8},
                ax=ax
            )

            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=8)

            st.pyplot(fig)

        with cm_col2:
            st.empty()

elif uploaded_file is None:
    st.info("Upload dataset to begin.")

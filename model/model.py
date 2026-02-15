import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

# =====================================================
# 1️⃣ Preprocessing
# =====================================================
def preprocess_data(data):

    X = data.drop("y", axis=1)
    y = data["y"].map({"yes": 1, "no": 0})

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    encoder = LabelEncoder()

    for col in cat_cols:
        X[col] = encoder.fit_transform(X[col])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# =====================================================
# 2️⃣ Train Model
# =====================================================
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# =====================================================
# 3️⃣ Evaluate Model
# =====================================================
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


# =====================================================
# Example Usage (Test Block)
# =====================================================
if __name__ == "__main__":

    from sklearn.ensemble import RandomForestClassifier

    # Load dataset
    data = pd.read_csv("bank-full.csv", sep=";")

    # Preprocess
    X, y = preprocess_data(data)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = train_model(model, X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("\nModel Results:\n")
    print(results)

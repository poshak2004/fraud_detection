# src/evaluate.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import seaborn as sns

from src.utils import load_data, preprocess_data

def evaluate():
    # Make sure results folder exists
    os.makedirs("results", exist_ok=True)

    # Load test data
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load trained model
    model = joblib.load("models/fraud_rf.pkl")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n📊 Model Evaluation Metrics:")
    print(f"   Accuracy   : {acc:.4f}")
    print(f"   Precision  : {prec:.4f}")
    print(f"   Recall     : {rec:.4f}")
    print(f"   F1-Score   : {f1:.4f}")
    print(f"   ROC-AUC    : {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("results/roc_curve.png")
    plt.close()

    print("\n✅ Plots saved in 'results/' folder.")

if __name__ == "__main__":
    evaluate()

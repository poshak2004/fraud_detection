# src/evaluate.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.utils import load_data, preprocess_data


def evaluate_model():
    """
    Evaluates the trained fraud detection model and
    saves metrics and visual artifacts to disk.
    """

    # Create result directories
    os.makedirs("results/figures", exist_ok=True)

    # Load data
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Load trained model
    model = joblib.load("models/fraud_rf.pkl")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics (accuracy intentionally excluded)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Save metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Value": [precision, recall, f1, roc_auc]
    })
    metrics_df.to_csv("results/metrics.csv", index=False)

    print("\n📊 Model Evaluation Metrics")
    print(metrics_df.to_string(index=False))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/roc_curve.png")
    plt.close()

    print("\n✅ Evaluation artifacts saved to `results/`")


if __name__ == "__main__":
    evaluate_model()

# src/train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .preprocess import load_and_split_data
import joblib
import os

def train_model():
    """
    Trains a Random Forest model for fraud detection
    and saves the trained model artifact.
    """

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(
        "data/creditcard.csv"
    )

    # Initialize model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Basic evaluation (console)
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_rf.pkl")

    return model, X_test, y_test


if __name__ == "__main__":
    model, _, _ = train_model()
    print("Model trained and saved to models/fraud_rf.pkl")

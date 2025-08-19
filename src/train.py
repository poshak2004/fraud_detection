from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .preprocess import load_and_split_data
import joblib
import os

def train_model():
    X_train, X_test, y_train, y_test = load_and_split_data("data/creditcard.csv")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_rf.pkl")

    return model

if __name__ == "__main__":
    trained_model = train_model()
    print("Model saved to models/fraud_rf.pkl")

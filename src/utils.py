# src/utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Preprocess the dataset:
    - Separate features and target
    - Perform stratified train-test split
    - Scale features using StandardScaler (fit on train only)

    Returns:
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Scale AFTER split to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

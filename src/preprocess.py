# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_and_split_data(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True
):
    """
    Loads the dataset, performs train-test split with stratification,
    applies feature scaling, and optionally handles class imbalance using SMOTE.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset
    test_size : float
        Proportion of test data
    random_state : int
        Random seed for reproducibility
    apply_smote : bool
        Whether to apply SMOTE on training data

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    # Load data
    df = pd.read_csv(filepath)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Feature scaling (important for SMOTE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance using SMOTE (training data only)
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(
            X_train_scaled, y_train
        )

    return X_train_scaled, X_test_scaled, y_train, y_test

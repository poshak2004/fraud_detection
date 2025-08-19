import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Loads dataset from a CSV file"""
    return pd.read_csv(file_path)

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the dataset:
    - Separate features (X) and target (y)
    - Normalize numerical values
    - Split into train and test sets
    """
    X = df.drop("Class", axis=1)  # Features
    y = df["Class"]              # Target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into train and test sets.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        target_column (str): Column to predict.
        test_size (float): Fraction of dataset for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

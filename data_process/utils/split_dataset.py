import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(X: pd.DataFrame, y:pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into train and test sets.

    Args:
        X (pd.DataFrame): Feature Matrix.
        Y (pd.DataFrame): Traget series
        test_size (float): Fraction of dataset for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

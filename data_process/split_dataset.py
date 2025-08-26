import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    s_split: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into train and test sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target series.
        test_size (float, optional): Fraction of dataset for testing. Default = 0.2.
        random_state (int, optional): Seed for reproducibility. Default = 42.
        s_split (bool, optional): If True, perform stratified split (classification tasks).
                                  If False, perform normal split. Default = False.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if s_split else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    return X_train, X_test, y_train, y_test

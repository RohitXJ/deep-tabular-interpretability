import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def handle_imbalance(X: np.ndarray, y: np.ndarray, method: str = "smote") -> tuple[np.ndarray, np.ndarray]:
    """
    Handles class imbalance using resampling techniques.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        method (str): "smote", "undersample", or "oversample".

    Returns:
        tuple: (Balanced X, Balanced y)
    """
    if method == "smote":
        sampler = SMOTE(random_state=42)
    elif method == "oversample":
        sampler = RandomOverSampler(random_state=42)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'smote', 'oversample', or 'undersample'.")

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

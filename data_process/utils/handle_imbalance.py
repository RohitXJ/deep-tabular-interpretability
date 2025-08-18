import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC

def handle_imbalance(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Handles class imbalance using SMOTE (for numeric) or SMOTENC (for categorical).

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.

    Returns:
        tuple: (Balanced X, Balanced y)
    """

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    categorical_idx = [X.columns.get_loc(col) for col in categorical_cols]

    if len(categorical_idx) == 0:
        sampler = SMOTE(random_state=42)
    else:
        sampler = SMOTENC(categorical_features=categorical_idx, random_state=42)

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

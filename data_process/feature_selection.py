import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt

def feature_search(X:pd.DataFrame, y:pd.DataFrame, task_type:str):
    """
    Keeps the important features and even lets the user choose the features they wanna keep

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
        task_type (str): "Classification" or "Regression"

    Returns:
        sorted_cols (list): Sorted feature names by importance
        sorted_scores (list): Corresponding importance scores
    """
    print(f"Estimating feature importance for {task_type}...")

    # For sampling, use Stratified for classification, normal Shuffle for regression
    if task_type == "Classification":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=min(0.3, 500 / len(X)), random_state=42)
        for train_idx, _ in sss.split(X, y):
            sample_X = X.iloc[train_idx].copy()
            sample_y = y.iloc[train_idx].copy()
    elif task_type == "Regression":
        sss = ShuffleSplit(n_splits=1, test_size=min(0.3, 500 / len(X)), random_state=42)
        for train_idx, _ in sss.split(X, y):
            sample_X = X.iloc[train_idx].copy()
            sample_y = y.iloc[train_idx].copy()
    else:
        raise ValueError("task_type must be either 'Classification' or 'Regression'.")

    # Drop low variance features
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    # Drop keyword-based columns
    keywords = ["id", "name", "serial", "timestamp", "date", "uuid"]
    for col in sample_X.columns:
        if any(key in col.lower() for key in keywords):
            print(f"Dropping non-informative column: {col}")
            sample_X.drop(columns=[col], inplace=True)

    # Fit model for feature importance
    if task_type == "Classification":
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        model.fit(sample_X, sample_y)
        importances = np.abs(model.coef_[0])
    else:  # Regression
        model = LinearRegression()
        model.fit(sample_X, sample_y)
        importances = np.abs(model.coef_)

    sorted_idx = np.argsort(importances)[::-1]
    sorted_cols = sample_X.columns[sorted_idx]
    sorted_scores = importances[sorted_idx]

    return sorted_cols, sorted_scores


def feature_selection(X: pd.DataFrame, top_n_features: str, sorted_cols, sorted_scores):
    if top_n_features.isdigit():
        top_n_features = int(top_n_features)
        X = X[sorted_cols[:top_n_features]]
        print(f"Using top {top_n_features} user-selected features.")

    elif top_n_features.strip().lower() == 'auto':
        scores = np.array(sorted_scores, dtype=float)

        # Always keep features >= 20% of max importance
        max_score = scores.max()
        high_keep = scores >= 0.2 * max_score

        # Also keep features >= median (if not near zero)
        median_score = np.median(scores)
        medium_keep = scores >= max(median_score, 0.01 * max_score)

        # Combine conditions
        keep_mask = high_keep | medium_keep
        keep_indices = np.where(keep_mask)[0]

        # Fallback: if still nothing, keep top 1
        if len(keep_indices) == 0:
            keep_indices = [0]

        X = X[[sorted_cols[i] for i in keep_indices]]
        print(f"'Auto' selected top {len(keep_indices)} features (robust bands method).")

    else:
        print("Invalid input. No feature filtering applied.")

    return list(X.columns)



def imp_plot(columns, scores):
    plt.figure(figsize=(10, 5))
    plt.barh(columns[::-1], scores[::-1], color="teal")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

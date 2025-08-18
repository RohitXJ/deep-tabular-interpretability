import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def feature_selection(X:pd.DataFrame,y:pd.DataFrame) -> list:
    """
    Keeps the important features and even lets the user choose the features they wanna keep

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target vector.
    Returns:
        img: A plot of important features
        pd.DataFrame: Dataset with important features only.
    """
    print("Estimating feature importance from balanced sample...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=min(0.3, 500 / len(X)), random_state=42)
    for train_idx, _ in sss.split(X, y):
        sample_X = X.iloc[train_idx].copy()
        sample_y = y.iloc[train_idx].copy()

    # Drop low variance and unnecessary features
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

    # Drop keyword-based columns
    keywords = ["id", "name", "serial", "timestamp", "date", "uuid"]
    for col in sample_X.columns:
        if any(key in col.lower() for key in keywords):
            print(f"Dropping non-informative column: {col}")
            sample_X.drop(columns=[col], inplace=True)

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(sample_X, sample_y)

    importances = np.abs(model.coef_[0])
    sorted_idx = np.argsort(importances)[::-1]
    sorted_cols = sample_X.columns[sorted_idx]
    sorted_scores = importances[sorted_idx]

    imp_plot(sorted_cols, sorted_scores)

    # Get user input
    top_n_features = input("Select number of features to keep based on importance scores,\nor enter 'auto' for automatic selection:\n")

    # Feature selection logic
    if top_n_features.isdigit():
        top_n_features = int(top_n_features)
        X = X[sorted_cols[:top_n_features]]
        print(f"Using top {top_n_features} user-selected features.")
    elif top_n_features.strip().lower() == 'auto':
        diffs = np.diff(sorted_scores)
        elbow = np.argmax(diffs < 0.01) + 1
        X = X[sorted_cols[:elbow]]
        print(f"'Auto' selected top {elbow} features using elbow method.")
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
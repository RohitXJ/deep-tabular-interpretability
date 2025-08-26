import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
import smogn  # for regression


def handle_imbalance(X: pd.DataFrame, y: pd.Series, task_type: str):
    """
    Handle class/target imbalance for classification and regression tasks.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        task_type (str): Either 'Classification' or 'Regression'

    Returns:
        X_res (pd.DataFrame), y_res (pd.Series): Resampled data
    """
    if task_type.lower() == "classification":
        # Handle categorical features properly for SMOTENC
        if any(dtype == 'object' or str(dtype).startswith('category') for dtype in X.dtypes):
            cat_indices = [
                i for i, dtype in enumerate(X.dtypes)
                if dtype == 'object' or str(dtype).startswith('category')
            ]
            sampler = SMOTENC(categorical_features=cat_indices, random_state=42)
        else:
            sampler = SMOTE(random_state=42)

        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)

    elif task_type.lower() == "regression":
        # smogn requires a clean DataFrame with target included
        data = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

        if y.name is None:
            target = "target"
            data.rename(columns={data.columns[-1]: target}, inplace=True)
        else:
            target = y.name

        try:
            data_res = smogn.smoter(
                data,
                y=target,
                k=5,          # neighbors
                pert=0.02,    # small noise
                rel_thres=0.8 # rare threshold
            )
        except Exception as e:
            print(f"[WARN] smogn failed with error: {e}")
            print("Falling back to original dataset (no resampling).")
            return X, y

        # Ensure valid output
        if data_res is None or target not in data_res:
            print("[WARN] smogn returned None or invalid data. Returning original dataset.")
            return X, y

        X_res = data_res.drop(columns=[target])
        y_res = data_res[target]
        return X_res, y_res

    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

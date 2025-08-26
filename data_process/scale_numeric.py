from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

def scale_numeric(df: pd.DataFrame, target_col: str, domain_name: str, model_name: str, prediction_type: str,scaler_type="standard")-> tuple[pd.DataFrame, object]:
    """
    Scale data with rules:
    - Numerical feature columns (excluding target) are always scaled.
    - Target column is scaled ONLY if:
        (domain_name == "DL") OR (model_name == "SVM" and prediction_type == "Regression")
    """

    if scaler_type == "standard":
        scaler = StandardScaler() 
    else :
        scaler = MinMaxScaler()
    df_scaled = df.copy()

    # Always scale numerical feature columns (excluding target)
    num_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if num_cols:  # scale only if there are numeric columns
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

    # Conditionally scale target column with try-except
    if (domain_name == "DL") or (model_name == "SVM" and prediction_type == "Regression"):
        try:
            df_scaled[target_col] = scaler.fit_transform(df_scaled[[target_col]])
        except KeyError:
            raise ValueError("Value missing Error: Target column not found in dataframe")

    return (df_scaled, scaler)
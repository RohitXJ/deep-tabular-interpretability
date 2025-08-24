from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

def scale_numeric(df:pd.DataFrame,target_col:str, scaler_type="standard")-> tuple[pd.DataFrame, object]:
    """
    Scales numeric columns.

    Args:
        df (pd.DataFrame): Input dataset.
        scaler_type (str): "standard" or "minmax".
    Returns:
        tuple: (Scaled DataFrame, fitted scaler object)
    """
    target_df = df[target_col]
    if scaler_type == "standard":
        scaler = StandardScaler() 
    else :
        scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    df_scaled[target_col]=target_df
    return (df_scaled, scaler)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

def scale_numeric(df, scaler_type="standard")-> tuple[pd.DataFrame, object]:
    """
    Scales numeric columns.

    Args:
        df (pd.DataFrame): Input dataset.
        scaler_type (str): "standard" or "minmax".
    Returns:
        tuple: (Scaled DataFrame, fitted scaler object)
    """
    if scaler_type == "standard":
        scaler = StandardScaler() 
    else :
        scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    return (df_scaled, scaler)

data = pd.read_csv(r"C:\Users\RJ\Downloads\archive (3)\tested.csv") 
print(data)
print(scale_numeric(data))

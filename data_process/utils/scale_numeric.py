from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

def scale_numeric(df, scaler_type="standard"):
    if scaler_type == "standard":
        scaler = StandardScaler() 
    else :
        scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    return df_scaled, scaler

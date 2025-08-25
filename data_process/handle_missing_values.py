import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_values(df: pd.DataFrame, target_col:str, strategy_num: str = "mean", strategy_cat: str = "most_frequent") -> pd.DataFrame:
    """
    Fills missing values for numeric and categorical columns.

    Args:
        df (pd.DataFrame): Input dataset.
        strategy_num (str): Strategy for numeric columns ("mean", "median").
        strategy_cat (str): Strategy for categorical columns ("most_frequent", "constant").
    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
    try:
        total_length = df.shape
        df = df.dropna(subset=[target_col])

        for col in df.columns:
            col_miss_total = df[col].isnull().sum().sum()
            miss_percent = (col_miss_total/total_length[0])*100
            if miss_percent > 40.0:
                df = df.drop(columns=[col],axis="columns")

        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype.kind in 'iufc':
                    if strategy_num == "mean":
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].median())
                elif isinstance(df[col].dtype, pd.CategoricalDtype):
                    if strategy_cat == "most_frequent":
                        imputer_most_frequent = SimpleImputer(strategy='most_frequent')
                        df[col] = pd.DataFrame(imputer_most_frequent.fit_transform(df[[col]]))
                    else:
                        imputer_most_frequent = SimpleImputer(strategy='constant',fill_value='Unknown')
                        df[col] = pd.DataFrame(imputer_most_frequent.fit_transform(df[[col]]))
                    
        print("Null Values Handled")
        return df
    except Exception as e:
        print(f"{e}")
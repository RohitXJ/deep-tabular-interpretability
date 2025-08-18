import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV dataset from the given file path.

    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Raw dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path,index_col=False)
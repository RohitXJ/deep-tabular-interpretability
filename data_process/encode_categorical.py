import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, encoding_type="onehot")-> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical columns into numeric form.

    Args:
        df (pd.DataFrame): Input dataset.
        encoding_type (str): "onehot" or "label".
    Returns:
        tuple: (Encoded DataFrame, mapping dictionary for label encoding)
    """
    df_copy = df.copy()
    cat_cols = df_copy.select_dtypes(include=["object", "category"]).columns

    if encoding_type == "onehot":
        return (pd.get_dummies(df_copy, columns=cat_cols,drop_first=True), {})

    elif encoding_type == "label":
        mapping = {}
        for col in cat_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        return (df_copy, mapping)

    else:
        raise ValueError("encoding_type must be 'onehot' or 'label'")
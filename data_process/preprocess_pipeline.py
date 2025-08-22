from data_process.utils import encode_categorical,feature_selection,handle_imbalance,handle_missing_values,load_dataset,scale_numeric,split_dataset
from data_process.utils import feature_search,imp_plot
import pandas as pd

def data_pipeline(file_path:str):
    """
    Main data preprocessing pipeline that loads the dataset, handles missing values,
    encodes categorical variables, scales numeric features, performs feature selection,
    and handles class imbalance.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        list: Processed training data ready for model training. X_train, X_test, y_train, y_test format.
    """
    df = load_dataset(file_path)
    col_dict = {}
    i = 1
    for cols in df.columns:
        col_dict[i]=cols
        i += 1

    print("These are the column names found from your dataset")
    for keys in col_dict.keys():
        print(f"{keys}. {col_dict[keys]}")
    target_col = int(input(f"Enter the option number of the target column -> "))
    target_col = col_dict[target_col]
    print(target_col)

    # Phase 1: Temporary encode/scale for feature selection
    df = handle_missing_values(df)

    df_copy = df.copy()

    df_copy,_ = scale_numeric(df_copy,target_col)

    df_copy,_ = encode_categorical(df_copy,encoding_type="label")
    X=df_copy.drop(columns=[target_col],axis="columns")
    y=df_copy[target_col]
    sorted_cols,sorted_scores = feature_search(X,y)

    imp_plot(sorted_cols,sorted_scores)

    top_n_features = input("Select number of features to keep based on importance scores,\nor enter 'auto' for automatic selection:\n")

    extracted_features = feature_selection(X,top_n_features,sorted_cols,sorted_scores)
    extracted_features.append(target_col)

    df = pd.DataFrame(df[extracted_features])
    del(df_copy,X,y)
    
    # Phase 2: Final encode/scale for actual model training
    df,scaler = scale_numeric(df,target_col)

    df,encoders = encode_categorical(df,encoding_type="label")

    X = df.drop(columns=[target_col],axis="columns")
    y = df[target_col]

    X,y = handle_imbalance(X,y)

    train_ready_data = split_dataset(X,y,test_size=0.3)
    return train_ready_data

#print(data_pipeline(r"Test_data\tested.csv"))
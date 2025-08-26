import warnings
warnings.filterwarnings("ignore")

from data_process import (
    encode_categorical,
    feature_selection,
    feature_search,
    handle_imbalance,
    handle_missing_values,
    imp_plot,
    load_dataset,
    scale_numeric,
    split_dataset,
)
from model_hub import ML_model_eval, ML_models_call, ML_model_train

import pandas as pd

domain_type = {
    1:{
        "ML":{
            "Classification":{
                1:"Logistic Regression",2:"SVM",3:"Random Forest Classifier",4:"XGBoost",5:"LightGBM",6:"CatBoost"
                },
            "Regression":{
                1:"Linear Regression",2:"Ridge",3:"Lasso",4:"Random Forest Regressor",5:"XGBoost",6:"LightGBM",7:"CatBoost"
                }
            }
        },
    2:{
        "DL": {}
        }
}

if __name__ == "__main__":
    data_path = r"Test_data\tested.csv"  #Default Input
    #######---------Options Input---------#######

    print("\nChoose your domain (Enter the option number):")
    for key, value in domain_type.items():
        print(f"{key}. {list(value.keys())[0]}")
    try:
        domain_input = input("Enter option number: ")
        domain_name = list(domain_type[int(domain_input)].keys())[0]
    except (KeyError, ValueError):
        print("Invalid option.")
        exit()

    if domain_name == "ML":
        print("\nChoose your prediction type (Enter the option number):")
        for key, value in domain_type[int(domain_input)]["ML"].items():
            print(f"{list(domain_type[1]['ML'].keys()).index(key) + 1}. {key}")
        try:
            type_input = input("Enter option number: ")
            prediction_type_dict = domain_type[int(domain_input)]["ML"][list(domain_type[1]["ML"].keys())[int(type_input)-1]]
            prediction_type = list(domain_type[1]["ML"].keys())[int(type_input)-1]
        except (KeyError, ValueError, IndexError):
            print("Invalid option.")
            exit()

        print(f"\nChoose your {prediction_type} model (Enter the option number):")
        models_dict = prediction_type_dict
        for key, value in models_dict.items():
            print(f"{key}. {value}")
        try:
            model_input = input("Enter option number: ")
            model_name = models_dict[int(model_input)]
            print(f"You have selected: {model_name}")
        except (KeyError, ValueError):
            print("Invalid model option.")

    elif domain_name == "DL":
        print("DL options will be added later.")
    else:
        raise ValueError("Wrong Domain Chosen!")

    print(f"domain name {domain_name} prediction type {prediction_type} model name {model_name}")
    pass
    #######---------Data Processing Starts---------#######
    
    df = load_dataset(data_path)
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
    df = handle_missing_values(df,target_col=target_col)
    df_copy = df.copy()
    df_copy,_ = scale_numeric(df_copy,target_col,domain_name,model_name,prediction_type)
    df_copy,_ = encode_categorical(df_copy,encoding_type="label")
    X=df_copy.drop(columns=[target_col],axis="columns")
    y=df_copy[target_col]
    sorted_cols,sorted_scores = feature_search(X,y,task_type=prediction_type)
    imp_plot(sorted_cols,sorted_scores)
    top_n_features = input("Select number of features to keep based on importance scores,\nor enter 'auto' for automatic selection:\n")
    extracted_features = feature_selection(X,top_n_features,sorted_cols,sorted_scores)
    extracted_features.append(target_col)
    df = pd.DataFrame(df[extracted_features])
    del(df_copy,X,y)
    
    # Phase 2: Final encode/scale for actual model training
    df,scaler = scale_numeric(df,target_col,domain_name,model_name,prediction_type)
    df,encoders = encode_categorical(df,encoding_type="label")
    X = df.drop(columns=[target_col],axis="columns")
    y = df[target_col]
    X,y = handle_imbalance(X,y,task_type=prediction_type)
    train_ready_data = split_dataset(X,y,test_size=0.3)


    #######---------Model Pipeline Starts---------#######

    if domain_name == "ML":
        RAW_model = ML_models_call(type=prediction_type,model=model_name)

        trained_model = ML_model_train(model=RAW_model,data=[train_ready_data[0],train_ready_data[2]])

        ML_model_eval(model=trained_model,test_data=[train_ready_data[1],train_ready_data[3]],type=prediction_type)
    else:
        print("DL part not yet added")
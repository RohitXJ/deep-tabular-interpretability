from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import torch
import torch.nn as nn
import rtdl
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor


def ML_models_call(type: str, model: str):
    """ type is the work type, classification or regression it will be 
    model will be the model names as they are in the end of the todo list
    """
    if type == "Regression":
        if model == "Linear Regression":
            M_obj = LinearRegression()
        elif model == "Ridge":
            M_obj = Ridge()
        elif model == "Lasso":
            M_obj = Lasso()
        elif model == "Random Forest Regressor":
            M_obj = RandomForestRegressor()
        elif model == "XGBoost":
            M_obj = XGBRegressor()
        elif model == "LightGBM":
            M_obj = LGBMRegressor()
        elif model == "CatBoost":
            M_obj = CatBoostRegressor()
        else:
            raise ValueError("Wrong choice of models.")
    elif type == "Classification":
        if model == "Logistic Regression":
            M_obj = LogisticRegression()
        elif model == "SVM":
            M_obj = SVC()
        elif model == "Random Forest Classifier":
            M_obj = RandomForestClassifier()
        elif model == "XGBoost":
            M_obj = XGBClassifier(verbosity=0)
        elif model == "LightGBM":
            M_obj = LGBMClassifier(verbose=-1)
        elif model == "CatBoost":
            M_obj = CatBoostClassifier(verbose=0)
        else:
            raise ValueError("Wrong choice of models.")
    else:
        raise ValueError("Wrong type choice.")
    return M_obj


def DL_models_call(
    model: str,
    task_type: str,
    input_dim: int,
    output_dim: int,
    cat_cardinalities: list = None,
    n_num_features: int = None,
):
    """
    This function initializes a Deep Learning model for tabular data.

    Parameters:
    - model (str): The name of the model to initialize.
      Options: "FNN", "TabNet", "TabTransformer", "NODE", "FT-Transformer".
    - task_type (str): The type of task.
      Options: "classification", "regression".
    - input_dim (int): The number of input features.
    - output_dim (int): The number of output features.
    - cat_cardinalities (list): A list of cardinalities of categorical features.
      Required for TabTransformer and FT-Transformer.
    - n_num_features (int): The number of numerical features.
      Required for TabTransformer and FT-Transformer.
    """
    # Type 1 Parameters (Hardcoded Defaults)
    hidden_dims = [128, 128]
    dropout = 0.1
    n_d = 8
    n_a = 8
    n_steps = 3
    gamma = 1.3
    n_blocks = 3
    d_token = 192
    attention_dropout = 0.2
    ffn_dropout = 0.1
    n_layers = 1
    layer_dim = 128
    num_trees = 2048
    depth = 6
    tree_dim = 3

    if model == "FNN":
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        if task_type == 'classification':
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
        M_obj = nn.Sequential(*layers)
    elif model == "TabNet":
        if task_type == 'classification':
            M_obj = TabNetClassifier(
                n_d=n_d,
                n_a=n_a,
                n_steps=n_steps,
                gamma=gamma,
                input_dim=input_dim,
                output_dim=output_dim,
            )
        else:
            M_obj = TabNetRegressor(
                n_d=n_d,
                n_a=n_a,
                n_steps=n_steps,
                gamma=gamma,
                input_dim=input_dim,
                output_dim=output_dim,
            )
    elif model == "TabTransformer":
        if cat_cardinalities is None or n_num_features is None:
            raise ValueError("cat_cardinalities and n_num_features are required for TabTransformer")
        M_obj = rtdl.TabTransformer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            n_blocks=n_blocks,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            d_out=output_dim,
        )
    elif model == "NODE":
        M_obj = rtdl.NODE(
            d_in=input_dim,
            n_layers=n_layers,
            layer_dim=layer_dim,
            num_trees=num_trees,
            depth=depth,
            tree_dim=tree_dim,
            d_out=output_dim,
        )
    elif model == "FT-Transformer":
        if cat_cardinalities is None or n_num_features is None:
            raise ValueError("cat_cardinalities and n_num_features are required for FT-Transformer")
        M_obj = rtdl.FTTransformer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            n_blocks=n_blocks,
            d_token=d_token,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            d_out=output_dim,
        )
    else:
        raise ValueError("Wrong choice of models.")
    return M_obj
import sys
sys.path.insert(0, './node')
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import torch
import torch.nn as nn

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from tab_transformer_pytorch import TabTransformer


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
    cat_idxs: list = None,
    cat_cardinalities: list = None,
    hyperparameters: dict = {},
    device: str = 'cpu'
):
    """
    This function initializes a Deep Learning model for tabular data.

    Parameters:
    - model (str): The name of the model to initialize.
      Options: "FNN", "TabNet", "NODE".
    - task_type (str): The type of task.
      Options: "classification", "regression".
    - input_dim (int): The number of input features.
    - output_dim (int): The number of output features.
    - cat_idxs (list): A list of indices of categorical features.
    - cat_cardinalities (list): A list of cardinalities of categorical features.
    - n_num_features (int): The number of numerical features.
    - hyperparameters (dict): A dictionary of hyperparameters for the model.
    - device (str): The device to run the model on, 'cpu' or 'cuda'.
    """
    # Get hyperparameters with defaults
    hidden_dims = hyperparameters.get('hidden_dims', [128, 128])
    dropout = hyperparameters.get('dropout', 0.1)
    n_d = hyperparameters.get('n_d', 8)
    n_a = hyperparameters.get('n_a', 8)
    n_steps = hyperparameters.get('n_steps', 3)
    gamma = hyperparameters.get('gamma', 1.3)
    n_blocks = hyperparameters.get('n_blocks', 3)
    d_token = hyperparameters.get('d_token', 192)
    attention_dropout = hyperparameters.get('attention_dropout', 0.2)
    ffn_dropout = hyperparameters.get('ffn_dropout', 0.1)
    n_layers = hyperparameters.get('n_layers', 1)
    layer_dim = hyperparameters.get('layer_dim', 128)
    num_trees = hyperparameters.get('num_trees', 2048)
    depth = hyperparameters.get('depth', 6)
    tree_dim = hyperparameters.get('tree_dim', 3)

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
                cat_idxs=cat_idxs,
                cat_dims=cat_cardinalities,
                device_name=device
            )
        else:
            M_obj = TabNetRegressor(
                n_d=n_d,
                n_a=n_a,
                n_steps=n_steps,
                gamma=gamma,
                cat_idxs=cat_idxs,
                cat_dims=cat_cardinalities,
                device_name=device
            )
    elif model == "TabTransformer":
        # TabTransformer expects categorical features to be passed separately
        # cat_cardinalities is a list of cardinalities for each categorical feature
        # cat_idxs is a list of indices for each categorical feature
        # input_dim is the total number of features (numerical + categorical)
        # output_dim is the number of output classes/regression target dimension

        # The TabTransformer library expects a list of (cardinality, embedding_dim) for categorical features.
        # For simplicity, we'll use a default embedding_dim for all categorical features.
        # A common choice is to use min(50, cardinality // 2) or a fixed small number like 4 or 8.
        # Let's use a fixed embedding dimension of 8 for now.
        cat_dims_with_emb = [(card, 8) for card in cat_cardinalities]

        M_obj = TabTransformer(
            categories=cat_dims_with_emb,
            num_continuous=input_dim - len(cat_idxs), # Total features - number of categorical features
            dim=hyperparameters.get('dim', 32),
            depth=hyperparameters.get('depth', 6),
            heads=hyperparameters.get('heads', 8),
            attn_dropout=hyperparameters.get('attn_dropout', 0.1),
            ff_dropout=hyperparameters.get('ff_dropout', 0.1),
            mlp_hidden_mults=(4, 2), # Default from library
            mlp_act=nn.ReLU(), # Default from library
            dim_out=output_dim if task_type == 'Classification' else 1 # For regression, dim_out is 1
        )
    else:
        raise ValueError("Wrong choice of models.")
    return M_obj
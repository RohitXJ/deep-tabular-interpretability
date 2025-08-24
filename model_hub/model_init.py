from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

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
            M_obj = XGBClassifier()
        elif model == "LightGBM":
            M_obj = LGBMClassifier()
        elif model == "CatBoost":
            M_obj = CatBoostClassifier()
        else:
            raise ValueError("Wrong choice of models.")
    else:
        raise ValueError("Wrong type choice.")
    return M_obj

def DL_models_call():
    pass
    """
    No one touches this one, Assigned to Rohit due to involvement of complex DL architectures
    """
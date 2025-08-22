import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_process import data_pipeline
from utils import ML_model_train,ML_models_call,ML_model_eval

def model_pipeline(file_path:str):
    data = data_pipeline(file_path)
    
    print("\nPreprocessed data fetched\n")

    RAW_model = ML_models_call(type="Classification",model="Logistic Regression")
    print("\nModel fetched\n")

    trained_model = ML_model_train(model=RAW_model,data=data)
    print("\nModel Trained\n")

    ML_model_eval(trained_model,test_data=[data[1],data[3]],type="Classification")
    print("\nModel evaluated")


model_pipeline(r"Test_data\tested.csv")
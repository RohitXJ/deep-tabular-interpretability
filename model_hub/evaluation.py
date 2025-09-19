from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score,accuracy_score,mean_absolute_percentage_error
import numpy as np

def ML_model_eval(model: object, test_data: list, type: str):
    X_test,y_test = test_data
    y_pred = model.predict(X_test)

    if type == "Classification":
        print("Classification Report")
        report_str = classification_report(y_true=y_test, y_pred=y_pred)
        report_dict = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
        print(report_str)

        accuracy = report_dict['accuracy']
        f1_score = report_dict['weighted avg']['f1-score']

        summary = "\n--- Model Performance Summary ---\n"
        if accuracy > 0.9 and f1_score > 0.9:
            summary += "Excellent! The model demonstrates high accuracy and a strong F1-score, indicating it is very effective at classifying the data."
        elif accuracy > 0.8 and f1_score > 0.8:
            summary += "Good. The model has a solid accuracy and F1-score, suggesting it performs well for most cases."
        elif accuracy > 0.7 and f1_score > 0.7:
            summary += "Fair. The model's performance is reasonable, but there might be room for improvement. It correctly classifies a moderate amount of the data."
        else:
            summary += "Needs Improvement. The model's accuracy and F1-score are low. It may struggle to make correct predictions and could benefit from further tuning or more data."
        
        summary += f"\n(Accuracy: {accuracy:.2f}, Weighted F1-Score: {f1_score:.2f})"
        print(summary)


    elif type == "Regression":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Regression Evaluation")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R Score: {r2:.2f}")
        print(f"Mean Absolute Error Percentage (MAPE): {(mape*100):.2f}%")

        summary = "\n--- Model Performance Summary ---\n"
        if r2 > 0.9 and mape < 0.1:
            summary += "Excellent! The model explains a very high proportion of the variance in the target variable and has a low prediction error."
        elif r2 > 0.8 and mape < 0.2:
            summary += "Good. The model provides a solid fit to the data, explaining a majority of the variance with a reasonable error margin."
        elif r2 > 0.6 and mape < 0.3:
            summary += "Fair. The model has some predictive power but may not be very precise. The R-squared value is moderate, and the error percentage is noticeable."
        else:
            summary += "Needs Improvement. The model does not fit the data well. The R-squared is low (or even negative), and the prediction error is high."

        summary += f"\n(R-squared: {r2:.2f}, MAPE: {mape:.2%})"
        print(summary)

    else:
        raise ValueError("Invalid type. Choose either 'Classification' or 'Regression'.")

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from tab_transformer_pytorch import TabTransformer
import torch.nn as nn


def DL_model_eval(model: object, test_data: list, type: str, cat_features:list, num_features:list, device:str='cpu'):
    X_test, y_test = test_data

    if isinstance(model, (TabNetClassifier, TabNetRegressor)):
        y_pred = model.predict(X_test.to_numpy())
        y_test = y_test.to_numpy()
    else:
        device = torch.device(device)
        model.to(device)

        # Convert data to PyTorch tensors
        if isinstance(model, TabTransformer):
            X_test_cat = torch.tensor(X_test[cat_features].values, dtype=torch.long)
            X_test_num = torch.tensor(X_test[num_features].values, dtype=torch.float32)
            X_test_tensor = (X_test_cat, X_test_num)
        elif isinstance(model, nn.Sequential): # Explicitly handle FNN
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        else:
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        if type == "Classification":
            y_test_tensor = y_test_tensor.long()

        # Create a TensorDataset and DataLoader
        if isinstance(model, TabTransformer):
            test_dataset = TensorDataset(X_test_tensor[0], X_test_tensor[1], y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=32)
        else:
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=32)

        cat_feature_indices = [X_test.columns.get_loc(c) for c in cat_features if c in X_test.columns]
        num_feature_indices = [X_test.columns.get_loc(c) for c in num_features if c in X_test.columns]

        model.eval()
        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for batch_data in test_loader:
                if isinstance(model, TabTransformer):
                    batch_X_cat, batch_X_num, batch_y = batch_data
                    batch_X_cat, batch_X_num, batch_y = batch_X_cat.to(device), batch_X_num.to(device), batch_y.to(device)
                    outputs = model(batch_X_cat, batch_X_num)
                else:
                    batch_X, batch_y = batch_data
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)

                if type == "Classification":
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred_list.extend(predicted.cpu().numpy())
                else:
                    y_pred_list.extend(outputs.cpu().numpy().flatten())
                
                y_true_list.extend(batch_y.cpu().numpy())

        y_pred = np.array(y_pred_list)
        y_test = np.array(y_true_list)

    if type == "Classification":
        print("Classification Report")
        report_str = classification_report(y_true=y_test, y_pred=y_pred)
        report_dict = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
        print(report_str)

        accuracy = report_dict['accuracy']
        f1_score = report_dict['weighted avg']['f1-score']

        summary = "\n--- Model Performance Summary ---"
        if accuracy > 0.9 and f1_score > 0.9:
            summary += "Excellent! The model demonstrates high accuracy and a strong F1-score, indicating it is very effective at classifying the data."
        elif accuracy > 0.8 and f1_score > 0.8:
            summary += "Good. The model has a solid accuracy and F1-score, suggesting it performs well for most cases."
        elif accuracy > 0.7 and f1_score > 0.7:
            summary += "Fair. The model's performance is reasonable, but there might be room for improvement. It correctly classifies a moderate amount of the data."
        else:
            summary += "Needs Improvement. The model's accuracy and F1-score are low. It may struggle to make correct predictions and could benefit from further tuning or more data."
        
        summary += f"\n(Accuracy: {accuracy:.2f}, Weighted F1-Score: {f1_score:.2f})"
        print(summary)

    elif type == "Regression":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Regression Evaluation")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R Score: {r2:.2f}")
        print(f"Mean Absolute Error Percentage (MAPE): {(mape*100):.2f}%")

        summary = "\n--- Model Performance Summary ---"
        if r2 > 0.9 and mape < 0.1:
            summary += "Excellent! The model explains a very high proportion of the variance in the target variable and has a low prediction error."
        elif r2 > 0.8 and mape < 0.2:
            summary += "Good. The model provides a solid fit to the data, explaining a majority of the variance with a reasonable error margin."
        elif r2 > 0.6 and mape < 0.3:
            summary += "Fair. The model has some predictive power but may not be very precise. The R-squared value is moderate, and the error percentage is noticeable."
        else:
            summary += "Needs Improvement. The model does not fit the data well. The R-squared is low (or even negative), and the prediction error is high."

        summary += f"\n(R-squared: {r2:.2f}, MAPE: {mape:.2%})"
        print(summary)

    else:
        raise ValueError("Invalid type. Choose either 'Classification' or 'Regression'.")


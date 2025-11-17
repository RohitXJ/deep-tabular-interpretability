import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np
import sys
import io

def DL_model_eval(model: nn.Module, test_data: list, prediction_type: str) -> str:
    """
    Evaluates a Deep Learning model and returns a formatted report.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_data (list): A list containing X_test_t (torch.Tensor) and y_test_t (torch.Tensor).
        prediction_type (str): "Classification" or "Regression".

    Returns:
        str: A formatted string containing the evaluation report.
    """
    X_test_t, y_test_t = test_data
    model.eval()
    
    # Capture stdout for live logging
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    with torch.no_grad():
        predictions = model(X_test_t)

    report = []

    if prediction_type == "Regression":
        predictions_np = predictions.numpy()
        y_test_np = y_test_t.numpy()

        mse = mean_squared_error(y_test_np, predictions_np)
        mae = mean_absolute_error(y_test_np, predictions_np)
        r2 = r2_score(y_test_np, predictions_np)

        report.append("--- Regression Model Evaluation ---")
        report.append(f"Mean Squared Error (MSE): {mse:.4f}")
        report.append(f"Mean Absolute Error (MAE): {mae:.4f}")
        report.append(f"R-squared (R2): {r2:.4f}")

    elif prediction_type == "Classification":
        predictions_np = (predictions > 0.5).float().numpy()
        y_test_np = y_test_t.numpy()

        accuracy = accuracy_score(y_test_np, predictions_np)
        precision = precision_score(y_test_np, predictions_np)
        recall = recall_score(y_test_np, predictions_np)
        f1 = f1_score(y_test_np, predictions_np)

        report.append("--- Classification Model Evaluation ---")
        report.append(f"Accuracy: {accuracy:.4f}")
        report.append(f"Precision: {precision:.4f}")
        report.append(f"Recall: {recall:.4f}")
        report.append(f"F1-Score: {f1:.4f}")
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    sys.stdout = old_stdout # Restore stdout
    
    return "\n".join(report)

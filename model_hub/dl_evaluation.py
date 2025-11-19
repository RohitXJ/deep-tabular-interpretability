import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

def _generate_dl_regression_summary(r2, mae):
    """Generates a qualitative summary for DL regression model performance."""
    r2_feedback = ""
    if r2 > 0.9:
        r2_feedback = f"R-squared ({r2:.2f}): Excellent. The model explains a very high percentage of the variance."
    elif r2 > 0.8:
        r2_feedback = f"R-squared ({r2:.2f}): Good. The model explains a majority of the variance."
    elif r2 > 0.6:
        r2_feedback = f"R-squared ({r2:.2f}): Fair. The model has moderate predictive power."
    else:
        r2_feedback = f"R-squared ({r2:.2f}): Needs Improvement. The model's predictive power is low."

    mae_feedback = ""
    # Note: MAE interpretation is context-dependent. These are general guidelines.
    # A good MAE depends on the scale of the target variable.
    # For this summary, we'll provide a general qualitative assessment.
    # A more advanced version could compare MAE to the target's standard deviation.
    if r2 > 0.8: # Only give strong positive feedback on MAE if R2 is also good
        mae_feedback = f"Mean Absolute Error ({mae:.2f}): Appears reasonable for a model with this R-squared."
    else:
        mae_feedback = f"Mean Absolute Error ({mae:.2f}): Review this value in the context of your target variable's scale to determine its significance."

    overall_summary = ""
    if r2 >= 0.8:
        overall_summary = "Overall Assessment: This appears to be a strong and reliable regression model."
    elif r2 >= 0.6:
        overall_summary = "Overall Assessment: A decent model, but with room for improvement in its predictive power."
    else:
        overall_summary = "Overall Assessment: This model has low predictive power and likely needs significant improvement or a different approach."

    summary = f"""
--- Model Performance Summary ---
- {r2_feedback}
- {mae_feedback}

{overall_summary}
"""
    return summary

def _generate_dl_classification_summary(accuracy, f1):
    """Generates a qualitative summary for DL classification model performance."""
    summary = "\n--- Model Performance Summary ---\n"
    if accuracy > 0.9 and f1 > 0.9:
        summary += "Excellent! The model demonstrates high accuracy and a strong F1-score, indicating it is very effective."
    elif accuracy > 0.8 and f1 > 0.8:
        summary += "Good. The model has solid accuracy and F1-score, suggesting it performs well."
    elif accuracy > 0.7 and f1 > 0.7:
        summary += "Fair. The model's performance is reasonable, but there might be room for improvement."
    else:
        summary += "Needs Improvement. The model's accuracy and F1-score are low, suggesting it may struggle to make correct predictions."
    
    summary += f"\n(Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f})"
    return summary

def DL_model_eval(model: nn.Module, test_data: list, prediction_type: str) -> str:
    """
    Evaluates a Deep Learning model and returns a formatted report including a qualitative summary.
    """
    X_test_t, y_test_t = test_data
    model.eval()
    
    report_lines = []

    with torch.no_grad():
        predictions = model(X_test_t)

    if prediction_type == "Regression":
        predictions_np = predictions.numpy()
        y_test_np = y_test_t.numpy()

        mse = mean_squared_error(y_test_np, predictions_np)
        mae = mean_absolute_error(y_test_np, predictions_np)
        r2 = r2_score(y_test_np, predictions_np)

        report_lines.append("--- Regression Model Evaluation ---")
        report_lines.append(f"Mean Squared Error (MSE): {mse:.4f}")
        report_lines.append(f"Mean Absolute Error (MAE): {mae:.4f}")
        report_lines.append(f"R-squared (R2): {r2:.4f}")
        
        # Add qualitative summary
        summary = _generate_dl_regression_summary(r2, mae)
        report_lines.append(summary)

    elif prediction_type == "Classification":
        # Using 0.5 as the threshold for binary classification
        predictions_np = (predictions > 0.5).float().numpy()
        y_test_np = y_test_t.numpy()

        accuracy = accuracy_score(y_test_np, predictions_np)
        precision = precision_score(y_test_np, predictions_np, zero_division=0)
        recall = recall_score(y_test_np, predictions_np, zero_division=0)
        f1 = f1_score(y_test_np, predictions_np, zero_division=0)

        report_lines.append("--- Classification Model Evaluation ---")
        report_lines.append(f"Accuracy: {accuracy:.4f}")
        report_lines.append(f"Precision: {precision:.4f}")
        report_lines.append(f"Recall: {recall:.4f}")
        report_lines.append(f"F1-Score: {f1:.4f}")

        # Add qualitative summary
        summary = _generate_dl_classification_summary(accuracy, f1)
        report_lines.append(summary)
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    return "\n".join(report_lines)

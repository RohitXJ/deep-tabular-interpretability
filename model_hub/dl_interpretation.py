import shap
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# Import ANN architectures for type hinting and model loading
from ANN_architecture import ANN_Shallow_Regression, ANN_Deep_Regression, ANN_Shallow_Classification, ANN_Deep_Classification

def generate_dl_interpretation(model, X_test_t, X_test_scaled_np, features, prediction_type, interp_dir, background_data_t):
    """
    Generates SHAP interpretation plots for Deep Learning models.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        X_test_t (torch.Tensor): Test features as a PyTorch tensor.
        X_test_scaled_np (np.ndarray): Test features as a NumPy array (scaled).
        features (list): List of feature names.
        prediction_type (str): "Classification" or "Regression".
        interp_dir (str): Directory to save interpretation plots.
        background_data_t (torch.Tensor): Background data for SHAP DeepExplainer.

    Returns:
        list: A list of dictionaries, each containing metadata about a generated plot.
    """
    plots_metadata = []
    model.eval()

    explainer = shap.DeepExplainer(model, background_data_t)
    
    # SHAP values calculation
    if prediction_type == "Regression":
        shap_values = explainer.shap_values(X_test_t)
    elif prediction_type == "Classification":
        # For classification, DeepExplainer returns a list of shap_values for each output.
        # We take the shap values for the positive class (index 0 for binary classification)
        raw_shap_values = explainer.shap_values(X_test_t)[0].squeeze()
        shap_values = np.atleast_2d(raw_shap_values)
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    # Global Feature Importance (Summary Plot)
    summary_plot_filename = f"shap_summary_plot_{prediction_type}.png"
    summary_plot_path = os.path.join(interp_dir, summary_plot_filename)
    plt.figure(figsize=(10, 6))
    
    # Slice the feature data to match the number of rows in the SHAP data
    n_rows_shap = shap_values.shape[0]
    features_for_plot = X_test_scaled_np[:n_rows_shap]
    
    shap.summary_plot(shap_values, features_for_plot, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()
    plots_metadata.append({
        'type': 'image',
        'filename': summary_plot_filename,
        'title': f'SHAP Summary Plot ({prediction_type})',
        'description': 'This plot shows the global importance of each feature. Features are ordered by their impact on the model output. The color indicates the feature value (red for high, blue for low).'
    })

    # Local Interpretation (Waterfall Plot for Sample 0)
    if X_test_scaled_np.shape[0] > 0: # Ensure there's at least one sample
        waterfall_plot_filename = f"shap_waterfall_plot_{prediction_type}_sample_0.png"
        waterfall_plot_path = os.path.join(interp_dir, waterfall_plot_filename)
        
        if prediction_type == "Regression":
            exp = shap.Explanation(
                values=shap_values[0].squeeze(),
                base_values=explainer.expected_value[0],
                data=X_test_scaled_np[0],
                feature_names=features
            )
        elif prediction_type == "Classification":
            exp = shap.Explanation(
                values=shap_values[0], # Already squeezed and 2D if needed
                base_values=explainer.expected_value[0], # For binary classification, usually one expected value
                data=X_test_scaled_np[0],
                feature_names=features
            )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.tight_layout()
        plt.savefig(waterfall_plot_path)
        plt.close()
        plots_metadata.append({
            'type': 'image',
            'filename': waterfall_plot_filename,
            'title': f'SHAP Waterfall Plot for Sample 0 ({prediction_type})',
            'description': 'This plot explains the prediction for a single instance. The base value is the average model output. Each feature\'s value shows how much it pushes the prediction from the base value to the final output.'
        })

    return plots_metadata

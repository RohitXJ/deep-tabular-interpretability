import shap
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Import ANN architectures for type hinting and model loading
from ANN_architecture import ANN_Shallow_Regression, ANN_Deep_Regression, ANN_Shallow_Classification, ANN_Deep_Classification

def generate_dl_interpretation(model, X_test_t, X_test_scaled_np, X_test_unscaled_np, features, prediction_type, interp_dir, background_data_t):
    """
    Generates SHAP interpretation plots for Deep Learning models.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        X_test_t (torch.Tensor): Test features as a PyTorch tensor (scaled).
        X_test_scaled_np (np.ndarray): Test features as a NumPy array (scaled).
        X_test_unscaled_np (np.ndarray): Test features as a NumPy array (unscaled, for plotting).
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
    features_for_plot = X_test_unscaled_np[:n_rows_shap]
    
    shap.summary_plot(shap_values, features_for_plot, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()
    plots_metadata.append({
        'type': 'image',
        'filename': summary_plot_filename,
        'title': f'SHAP Summary Plot ({prediction_type})',
        'description': """
            <strong>How to Read This Plot:</strong>
            <br>
            This plot, often called a "beeswarm" plot, provides a high-level overview of how much each feature impacts the model's predictions across all the samples in your test dataset.
            <ul>
                <li><strong>Feature Importance:</strong> Features are ranked by importance along the y-axis, with the most influential feature at the top. A feature's importance is calculated by taking the average of the absolute SHAP values for that feature over all samples.</li>
                <li><strong>Impact on Prediction (X-axis):</strong> The horizontal location of each dot shows the SHAP value for a specific prediction. A positive SHAP value means that the feature's value pushed the prediction higher (e.g., towards a higher price or a higher probability of being in the positive class). A negative SHAP value pushed the prediction lower.</li>
                <li><strong>Original Feature Value (Color):</strong> The color of each dot represents the actual value of that feature for that specific data point. Typically, red indicates a high value for the feature, and blue indicates a low value.</li>
                <li><strong>Putting It Together:</strong> For example, if the "Room_Number" feature has many red dots on the positive side of the x-axis, it suggests that a higher number of rooms generally increases the model's output (e.g., predicts a higher house price).</li>
            </ul>
        """
    })

    # Local Interpretation (Waterfall Plot for Sample 0)
    if X_test_unscaled_np.shape[0] > 0: # Ensure there's at least one sample
        waterfall_plot_filename = f"shap_waterfall_plot_{prediction_type}_sample_0.png"
        waterfall_plot_path = os.path.join(interp_dir, waterfall_plot_filename)
        
        if prediction_type == "Regression":
            exp = shap.Explanation(
                values=shap_values[0].squeeze(),
                base_values=explainer.expected_value[0],
                data=X_test_unscaled_np[0],
                feature_names=features
            )
        elif prediction_type == "Classification":
            exp = shap.Explanation(
                values=shap_values[0], # Already squeezed and 2D if needed
                base_values=explainer.expected_value[0], # For binary classification, usually one expected value
                data=X_test_unscaled_np[0],
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
            'description': """
            <strong>How to Read This Plot:</strong>
            <br>
            This waterfall plot breaks down the prediction for a single, specific data point (in this case, the very first sample in your test set). It shows exactly how each feature contributed to pushing the model's output from a starting "base value" to the final prediction.
            <ul>
                <li><strong>The Base Value E[f(x)]:</strong> This is the starting point of the plot, shown at the bottom. It represents the average prediction the model would make across the entire training dataset if it had no information about the current data point.</li>
                <li><strong>Feature Contributions:</strong> Each row in the plot shows a feature and its value for this specific data point. The red and blue arrows show the "force" that each feature's value exerted on the prediction.
                    <ul>
                        <li><span style="color: red;">■</span> <strong>Red arrows</strong> represent features that pushed the prediction to a <strong>higher value</strong>. The longer the arrow, the stronger the push.</li>
                        <li><span style="color: blue;">■</span> <strong>Blue arrows</strong> represent features that pushed the prediction to a <strong>lower value</strong>.</li>
                    </ul>
                </li>
                <li><strong>The Final Prediction f(x):</strong> This is the final output of the model for this data point, shown at the top. It is the sum of the base value and all the individual feature contributions. By following the arrows from bottom to top, you can see how the model "built" its final prediction.</li>
            </ul>
        """
        })

    return plots_metadata

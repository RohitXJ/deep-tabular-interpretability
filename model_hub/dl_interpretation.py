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
    Generates SHAP interpretation plots for Deep Learning models, including detailed explanations.
    """
    # --- Subsample data for performance ---
    if X_test_t.shape[0] > 1000:
        print(f"Subsampling data from {X_test_t.shape[0]} to 1000 rows for SHAP analysis.")
        X_test_t = X_test_t[:1000]
        X_test_scaled_np = X_test_scaled_np[:1000]
        X_test_unscaled_np = X_test_unscaled_np[:1000]

    plots_metadata = []
    model.eval()

    explainer = shap.DeepExplainer(model, background_data_t)
    
    # --- SHAP Value Calculation (Robust approach inspired by debug notebook) ---
    shap_values_raw = explainer.shap_values(X_test_t)

    # If it returns a list (common for classification), take the first element.
    if isinstance(shap_values_raw, list):
        shap_values_raw = shap_values_raw[0]

    # If it's a 3D array (e.g., Samples, Features, 1), remove the last dimension.
    if len(shap_values_raw.shape) == 3:
        shap_values_raw = shap_values_raw.squeeze(-1)

    # For a single prediction, squeeze might make it 1D, so ensure it is at least 2D.
    shap_values = np.atleast_2d(shap_values_raw)
    
    # Ensure data for plotting has the same number of samples as the SHAP values
    n_rows_shap = shap_values.shape[0]
    features_for_plot_scaled = X_test_scaled_np[:n_rows_shap]
    features_for_plot_unscaled = X_test_unscaled_np[:n_rows_shap]

    # --- 1. Global Feature Importance (Bar Plot) ---
    bar_plot_filename = f"shap_bar_plot_{prediction_type}.png"
    bar_plot_path = os.path.join(interp_dir, bar_plot_filename)
    plt.figure()
    
    # Create a full Explanation object including data to prevent shape mismatches
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=features_for_plot_unscaled,
        feature_names=features
    )
    
    shap.plots.bar(shap_explanation, show=False)
    plt.title("Global Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(bar_plot_path)
    plt.close()
    plots_metadata.append({
        'type': 'image',
        'filename': bar_plot_filename,
        'title': 'Global Feature Importance (The "What")',
        'description': """
            <strong>What It Is:</strong> This bar chart shows the average impact of each feature on the model's prediction magnitude, across the entire dataset.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li>The features are listed on the y-axis, ordered from most important at the top to least important at the bottom.</li>
                <li>The length of the bar on the x-axis represents the mean absolute SHAP value for that feature. A longer bar means the feature has a greater average impact on the model's predictions.</li>
                <li>This plot gives you a straightforward ranking of which features are the most influential overall, but it does not show the direction of the impact (i.e., whether a high value of a feature increases or decreases the prediction).</li>
            </ul>
        """
    })

    # --- 2. Global Summary Plot (Beeswarm) ---
    summary_plot_filename = f"shap_summary_plot_{prediction_type}.png"
    summary_plot_path = os.path.join(interp_dir, summary_plot_filename)
    plt.figure()
    shap.summary_plot(shap_values, features_for_plot_scaled, feature_names=features, show=False)
    plt.title("Feature Impact Summary", fontsize=16)
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()
    plots_metadata.append({
        'type': 'image',
        'filename': summary_plot_filename,
        'title': 'Feature Impact Summary (The "How")',
        'description': """
            <strong>What It Is:</strong> This plot provides a rich overview of how every feature impacts the model's output for every sample in your dataset. It combines feature importance with feature effects.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Vertical Axis (Feature Importance):</strong> Features are ranked from most important (top) to least important (bottom). A feature's importance is the average absolute SHAP value across all samples.</li>
                <li><strong>Horizontal Axis (Impact on Prediction):</strong> This is the <strong>SHAP Value</strong>. It shows how much a feature's value for a specific data point pushed the model's output.
                    <ul>
                        <li>Values > 0 mean the feature pushed the prediction <strong>higher</strong> (e.g., higher house price, or higher probability of being a "positive" class).</li>
                        <li>Values < 0 mean the feature pushed the prediction <strong>lower</strong>.</li>
                    </ul>
                </li>
                <li><strong>Dot Color (Feature Value):</strong> The color of each dot shows if that feature's value was high or low for that data point.
                    <ul>
                        <li><span style="color:red;">■ Red dots</span> represent <strong>high values</strong> of a feature.</li>
                        <li><span style="color:blue;">■ Blue dots</span> represent <strong>low values</strong> of a feature.</li>
                        <li><strong>Black dots</strong> typically represent instances where the feature's value was <strong>missing (NaN)</strong>. The plot shows the impact this missingness has on the prediction.</li>
                    </ul>
                </li>
                <li><strong>Dot Position:</strong> Each dot is one sample from your data. They pile up horizontally to show the density of SHAP values for each feature.</li>
            </ul>
            <strong>Putting It Together:</strong> For example, if the "worst radius" feature for a cancer prediction has many red dots on the right side (positive SHAP values), it strongly suggests that a larger radius significantly increases the model's prediction towards "malignant".
        """
    })

    # --- 3. Local Interpretation (Waterfall Plot) ---
    if features_for_plot_unscaled.shape[0] > 0:
        # Select a random sample for the waterfall plot
        random_index = np.random.randint(0, features_for_plot_unscaled.shape[0])
        
        waterfall_plot_filename = f"shap_waterfall_plot_{prediction_type}_sample_{random_index}.png"
        waterfall_plot_path = os.path.join(interp_dir, waterfall_plot_filename)
        
        base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (np.ndarray, list)) else explainer.expected_value

        exp = shap.Explanation(
            values=shap_values[random_index].squeeze(),
            base_values=base_value,
            data=features_for_plot_unscaled[random_index], # Use unscaled data for readability
            feature_names=features
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, max_display=20, show=False)
        plt.title(f"Breakdown of a Single Prediction (Sample #{random_index})", fontsize=16)
        plt.tight_layout()
        plt.savefig(waterfall_plot_path)
        plt.close()
        plots_metadata.append({
            'type': 'image',
            'filename': waterfall_plot_filename,
            'title': 'Single Prediction Explained (The "Why")',
            'description': """
            <strong>What It Is:</strong> This waterfall plot dissects a single prediction to show you exactly how the model made its decision for one specific data point.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Base Value (E[f(x)]):</strong> This is the starting point at the bottom of the plot. The notation <strong>E[f(x)]</strong> represents the <em>average model output</em> over the entire training dataset. Think of this as the model's default prediction before it has seen any features for this specific data point.</li>
                <li><strong>Feature Contributions:</strong> Each bar shows how a feature's value for this data point pushed the prediction away from the base value.
                    <ul>
                        <li><span style="color:red;">■ Red bars</span> are features that <strong>pushed the prediction higher</strong> (a positive impact). The longer the bar, the stronger the push.</li>
                        <li><span style="color:blue;">■ Blue bars</span> are features that <strong>pushed the prediction lower</strong> (a negative impact).</li>
                    </ul>
                </li>
                <li><strong>Feature Values:</strong> The numbers next to the feature names (e.g., `mean radius = 17.99`) are their actual, real-world values for this one data point, making the explanation easy to interpret.</li>
                <li><strong>Final Prediction (f(x)):</strong> This is the final prediction value at the top of the plot. The notation <strong>f(x)</strong> represents the model's output for this <em>specific input (x)</em>. It is calculated by summing the base value (E[f(x)]) and the contributions of all the features.</li>
            </ul>
        """
        })

    return plots_metadata
def plot_training_loss(loss_history, model_name, plot_path):
    """
    Generates and saves a line chart of the model's training loss.
    
    Args:
        loss_history (list): A list of float values representing loss per epoch.
        model_name (str): The name of the model for the plot title.
        plot_path (str): The full path where the plot image should be saved.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training Convergence', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

import shap
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Import ANN architectures for type hinting and model loading
from ANN_architecture import ANN_Shallow_Regression, ANN_Deep_Regression, ANN_Shallow_Classification, ANN_Deep_Classification

def generate_dl_interpretation(model, X_test_t, X_test_scaled_np, X_test_unscaled_np, features, prediction_type, interp_dir, background_data_t, y_test_np):
    """
    Generates SHAP interpretation plots for Deep Learning models, including detailed explanations.
    """
    # --- Subsample data for performance ---
    if X_test_t.shape[0] > 1000:
        print(f"Subsampling data from {X_test_t.shape[0]} to 1000 rows for SHAP analysis.")
        X_test_t = X_test_t[:1000]
        X_test_scaled_np = X_test_scaled_np[:1000]
        X_test_unscaled_np = X_test_unscaled_np[:1000]
        y_test_np = y_test_np[:1000] # Subsample y_test_np too

    # --- NEW SAMPLING LOGIC FOR SHAP INTERPRETATION ---
    y_series = pd.Series(y_test_np.flatten()) # Flatten y_test_np for Series conversion

    if prediction_type == "Classification":
        print("Applying stratified sampling for Classification task.")
        n_classes = y_series.nunique()
        
        if n_classes == 2:
            n_per_class = 5
            print(f"Binary classification: selecting {n_per_class} samples per class.")
        else:
            n_total = max(10, n_classes)
            n_per_class = n_total // n_classes
            print(f"Multi-class ({n_classes} classes): selecting {n_per_class} samples per class.")

        # Group by target variable and sample within each group
        sample_indices = y_series.groupby(y_series).apply(
            lambda x: x.sample(n=min(n_per_class, len(x)), random_state=42)
        ).index.get_level_values(1).values # Get original indices as numpy array

    else: # Regression
        n_shap_samples = 15
        print(f"Applying random sampling for Regression task: {n_shap_samples} samples.")
        # Need to sample from the original indices range
        all_indices = np.arange(X_test_t.shape[0])
        sample_indices = np.random.choice(all_indices, size=n_shap_samples, replace=False)

    # Apply the selected indices to all relevant data arrays
    X_test_t = X_test_t[sample_indices]
    X_test_scaled_np = X_test_scaled_np[sample_indices]
    X_test_unscaled_np = X_test_unscaled_np[sample_indices]
    y_test_np = y_test_np[sample_indices] # Keep y_test_np consistent

    print(f"Total samples selected for SHAP analysis: {len(sample_indices)}")

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
            <strong>What It Is:</strong> This bar chart shows the average impact of each feature on the model's prediction magnitude, across the entire dataset. It answers the question: "Overall, which features are the most important for the model's decisions?"
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li>The features are listed on the y-axis, ordered from most important at the top to least important at the bottom.</li>
                <li>The length of the bar on the x-axis represents the <strong>mean absolute SHAP value</strong> for that feature. In simple terms, a longer bar means the feature has a greater average impact on the model's predictions, regardless of whether that impact is positive or negative.</li>
                <li><strong>Example:</strong> If this were a house price prediction model, a long bar for "sqft_living" means that the size of the house is consistently one of the most influential factors in determining the final price prediction.</li>
                <li>This plot gives a straightforward ranking of which features are most influential overall, but it does not show the direction or nature of the impact. For that, we use the Summary (Beeswarm) Plot.</li>
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
            <strong>What It Is:</strong> This plot is one of the most powerful SHAP visualizations. It shows not only which features are important, but also *how* the value of that feature affects the model's output for every single sample in your dataset.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Vertical Axis (Feature Importance):</strong> Features are ranked from most important (top) to least important (bottom), based on their average impact.</li>
                <li><strong>Horizontal Axis (Impact on Prediction):</strong> This is the <strong>SHAP Value</strong>. It shows the precise impact of a feature on the final prediction.
                    <ul>
                        <li>Values > 0 (to the right of the center line) mean the feature pushed the prediction <strong>higher</strong> (e.g., a higher house price, or a higher probability of being in the "Yes" class).</li>
                        <li>Values < 0 (to the left) mean the feature pushed the prediction <strong>lower</strong>.</li>
                    </ul>
                </li>
                <li><strong>Dot Color (Feature Value):</strong> The color of each dot shows whether that feature's value was high or low for that data point.
                    <ul>
                        <li><span style="color:red;">■ Red dots</span> represent <strong>high values</strong> of a feature (e.g., a large house).</li>
                        <li><span style="color:blue;">■ Blue dots</span> represent <strong>low values</strong> of a feature (e.g., a small house).</li>
                    </ul>
                </li>
                <li><strong>Each Dot:</strong> Every dot on the plot is a single prediction for one data sample. They pile up to show the distribution of impact for each feature.</li>
            </ul>
            <strong>Putting It All Together (Example):</strong> Imagine a cancer prediction model. If the feature "tumor_size" has a trail of red dots extending far to the right, it tells you that high values of "tumor_size" (large tumors) consistently and strongly push the model's prediction towards "Malignant" (a higher output value). Conversely, blue dots on the left would mean small tumors push the prediction towards "Benign".
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
            <strong>What It Is:</strong> This waterfall plot provides a transparent, step-by-step breakdown of how the model arrived at its final prediction for one specific data point. It's the ultimate "show your work" for the model.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Base Value (E[f(x)]):</strong> The term <strong>E[f(x)]</strong> stands for the "Expected Value of the model's function f(x)". In simple terms, this is the <strong>average prediction</strong> over the entire dataset. It's the baseline or starting point for any prediction.
                    <ul><li><strong>Example:</strong> If predicting house prices, this would be the average house price in your dataset (e.g., $300,000). If predicting loan default, this might be the average default rate (e.g., 5%).</li></ul>
                </li>
                <li><strong>Feature Contributions:</strong> Each bar represents how the value of a specific feature for this single data point moved the prediction away from the average base value.
                    <ul>
                        <li><span style="color:red;">■ Red bars</span> show features that <strong>pushed the prediction higher</strong>. The number next to the feature name is its actual value (e.g., `sqft_living = 2100`).</li>
                        <li><span style="color:blue;">■ Blue bars</span> show features that <strong>pushed the prediction lower</strong>.</li>
                        <li>The <strong>length</strong> of the bar shows the magnitude of that feature's impact.</li>
                    </ul>
                </li>
                <li><strong>Final Prediction (f(x)):</strong> The term <strong>f(x)</strong> represents the model's final output for this specific input data point (x). It is calculated by summing the base value and all the individual feature contributions.
                    <ul><li><strong>Example:</strong> After all the red and blue bars are added to the base value, we get the final prediction for this specific house, such as $450,000. For a classification, the final output is in "log-odds" space, where a positive value predicts the "Yes" class and a negative value predicts the "No" class.</li></ul>
                </li>
            </ul>
        """
        ,
            'predicted_output': predicted_output_str
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

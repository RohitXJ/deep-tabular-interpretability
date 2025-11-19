import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize SHAP for JavaScript plots
shap.initjs()

TREE_BASED_MODELS = ["Random Forest Classifier", "Random Forest Regressor", "XGBoost", "LightGBM", "CatBoost"]

def generate_interpretation(model, X_test, X_test_unscaled, config, interp_dir):
    model_name = config['model']
    prediction_type = config['prediction_type']
    plots = []

    # --- Subsample data if too wide --- #
    max_features = config.get('max_interpretation_features', 2000) # Get from config, default to 2000
    if X_test.shape[1] > max_features:
        print(f"Dataset has {X_test.shape[1]} features, subsampling to {max_features} for SHAP analysis.")
        # Get feature importances to guide sampling
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X_test.columns)
            
            # Calculate top_n and random_n based on max_features
            top_n = int(max_features * 0.8)
            random_n = max_features - top_n

            top_features = importances.nlargest(top_n).index.tolist()
            remaining_features = [col for col in X_test.columns if col not in top_features]
            
            # Ensure we don't try to sample more than available remaining features
            random_sample_count = min(random_n, len(remaining_features))
            random_sample = np.random.choice(remaining_features, random_sample_count, replace=False).tolist()
            
            final_cols = top_features + random_sample
        else: # Fallback to random sampling if no importances
            final_cols = np.random.choice(X_test.columns, max_features, replace=False).tolist()
        
        X_test = X_test[final_cols]
        X_test_unscaled = X_test_unscaled[final_cols]

    # Initialize SHAP for JavaScript plots
    shap.initjs()

    # --- SHAP Analysis --- #
    plt.figure(figsize=(10, 8))

    # Select the correct explainer based on model type
    if model_name in TREE_BASED_MODELS:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    else:  # Linear models, SVM, etc.
        background = shap.sample(X_test, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test)

    # Normalize SHAP values and base value for plotting, especially for classifiers.
    # This block handles the different output formats from SHAP.
    shap_values_for_plot = shap_values
    base_value_for_plot = explainer.expected_value

    if prediction_type == "Classification":
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Handles format: [array(class_0), array(class_1)]
            shap_values_for_plot = shap_values[1]
            base_value_for_plot = explainer.expected_value[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Handles format: array(n_samples, n_features, n_classes)
            shap_values_for_plot = shap_values[:, :, 1]
            base_value_for_plot = explainer.expected_value[1]

    # --- 1. Global Importance (SHAP Beeswarm) --- #
    shap.summary_plot(shap_values_for_plot, X_test_unscaled, show=False)
    plt.title("Overall Feature Importance and Impact")
    plt.tight_layout()
    filename_summary = "shap_summary.png"
    plt.savefig(os.path.join(interp_dir, filename_summary))
    plt.close()
    plots.append({
        'title': "Overall Feature Importance (The What)",
        'explanation': "This plot shows the overall importance and impact of each feature on the model's predictions. Each row represents a feature, ordered by its importance (top features are most impactful). Each dot is a data point from your dataset. The position of the dot on the horizontal axis indicates the SHAP value for that feature, showing how much that feature's value contributed to pushing the prediction higher or lower. The color of the dot (red to blue) indicates the original value of the feature for that data point (red for high values, blue for low values). This helps you understand which features are most influential and how their values affect the outcome.",
        'type': 'image',
        'filename': filename_summary
    })

    # --- 2. Individual Prediction Breakdown (Force Plot) --- #
    force_plot_html_path = os.path.join(interp_dir, "force_plot.html")

    # Safely convert the base value to a scalar float. This is critical.
    if hasattr(base_value_for_plot, 'item'):
        # This is the correct way to get a scalar from a numpy array/float
        final_base_value = base_value_for_plot.item()
    else:
        # It's already a standard Python float
        final_base_value = base_value_for_plot

    # Generate a stacked force plot for all test instances.
    force_plot = shap.plots.force(
        final_base_value,
        shap_values_for_plot,
        X_test_unscaled,
        show=False
    )
    shap.save_html(force_plot_html_path, force_plot)
    plots.append({
        'title': "Individual Prediction Breakdown (The Why)",
        'explanation': "This interactive plot visualizes how each feature contributes to a single prediction, or a set of predictions. The 'Base Value' (f(x) in the plot) is the average prediction across the entire dataset. The plot shows how individual feature values (represented by colored bands) push the prediction from this base value towards the final output. Features pushing the prediction higher are shown in red, and those pushing it lower are in blue. The size of each band indicates the magnitude of that feature's impact. You can interact with this plot to select individual predictions and see their unique explanations, or explore the impact of different features.",
        'type': 'html',
        'filename': "force_plot.html"
    })

    # --- 3. Feature Dependence Plots --- #
    vals = np.abs(shap_values_for_plot).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_test.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_features = feature_importance['col_name'].head(3).tolist()

    dependence_plots = []
    for feature in top_features:
        shap.dependence_plot(feature, shap_values_for_plot, X_test_unscaled, interaction_index=None, show=False)
        plt.tight_layout()
        filename_dep = f"dependence_{feature}.png"
        plt.savefig(os.path.join(interp_dir, filename_dep))
        plt.close()
        dependence_plots.append(filename_dep)
    
    plots.append({
        'title': "Feature Dependence (The How)",
        'explanation': "These plots illustrate how the value of a single feature (on the X-axis) affects the SHAP value (on the Y-axis) for that feature, and thus the model's prediction. Each dot represents a data point. The X-axis shows the actual value of the feature, and the Y-axis shows its impact on the prediction. This helps answer questions like: 'Does increasing this feature's value always increase the prediction?' The color of the dots often indicates the value of another feature that the primary feature interacts with, revealing complex relationships within the model.",
        'type': 'image_gallery',
        'filenames': dependence_plots
    })

    return plots
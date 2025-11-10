import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize SHAP for JavaScript plots
shap.initjs()

TREE_BASED_MODELS = ["Random Forest Classifier", "Random Forest Regressor", "XGBoost", "LightGBM", "CatBoost"]

def generate_interpretation(model, X_test, config, interp_dir):
    model_name = config['model']
    prediction_type = config['prediction_type']
    plots = []

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
    shap.summary_plot(shap_values_for_plot, X_test, show=False)
    plt.title("Overall Feature Importance and Impact")
    plt.tight_layout()
    filename_summary = "shap_summary.png"
    plt.savefig(os.path.join(interp_dir, filename_summary))
    plt.close()
    plots.append({
        'title': "Overall Feature Importance (The What)",
        'explanation': "This chart is like a leaderboard for your features. Features at the top have the most impact on predictions. Each dot is a prediction from your data. Red dots mean a high feature value pushed the prediction up; blue dots mean a low feature value pushed it down.",
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
        X_test,
        show=False
    )
    shap.save_html(force_plot_html_path, force_plot)
    plots.append({
        'title': "Individual Prediction Breakdown (The Why)",
        'explanation': "This chart explains predictions for your entire test set, stacked together. The 'base value' is the average prediction. Red arrows show features that pushed a prediction higher; blue arrows pushed it lower. You can use the dropdown to explore different features.",
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
        shap.dependence_plot(feature, shap_values_for_plot, X_test, interaction_index=None, show=False)
        plt.tight_layout()
        filename_dep = f"dependence_{feature}.png"
        plt.savefig(os.path.join(interp_dir, filename_dep))
        plt.close()
        dependence_plots.append(filename_dep)
    
    plots.append({
        'title': "Feature Dependence (The How)",
        'explanation': "These charts show how a single feature's value affects the model's prediction. It helps you answer questions like, 'Does a higher value for this feature always lead to a higher prediction?' The color of the dots shows potential interaction with another feature.",
        'type': 'image_gallery',
        'filenames': dependence_plots
    })

    return plots
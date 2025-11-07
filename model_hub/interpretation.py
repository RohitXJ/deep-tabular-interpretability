import shap
import torch
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

    # --- 1. Global Feature Importance --- #
    plt.figure(figsize=(10, 8))
    if model_name == "TabNet":
        importances = model.feature_importances_
        feature_names = X_test.columns
        sorted_idx = np.argsort(importances)
        
        plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
        plt.title("Global Feature Importance (TabNet)")
        plt.xlabel("Attention Importance")
        plt.tight_layout()
        
        filename = "global_importance.png"
        plt.savefig(os.path.join(interp_dir, filename))
        plt.close()
        
        plots.append({
            'title': "Overall Feature Importance",
            'explanation': "This chart shows how much attention the TabNet model paid to each feature while learning. Features with longer bars were more important to the model's decisions overall.",
            'type': 'image',
            'filename': filename
        })
        return plots # TabNet only provides global importance

    # --- SHAP Analysis for other models --- #
    
    # Select the correct explainer
    if model_name in TREE_BASED_MODELS:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
    
    elif isinstance(model, (torch.nn.Module)):
        background = X_test.sample(n=min(100, len(X_test)), random_state=42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_name == "TabTransformer":
            dl_params = config['dl_params']
            cat_features = dl_params['cat_features']
            num_features = dl_params['num_features']
            background_cat = torch.tensor(background[cat_features].values, dtype=torch.long).to(device)
            background_num = torch.tensor(background[num_features].values, dtype=torch.float32).to(device)
            # Pass inputs as a list of tensors to DeepExplainer
            explainer = shap.DeepExplainer(model, [background_cat, background_num])
            X_test_cat = torch.tensor(X_test[cat_features].values, dtype=torch.long).to(device)
            X_test_num = torch.tensor(X_test[num_features].values, dtype=torch.float32).to(device)
            shap_values = explainer.shap_values([X_test_cat, X_test_num])
        else: # FNN
            background_tensor = torch.tensor(background.values, dtype=torch.float32).to(device)
            explainer = shap.DeepExplainer(model, background_tensor)
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
            shap_values = explainer.shap_values(X_test_tensor)
    
    else: # Linear models, SVM, etc.
        background = shap.sample(X_test, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test)

    # For single output models, shap_values can sometimes be a list with one element.
    # We extract the 2D array from the list if that's the case.
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]

    # For classification, shap_values can be a list of arrays (one per class)
    shap_values_for_plot = shap_values
    if prediction_type == "Classification" and isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_for_plot = shap_values[1]

    # Defensive reshape: If we have a 1D array, it means we likely have SHAP values for a single prediction.
    # Reshape it to 2D to make it compatible with the plotting functions.
    if len(shap_values_for_plot.shape) == 1:
        shap_values_for_plot = np.reshape(shap_values_for_plot, (1, -1))

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
    
    # Correctly handle expected_value for multi-class
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    # Conditionally select SHAP values for the first prediction to avoid indexing errors
    if len(shap_values_for_plot.shape) == 1:
        shap_values_for_first_prediction = shap_values_for_plot
    else:
        shap_values_for_first_prediction = shap_values_for_plot[0,:]

    force_plot = shap.force_plot(expected_value, shap_values_for_first_prediction, X_test.iloc[0,:], show=False)
    shap.save_html(force_plot_html_path, force_plot)
    plots.append({
        'title': "Individual Prediction Breakdown (The Why)",
        'explanation': "This chart explains one single prediction. The 'base value' is the average prediction. Red arrows show features that pushed this prediction higher than the average; blue arrows pushed it lower. Bigger arrows mean a bigger impact.",
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
        shap.dependence_plot(feature, shap_values_for_plot, X_test, show=False)
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
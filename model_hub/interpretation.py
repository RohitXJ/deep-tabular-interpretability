import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize SHAP for JavaScript plots
shap.initjs()

TREE_BASED_MODELS = ["Random Forest Classifier", "Random Forest Regressor", "XGBoost", "LightGBM", "CatBoost"]

def generate_interpretation(model, X_test, X_test_unscaled, y_test, config, interp_dir):
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

    # Subsample data to 1000 rows if it's larger for performance reasons
    if X_test.shape[0] > 1000:
        print(f"Subsampling data from {X_test.shape[0]} to 1000 rows for SHAP analysis.")
        # Ensure y_test is also sliced consistently with X_test
        indices_to_keep = X_test.head(1000).index
        X_test = X_test.loc[indices_to_keep]
        X_test_unscaled = X_test_unscaled.loc[indices_to_keep]
        y_test = y_test.loc[indices_to_keep]


    # Initialize SHAP for JavaScript plots
    shap.initjs()

    # --- SHAP Analysis --- #
    plt.figure(figsize=(10, 8))

    # --- NEW SAMPLING LOGIC & EXPLAINER SELECTION ---
    if prediction_type == "Classification":
        print("Applying stratified sampling for Classification task.")
        y_series = y_test.squeeze()
        n_classes = y_series.nunique()
        
        if n_classes == 2:
            n_per_class = 5
            print(f"Binary classification: selecting {n_per_class} samples per class.")
        else:
            n_total = max(10, n_classes)
            n_per_class = n_total // n_classes
            print(f"Multi-class ({n_classes} classes): selecting {n_per_class} samples per class.")

        # Group by target variable and sample within each group
        # This handles cases where a class has fewer samples than n_per_class
        sample_indices = y_series.groupby(y_series).apply(
            lambda x: x.sample(n=min(n_per_class, len(x)), random_state=42)
        ).index.get_level_values(1)

    else: # Regression
        n_shap_samples = 15
        print(f"Applying random sampling for Regression task: {n_shap_samples} samples.")
        sample_indices = X_test.sample(n=n_shap_samples, random_state=42).index

    # Create the sampled dataframes using the selected indices
    X_test_shap = X_test.loc[sample_indices]
    X_test_unscaled_shap = X_test_unscaled.loc[sample_indices]
    
    print(f"Total samples selected for SHAP analysis: {len(sample_indices)}")

    # Select the correct explainer based on model type
    if model_name in TREE_BASED_MODELS:
        print("Using TreeExplainer for SHAP analysis.")
        explainer = shap.TreeExplainer(model)
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test_shap, check_additivity=False)
    else:  # Linear models, SVM, etc.
        print("Using KernelExplainer for non-tree-based model. This can be slow, applying optimizations.")
        
        # 1. Summarize the background data using k-means for better representation.
        print("Summarizing background data with k-means...")
        background_data = shap.kmeans(X_test, 100) # Use full X_test for a good background
        
        explainer = shap.KernelExplainer(model.predict, background_data)
        print("Calculating SHAP values on a subset of data... (This may still take a moment)")
        shap_values = explainer.shap_values(X_test_shap)

    # After calculations, we use the smaller `_shap` versions for plotting
    X_test = X_test_shap
    X_test_unscaled = X_test_unscaled_shap

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
                'explanation': """
            <strong>What It Is:</strong> This plot provides a rich overview of how every feature impacts the model's output. It combines feature importance with feature effects for every sample in your dataset.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Vertical Axis (Feature Importance):</strong> Features are ranked from most important (top) to least important (bottom). A feature's importance is determined by the average absolute SHAP value across all samples.</li>
                <li><strong>Horizontal Axis (Impact on Prediction):</strong> This is the <strong>SHAP Value</strong>. It represents the feature's contribution to the prediction.
                    <ul>
                        <li>For <strong>Regression</strong>, positive values mean the feature pushed the prediction higher (e.g., a higher house price), while negative values pushed it lower.</li>
                        <li>For <strong>Classification</strong>, positive values pushed the prediction towards the positive class (e.g., 'Yes', 'True', or 1), while negative values pushed it towards the negative class.</li>
                    </ul>
                </li>
                <li><strong>Dot Color (Feature Value):</strong> The color shows if a feature's value was high or low for a specific data point.
                    <ul>
                        <li><span style="color:red;">■ Red dots</span> represent <strong>high values</strong> of a feature.</li>
                        <li><span style="color:blue;">■ Blue dots</span> represent <strong>low values</strong> of a feature.</li>
                        <li><strong>Black dots</strong> typically represent instances where the feature's value was <strong>missing (NaN)</strong>. The plot shows the impact this missingness has on the prediction.</li>
                    </ul>
                </li>
            </ul>
            <strong>Putting It Together:</strong> For a house price prediction, if the 'sqft_living' feature has many red dots on the right (positive SHAP values), it means that larger living areas strongly increase the predicted price.
        """,
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
                'explanation': """
            <strong>What It Is:</strong> This interactive plot visualizes the forces driving a single prediction. It provides a detailed look at how each feature's value contributes to the final model output.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Base Value (E[f(x)]):</strong> This is the starting point, representing the average model prediction over the entire dataset. It's the baseline before considering the features of this specific data point.</li>
                <li><strong>Feature Contributions:</strong>
                    <ul>
                        <li><span style="color:red;">■ Red arrows/blocks</span> represent features that <strong>pushed the prediction higher</strong> than the base value.</li>
                        <li><span style="color:blue;">■ Blue arrows/blocks</span> represent features that <strong>pushed the prediction lower</strong>.</li>
                        <li>The <strong>size</strong> of the block corresponds to the magnitude of that feature's impact.</li>
                    </ul>
                </li>
                <li><strong>Output Value (f(x)):</strong> This bold number is the <strong>final prediction</strong> for this specific data point. It's the result of the base value plus the sum of all feature contributions.
                    <ul>
                        <li>For <strong>Regression</strong>, this is the predicted value (e.g., a price of $350,000).</li>
                        <li>For <strong>Classification</strong>, this is the raw model output in log-odds. A positive value corresponds to a prediction for the positive class, while a negative value corresponds to the negative class.</li>
                    </ul>
                </li>
            </ul>
            You can interact with the plot to explore different predictions and see how features impact each one.
        """,
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
                'explanation': """
            <strong>What It Is:</strong> These plots show how a single feature's value affects its own SHAP value, and thus the model's prediction. It helps to uncover more complex relationships like non-linearity and interactions.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Horizontal Axis:</strong> This shows the actual, unscaled value of the feature across all data points.</li>
                <li><strong>Vertical Axis:</strong> This shows the <strong>SHAP value</strong> for that feature. A positive SHAP value means that feature's value pushed the prediction higher, while a negative SHAP value pushed it lower.</li>
                <li><strong>Each dot</strong> is a single prediction from the dataset.</li>
                <li><strong>Vertical Color (Interaction):</strong> The color of the dots often represents the value of a second feature that has the strongest interaction with the feature being plotted. This helps reveal how two features work together to influence the prediction. For example, it might show that the age of a house only has a strong negative impact on price if the house is also in poor condition.</li>
            </ul>
            <strong>Putting It Together:</strong> If the plot for 'Age' shows that as the value on the x-axis increases, the SHAP values on the y-axis trend downwards, it indicates that older items are generally predicted to have a lower value.
        """,
        'type': 'image_gallery',
        'filenames': dependence_plots
    })

    # --- 4. Local Explanation (Waterfall Plot for a Random Prediction) --- #
    try:
        # Select a random sample for the waterfall plot, ensuring there's data to sample from.
        if not X_test_unscaled.empty:
            random_index = np.random.randint(0, X_test_unscaled.shape[0])
            
            plt.figure()
            # Create an explanation object for the randomly selected instance
            explanation_for_waterfall = shap.Explanation(
                values=shap_values_for_plot[random_index],
                base_values=base_value_for_plot,
                data=X_test_unscaled.iloc[random_index].values,
                feature_names=X_test_unscaled.columns.tolist()
            )
            shap.plots.waterfall(explanation_for_waterfall, max_display=20, show=False)
            plt.title(f"Breakdown of Prediction for Sample #{random_index}")
            plt.tight_layout()
            filename_waterfall = "shap_waterfall.png"
            plt.savefig(os.path.join(interp_dir, filename_waterfall))
            plt.close()
            plots.append({
                'title': "Single Prediction Explained (Waterfall)",
                'explanation': """
            <strong>What It Is:</strong> This waterfall plot provides a detailed, step-by-step breakdown of how a single prediction was made. It shows how each feature's contribution moves the prediction from the baseline to the final result.
            <br><br>
            <strong>How to Read It:</strong>
            <ul>
                <li><strong>Base Value (E[f(x)]):</strong> This is the starting point at the bottom, representing the average prediction over the entire dataset (e.g., the average house price).</li>
                <li><strong>Feature Contributions:</strong> Each bar represents a feature's impact on this single prediction.
                    <ul>
                        <li><span style="color:red;">■ Red bars</span> show features that <strong>pushed the prediction higher</strong>. The number next to the feature name is its actual, unscaled value (e.g., `sqft_living = 2100`).</li>
                        <li><span style="color:blue;">■ Blue bars</span> show features that <strong>pushed the prediction lower</strong>.</li>
                        <li>The <strong>length</strong> of the bar shows the magnitude of the feature's impact.</li>
                    </ul>
                </li>
                <li><strong>Final Prediction (f(x)):</strong> This is the final model output at the top, which is the sum of the base value and all the individual feature contributions.
                    <ul>
                        <li>For <strong>Regression</strong>, this is the final predicted value for this one data point.</li>
                        <li>For <strong>Classification</strong>, this is the log-odds output for this data point. A positive value means the model predicts the positive class.</li>
                    </ul>
                </li>
            </ul>
        """,
                'type': 'image',
                'filename': filename_waterfall
            })
    except Exception as e:
        print(f"Could not generate waterfall plot: {e}")

    return plots
"""
================================
Type 2: SHAP Interpretation Code
================================

This file contains all SHAP interpretation code blocks from the notebook.
It is designed to be run *after* the models and data variables have been
created and loaded in your environment.

User: Gemini CLI
Project: Deep Learning Interpretability

This script assumes the following variables are already defined:

For Regression:
- model1_reg (trained ANN_Shallow_Regression)
- model2_reg (trained ANN_Deep_Regression)
- background_data_reg_t (tensor sample, e.g., X_train_t_reg[:100])
- X_test_scaled_reg (numpy array of test features)
- X_test_t_reg (tensor of test features)
- features_reg (list of feature names)

For Classification:
- model1_class (trained ANN_Shallow_Classification)
- model2_class (trained ANN_Deep_Classification)
- background_data_class_t (tensor sample, e.g., X_train_t_class[:100])
- X_test_scaled_class (numpy array of test features)
- X_test_t_class (tensor of test features)
- features_class (list of feature names)
"""

# 1. Required Imports and Setup
import torch
import numpy as np
import shap

# This is crucial for SHAP plots to render in a notebook environment
shap.initjs()

# ----------------------------------------------------------------------
# MODULE 1: REGRESSION INTERPRETATIONS
# ----------------------------------------------------------------------

print("--- Starting Regression Model 1 (Shallow) SHAP Analysis ---")

# --- Regression Model 1: SHAP Calculation ---
model1_reg.eval()
explainer_reg1 = shap.DeepExplainer(model1_reg, background_data_reg_t)
print("Calculating SHAP values for Regression Model 1...")
shap_values_reg1 = explainer_reg1.shap_values(X_test_t_reg)
print("Calculation complete.")


# --- Regression Model 1: Global Feature Importance (Summary Plot) ---
print("Plotting Model 1: Global Feature Importance")
shap.summary_plot(shap_values_reg1, X_test_scaled_reg, feature_names=features_reg)


# --- Regression Model 1: Local Interpretation (Waterfall Plot) ---
print("Plotting Model 1: Local Interpretation (Sample 0)")
# We must .squeeze() the values for the 1st sample to make them 1D for the plot
exp_reg1 = shap.Explanation(
    values=shap_values_reg1[0].squeeze(),
    base_values=explainer_reg1.expected_value[0],
    data=X_test_scaled_reg[0],
    feature_names=features_reg
)
shap.plots.waterfall(exp_reg1)


print("\n--- Starting Regression Model 2 (Deep) SHAP Analysis ---")

# --- Regression Model 2: SHAP Calculation ---
model2_reg.eval()
explainer_reg2 = shap.DeepExplainer(model2_reg, background_data_reg_t)
print("Calculating SHAP values for Regression Model 2...")
shap_values_reg2 = explainer_reg2.shap_values(X_test_t_reg)
print("Calculation complete.")


# --- Regression Model 2: Global Feature Importance (Summary Plot) ---
print("Plotting Model 2: Global Feature Importance")
shap.summary_plot(shap_values_reg2, X_test_scaled_reg, feature_names=features_reg)


# --- Regression Model 2: Local Interpretation (Waterfall Plot) ---
print("Plotting Model 2: Local Interpretation (Sample 0)")
exp_reg2 = shap.Explanation(
    values=shap_values_reg2[0].squeeze(),
    base_values=explainer_reg2.expected_value[0],
    data=X_test_scaled_reg[0],
    feature_names=features_reg
)
shap.plots.waterfall(exp_reg2)


# ----------------------------------------------------------------------
# MODULE 2: CLASSIFICATION INTERPRETATIONS
# ----------------------------------------------------------------------

print("\n--- Starting Classification Model 1 (Shallow) SHAP Analysis ---")

# --- Classification Model 1: SHAP Calculation ---
model1_class.eval()
explainer_class1 = shap.DeepExplainer(model1_class, background_data_class_t)
print("Calculating SHAP values for Classification Model 1...")
# .shap_values() returns a LIST. We select the 1st (and only) class output [0].
# .squeeze() removes the extra dimension, e.g. (114, 30, 1) -> (114, 30)
# np.atleast_2d() ensures it's a 2D matrix (fixes summary_plot AssertionError)
raw_shap_values_c1 = explainer_class1.shap_values(X_test_t_class)[0].squeeze()
shap_values_class1 = np.atleast_2d(raw_shap_values_c1)
print("Calculation complete.")


# --- Classification Model 1: Global Feature Importance (Summary Plot) ---
print("Plotting Model 1: Global Feature Importance")
# We must slice the feature data to match the number of rows in the SHAP data
# This fixes the row count mismatch AssertionError
n_rows_shap_c1 = shap_values_class1.shape[0]
features_for_plot_c1 = X_test_scaled_class[:n_rows_shap_c1]
shap.summary_plot(shap_values_class1, features_for_plot_c1, feature_names=features_class)


# --- Classification Model 1: Local Interpretation (Waterfall Plot) ---
print("Plotting Model 1: Local Interpretation (Sample 0)")
# We select the first sample [0], which is now correctly 1D
exp_class1 = shap.Explanation(
    values=shap_values_class1[0],
    base_values=explainer_class1.expected_value[0],
    data=X_test_scaled_class[0],
    feature_names=features_class
)
shap.plots.waterfall(exp_class1)


print("\n--- Starting Classification Model 2 (Deep) SHAP Analysis ---")

# --- Classification Model 2: SHAP Calculation ---
model2_class.eval()
explainer_class2 = shap.DeepExplainer(model2_class, background_data_class_t)
print("Calculating SHAP values for Classification Model 2...")
# Apply the same robust fixes as Model 1
raw_shap_values_c2 = explainer_class2.shap_values(X_test_t_class)[0].squeeze()
shap_values_class2 = np.atleast_2d(raw_shap_values_c2)
print("Calculation complete.")


# --- Classification Model 2: Global Feature Importance (Summary Plot) ---
print("Plotting Model 2: Global Feature Importance")
# Apply the same row-slicing fix
n_rows_shap_c2 = shap_values_class2.shape[0]
features_for_plot_c2 = X_test_scaled_class[:n_rows_shap_c2]
shap.summary_plot(shap_values_class2, features_for_plot_c2, feature_names=features_class)


# --- Classification Model 2: Local Interpretation (Waterfall Plot) ---
print("Plotting Model 2: Local Interpretation (Sample 0)")
exp_class2 = shap.Explanation(
    values=shap_values_class2[0],
    base_values=explainer_class2.expected_value[0],
    data=X_test_scaled_class[0],
    feature_names=features_class
)
shap.plots.waterfall(exp_class2)

print("\n--- SHAP Analysis Complete ---")
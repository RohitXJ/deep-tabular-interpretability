### [APPEND | 2025-11-26 | model_hub/__init__.py]

- What this file does: Aggregates and exposes core Machine Learning (ML) model functions from individual modules within the `model_hub` package.
- What logic is implemented: Imports specific ML model functions (`ML_model_eval`, `ML_models_call`, `ML_model_train`) from other files within the `model_hub` directory. Defines `__all__` to control what symbols are exported when `*` is used in an import statement.
- What algorithms or techniques are used: Python package initialization and modular programming.
- What inputs it takes: N/A (serves as an aggregator).
- What outputs it produces: Makes core ML model functionalities available for external use.
- How it connects to other files: Imports from `evaluation.py`, `model_init.py`, and `training.py`. It is imported by `app/routes.py` to access ML model functionalities.

### [APPEND | 2025-11-26 | model_hub/dl_evaluation.py]

- What this file does: Provides functions to evaluate Deep Learning (DL) models for both regression and classification tasks, generating a formatted report with qualitative summaries.
- What logic is implemented:
    - `_generate_regression_summary`: Creates a qualitative summary of regression model performance based on R-squared and MAPE values.
    - `_generate_dl_classification_summary`: Creates a qualitative summary of classification model performance based on accuracy and F1-score.
    - `DL_model_eval`:
        - Sets the model to evaluation mode (`model.eval()`).
        - Makes predictions on the test data.
        - For regression tasks: Calculates MAE, MSE, RMSE, R-squared, and MAPE. Generates a regression performance report.
        - For classification tasks: Calculates accuracy, precision, recall, and F1-score. Generates a classification performance report.
- What algorithms or techniques are used:
    - Regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared, Mean Absolute Percentage Error (MAPE).
    - Classification metrics: Accuracy, Precision, Recall, F1-Score.
    - PyTorch for model inference.
- What inputs it takes:
    - `model` (torch.nn.Module): The trained Deep Learning model.
    - `test_data` (list): A list containing `X_test_t` (test features as PyTorch tensor) and `y_test_t` (test targets as PyTorch tensor).
    - `prediction_type` (str): "Regression" or "Classification".
- What outputs it produces: A formatted string containing the evaluation metrics and a qualitative summary of the model's performance.
- How it connects to other files: Used by `app/routes.py` to evaluate DL models after training.

### [APPEND | 2025-11-26 | model_hub/dl_interpretation.py]

- What this file does: Generates SHAP (SHapley Additive exPlanations) interpretation plots and explanations specifically for Deep Learning models. It also includes a function to plot training loss.
- What logic is implemented:
    - `generate_dl_interpretation`:
        - Subsamples data for SHAP analysis if the dataset is too large, using stratified sampling for classification and random sampling for regression.
        - Loads encoders and feature names to handle categorical and numerical features correctly.
        - Initializes a `shap.DeepExplainer` with the DL model and background data.
        - Calculates SHAP values for the subsampled test data.
        - Generates and saves three types of SHAP plots:
            - **Global Feature Importance (Bar Plot):** Shows overall feature impact.
            - **Global Summary Plot (Beeswarm):** Illustrates how feature values affect prediction output across samples, including direction (positive/negative impact) and magnitude.
            - **Local Interpretation (Waterfall Plot):** Explains a single prediction by showing how each feature contributes to pushing the prediction from the expected base value to the final output.
        - Provides detailed textual explanations for each plot.
    - `plot_training_loss`: Generates and saves a line plot of the training loss over epochs to visualize model convergence.
- What algorithms or techniques are used:
    - SHAP (SHapley Additive exPlanations) with `DeepExplainer` for model interpretability.
    - Stratified sampling (for classification) and random sampling (for regression).
    - `matplotlib` for plot generation.
- What inputs it takes:
    - `model` (torch.nn.Module): The trained Deep Learning model.
    - `X_test_t`: Test features as PyTorch tensor.
    - `X_test_scaled_np`, `X_test_unscaled_np`: Scaled and unscaled test features as NumPy arrays.
    - `features`: List of feature names.
    - `prediction_type` (str): "Regression" or "Classification".
    - `interp_dir` (str): Directory to save interpretation artifacts.
    - `background_data_t`: Background data for `DeepExplainer` (PyTorch tensor).
    - `y_test_np`: Test targets as NumPy array.
    - `loss_history` (list, for `plot_training_loss`): List of loss values per epoch.
    - `model_name` (str, for `plot_training_loss`): Name of the model.
    - `plot_path` (str, for `plot_training_loss`): Path to save the loss plot.
- What outputs it produces:
    - `generate_dl_interpretation`: A list of dictionaries containing metadata for generated plots (filename, title, description, type). Image files (PNG) for SHAP plots saved to `interp_dir`.
    - `plot_training_loss`: A saved image file (PNG) of the training loss plot.
- How it connects to other files: Used by `app/routes.py` to generate and display DL model interpretations and training loss plots.

### [APPEND | 2025-11-26 | model_hub/dl_model_init.py]

- What this file does: Initializes and returns instances of Deep Learning (DL) models based on specified prediction type and model name.
- What logic is implemented:
    - Acts as a factory for ANN (Artificial Neural Network) architectures.
    - Based on `type` ("Classification" or "Regression") and `model` ("Shallow ANN" or "Deep ANN"), it instantiates the corresponding PyTorch `nn.Module` class (`ANN_Shallow_Regression`, `ANN_Deep_Regression`, `ANN_Shallow_Classification`, `ANN_Deep_Classification`).
- What algorithms or techniques are used: PyTorch for neural network definitions.
- What inputs it takes:
    - `type` (str): "Classification" or "Regression".
    - `model` (str): "Shallow ANN" or "Deep ANN".
    - `input_shape` (int): Number of input features for the model.
- What outputs it produces: An instance of a `torch.nn.Module` representing the specified DL model.
- How it connects to other files: Imports ANN architectures from `ANN_architecture.py`. Used by `app/routes.py` to initialize DL models for training.

### [APPEND | 2025-11-26 | model_hub/dl_training.py]

- What this file does: Handles the training process for Deep Learning (DL) models.
- What logic is implemented:
    - Initializes the loss function (`nn.MSELoss` for regression, `nn.BCELoss` for binary classification).
    - Sets up the Adam optimizer.
    - Iterates through a specified number of epochs, performing forward and backward passes, calculating loss, and updating model parameters.
    - Logs training progress (loss per epoch).
- What algorithms or techniques are used:
    - PyTorch for model training.
    - Mean Squared Error (MSE) loss for regression.
    - Binary Cross-Entropy (BCE) loss for classification.
    - Adam optimizer for weight updates.
- What inputs it takes:
    - `model` (torch.nn.Module): The PyTorch model to train.
    - `train_loader` (DataLoader): DataLoader for the training data.
    - `prediction_type` (str): "Classification" or "Regression".
    - `epochs` (int): Number of training epochs.
- What outputs it produces: A tuple containing the trained PyTorch model (`nn.Module`) and a list of loss values (`list`) recorded per epoch.
- How it connects to other files: Used by `app/routes.py` to train DL models.

### [APPEND | 2025-11-26 | model_hub/evaluation.py]

- What this file does: Provides functions to evaluate Machine Learning (ML) models for both regression and classification tasks, generating a formatted report with qualitative summaries.
- What logic is implemented:
    - `_generate_regression_summary`: Creates a qualitative summary of regression model performance based on R-squared and MAPE values.
    - `ML_model_eval`:
        - Makes predictions on the test data using the trained ML model.
        - For classification tasks: Generates a `classification_report` (including accuracy, precision, recall, F1-score) and provides a qualitative summary based on overall accuracy and F1-score.
        - For regression tasks: Calculates MAE, MSE, RMSE, R-squared, and MAPE. Provides a qualitative summary based on R-squared and MAPE.
- What algorithms or techniques are used:
    - Classification metrics: `classification_report`, accuracy, precision, recall, F1-score.
    - Regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared, Mean Absolute Percentage Error (MAPE).
- What inputs it takes:
    - `model` (object): The trained scikit-learn compatible ML model.
    - `test_data` (list): A list containing `X_test` (test features) and `y_test` (test targets).
    - `type` (str): "Classification" or "Regression".
- What outputs it produces: Prints a formatted string containing the evaluation metrics and a qualitative summary of the model's performance to the console (or captured stdout).
- How it connects to other files: Used by `app/routes.py` to evaluate ML models after training.

### [APPEND | 2025-11-26 | model_hub/interpretation.py]

- What this file does: Generates SHAP (SHapley Additive exPlanations) interpretation plots and explanations for traditional Machine Learning (ML) models.
- What logic is implemented:
    - `generate_interpretation`:
        - Subsamples data if the number of features exceeds a configured limit or if the number of samples is too large for SHAP analysis, optimizing for performance.
        - Selects the appropriate SHAP explainer (`TreeExplainer` for tree-based models, `KernelExplainer` for others) and calculates SHAP values.
        - Samples instances for SHAP analysis, using stratified sampling for classification and random sampling for regression.
        - Generates and saves various SHAP plots:
            - **Overall Feature Importance (Beeswarm Plot):** Shows global feature importance and the impact of feature values.
            - **Individual Prediction Breakdown (Force Plot):** Provides an interactive HTML visualization of how features contribute to a single prediction.
            - **Feature Dependence Plots:** Illustrates the relationship between a feature's value and its impact on the prediction for top features.
            - **Local Explanation (Waterfall Plot):** Offers a detailed, static breakdown of a single prediction.
        - Provides rich textual explanations for each plot to aid interpretation.
- What algorithms or techniques are used:
    - SHAP (SHapley Additive exPlanations) with `TreeExplainer` and `KernelExplainer`.
    - Data subsampling techniques.
    - `matplotlib` for plot generation.
- What inputs it takes:
    - `model` (object): The trained ML model.
    - `X_test`, `X_test_unscaled`: Scaled and unscaled test features as pandas DataFrames.
    - `y_test`: Test targets as pandas Series.
    - `config` (dict): Application configuration, including `max_interpretation_features`.
    - `interp_dir` (str): Directory to save interpretation artifacts.
- What outputs it produces: A list of dictionaries containing metadata for generated plots (title, explanation, type, filename/urls). Image files (PNG) for SHAP plots and an HTML file for the force plot are saved to `interp_dir`.
- How it connects to other files: Used by `app/routes.py` to generate and display ML model interpretations.

### [APPEND | 2025-11-26 | model_hub/model_init.py]

- What this file does: Initializes and returns instances of various Machine Learning (ML) models based on specified prediction type and model name.
- What logic is implemented:
    - Acts as a factory for scikit-learn compatible ML models.
    - Based on `type` ("Classification" or "Regression") and `model` (e.g., "Linear Regression", "Random Forest Classifier"), it instantiates the corresponding model class.
    - Handles specific model initialization parameters like `verbose` for `XGBoost` and `CatBoost` to suppress excessive output.
- What algorithms or techniques are used:
    - Linear Models: `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`.
    - Ensemble Models: `RandomForestRegressor`, `RandomForestClassifier`, `XGBRegressor`, `XGBClassifier`, `LGBMRegressor`, `LGBMClassifier`, `CatBoostRegressor`, `CatBoostClassifier`.
    - Support Vector Machines: `SVC`.
- What inputs it takes:
    - `type` (str): "Classification" or "Regression".
    - `model` (str): Name of the specific ML model.
- What outputs it produces: An instance of the specified ML model.
- How it connects to other files: Imported by `model_hub/__init__.py` and used by `app/routes.py` to initialize ML models for training.

### [APPEND | 2025-11-26 | model_hub/training.py]

- What this file does: Provides a generic function to train Machine Learning (ML) models.
- What logic is implemented: Takes a model object and training data (`X_train`, `y_train`) and calls the `fit` method of the model.
- What algorithms or techniques are used: Supervised learning model training (fitting).
- What inputs it takes:
    - `model` (object): An initialized ML model with a `.fit()` method.
    - `data` (list): A list containing `X_train` (training features) and `y_train` (training targets).
- What outputs it produces: The trained model object.
- How it connects to other files: Imported by `model_hub/__init__.py` and used by `app/routes.py` to train ML models.

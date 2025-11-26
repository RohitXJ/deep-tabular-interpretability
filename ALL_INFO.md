### [APPEND | 2025-11-26 | Project File Structure]

- What this section does: Provides a hierarchical overview of the project's directory and file organization.
- What logic is implemented: Displays the tree structure of files and folders within the project.
- What algorithms or techniques are used: N/A (static content).
- What inputs it takes: N/A.
- What outputs it produces: A visual representation of the project's layout.
- How it connects to other files: Illustrates the location and relationships of all described files in the project.

```
F:\AI ENGINEER LEVEL MAP\HIGH LEVEL\deep-tabular-interpretability\
├───.gitattributes
├───.gitignore
├───ANN_architecture.py
├───APP.md
├───config.py
├───DATA_PROCESS.md
├───GEMINI.md
├───INFO.md
├───LICENSE
├───main.py
├───MODEL_HUB.md
├───requirements.txt
├───run.py
├───__pycache__\
├───.git\...
├───.venv\...
├───ANN_archs\
│   ├───Classification_Deep_ANN.svg
│   ├───Classification_Shallow_ANN.svg
│   ├───Regression_Deep_ANN.svg
│   └───Regression_Shallow_ANN.svg
├───app\
│   ├───__init__.py
│   ├───forms.py
│   ├───routes.py
│   ├───__pycache__\
│   ├───static\
│   │   └───images\...
│   └───templates\
│       ├───1_upload.html
│       ├───2_configure_model.html
│       ├───3_configure_data.html
│       ├───4_results.html
│       ├───5_interpretation.html
│       ├───6_dl_interpretation.html
│       └───base.html
├───data_process\
│   ├───__init__.py
│   ├───encode_categorical.py
│   ├───feature_selection.py
│   ├───handle_imbalance.py
│   ├───handle_missing_values.py
│   ├───load_dataset.py
│   ├───scale_numeric.py
│   ├───split_dataset.py
│   ├───TO-DO.md
│   └───__pycache__\
├───instance\
│   └───interpretations\...
├───model_hub\
│   ├───__init__.py
│   ├───dl_evaluation.py
│   ├───dl_interpretation.py
│   ├───dl_model_init.py
│   ├───dl_training.py
│   ├───evaluation.py
│   ├───interpretation.py
│   ├───model_init.py
│   ├───TO-DO.md
│   ├───training.py
│   └───__pycache__\
├───Notebooks\...
├───Test_data\
│   ├───Cleaned.csv
│   ├───dataset_.csv
│   ├───house_price_regression_dataset.csv
│   ├───taxi_trip_pricing.csv
│   └───tested.csv
└───uploads\...
```

### [APPEND | 2025-11-26 | run.py]

- What this file does: This is the entry point for running the Flask web application.
- What logic is implemented: It imports the `create_app` function from the `app` package, creates an application instance, and then runs the development server if the script is executed directly.
- What algorithms or techniques are used: Flask application initialization and execution.
- What inputs it takes: None directly, relies on Flask's internal mechanisms and configured `app` settings.
- What outputs it produces: Starts the Flask web server, making the application accessible.
- How it connects to other files: Imports `create_app` from `app/__init__.py`.

### [APPEND | 2025-11-26 | requirements.txt]

- What this file does: Lists all the Python package dependencies required to run the project.
- What logic is implemented: Specifies package names and their versions (implicitly, if not explicitly stated).
- What algorithms or techniques are used: Standard Python dependency management.
- What inputs it takes: None (it's a static list of dependencies).
- What outputs it produces: Used by `pip` to install project dependencies (`pip install -r requirements.txt`).
- How it connects to other files: Defines the external libraries that the `app/`, `data_process/`, `model_hub/` modules, and `run.py` rely on.

APP DIR STARTS

### [APPEND | 2025-11-26 | app/__init__.py]

- What this file does: Initializes and configures the Flask application.
- What logic is implemented:
    - Creates a Flask app instance.
    - Loads configuration from `config.py`.
    - Ensures instance folder and upload folder exist.
    - Registers blueprints (specifically, `app.routes`).
- What algorithms or techniques are used: Flask application factory pattern.
- What inputs it takes: Optional `config_class` for configuration.
- What outputs it produces: A configured Flask application instance.
- How it connects to other files: Imports `Flask` from `flask`, `Config` from `config`, and `routes` from `app`. It sets up the upload folder based on `app.config['UPLOAD_FOLDER']` and registers the `routes` blueprint.

### [APPEND | 2025-11-26 | app/forms.py]

- What this file does: Defines various forms used in the web application for user input.
- What logic is implemented:
    - `UploadForm`: For uploading CSV files.
    - `ModelConfigureForm`: For configuring model domain (ML/DL), prediction type (Classification/Regression), selecting specific models, setting epochs, and choosing hyperparameter mode.
    - `DataConfigureForm`: For selecting the target column and configuring feature selection.
- What algorithms or techniques are used: Flask-WTF for form handling, WTForms for field definitions, and WTForms validators for input validation.
- What inputs it takes: User input via web forms.
- What outputs it produces: Validated form data.
- How it connects to other files: Imported by `app/routes.py` to render and process user input from forms in HTML templates.

### [APPEND | 2025-11-26 | app/routes.py]

- What this file does: Handles all the routing logic for the web application, managing user flow from file upload to model interpretation.
- What logic is implemented:
    - **File Upload (`/`):** Manages CSV file uploads, saves them with unique IDs, and stores file paths in the session.
    - **Model Configuration (`/configure_model`):** Allows users to select ML/DL domain, prediction type, specific models, hyperparameters (epochs for DL), and dynamically updates model choices based on selections.
    - **Data Configuration (`/configure_data`):** Enables selection of target column and feature selection method (auto/manual). Triggers feature analysis to populate feature selection options.
    - **Model Training & Evaluation (`/run_analysis`):** Orchestrates the full ML/DL pipeline:
        - Loads and preprocesses data (missing values, feature selection, splitting, encoding, scaling, imbalance handling).
        - Initializes and trains selected ML or DL models.
        - Evaluates models and generates performance reports.
        - Saves trained models, data, and configuration for interpretation.
        - Handles cleanup of temporary files.
    - **Model Interpretation (`/interpretation/<session_id>`):** Loads saved models and data to generate interpretation plots (e.g., SHAP, feature importance) and display model architecture visualizations for DL models. Cleans up interpretation artifacts after display.
    - **Dynamic Model Choices (`/models/<domain>/<prediction_type>`):** API endpoint to provide model options based on domain and prediction type.
- What algorithms or techniques are used:
    - Flask routing and session management.
    - `werkzeug.utils.secure_filename` for secure file handling.
    - Data preprocessing techniques (missing value imputation, categorical encoding, feature scaling, imbalanced data handling).
    - Feature selection using `feature_search`.
    - ML model training and evaluation (via `ML_models_call`, `ML_model_train`, `ML_model_eval`).
    - DL model initialization, training, and evaluation (via `dl_model_init`, `dl_training`, `dl_evaluation`).
    - Model interpretation techniques (via `interpretation` and `dl_interpretation`).
    - `joblib` and `torch` for saving/loading models and data.
    - `pandas` for data manipulation, `numpy` for numerical operations, `matplotlib` for plotting.
    - Bootstrap for UI.
- What inputs it takes:
    - CSV file uploads.
    - User selections from web forms (domain, prediction type, model, epochs, hyperparameter mode, target column, feature selection).
    - JSON requests for dynamic model choices and feature analysis.
- What outputs it produces:
    - Rendered HTML templates (for different stages of the workflow).
    - Redirects to other routes.
    - JSON responses (for dynamic model lists, feature analysis results, and training results).
    - Saved model files, processed data, configuration files, and interpretation plots.
- How it connects to other files:
    - Imports `UploadForm`, `ModelConfigureForm`, `DataConfigureForm` from `app.forms`.
    - Imports various data processing functions from `data_process` module.
    - Imports various model hub functions from `model_hub` module.
    - Imports `create_pytorch_tensors_and_dataloaders` from `ANN_architecture`.
    - Uses configuration from `current_app.config`.
    - Renders HTML templates located in `app/templates/`.
    - Saves interpretation results to `instance/interpretations/`.
    - Uses model architecture SVG files from `ANN_archs/`.

### [APPEND | 2025-11-26 | app/templates/1_upload.html]

- What this file does: Provides the user interface for uploading a CSV dataset.
- What logic is implemented:
    - Displays a file input field for CSV files and a submit button.
    - Includes basic client-side JavaScript to show a loading spinner when the form is submitted.
- What algorithms or techniques are used: Jinja2 templating, HTML form handling, Bootstrap for styling, basic JavaScript.
- What inputs it takes: A CSV file from the user.
- What outputs it produces: Submits the CSV file to the `/` route.
- How it connects to other files: Extends `base.html`. Renders the `UploadForm` defined in `app/forms.py`. Submits data to the `upload` function in `app/routes.py`.

### [APPEND | 2025-11-26 | app/templates/2_configure_model.html]

- What this file does: Provides the user interface for configuring model selection parameters.
- What logic is implemented:
    - Displays dropdowns for model domain (ML/DL), prediction type (Classification/Regression), and specific model.
    - Dynamically updates the available model choices based on the selected domain and prediction type using a JavaScript fetch request to `/models/<domain>/<prediction_type>`.
    - Conditionally shows/hides the "Epochs" field if a Deep Learning (DL) model domain is selected.
- What algorithms or techniques are used: Jinja2 templating, HTML form handling, Bootstrap for styling, JavaScript for dynamic updates.
- What inputs it takes: User selections for model domain, prediction type, model name, and epochs (for DL).
- What outputs it produces: Submits the model configuration to the `/configure_model` route.
- How it connects to other files: Extends `base.html`. Renders the `ModelConfigureForm` defined in `app/forms.py`. Interacts with the `get_models` function in `app/routes.py` via JavaScript. Submits data to the `configure_model` function in `app/routes.py`.

### [APPEND | 2025-11-26 | app/templates/3_configure_data.html]

- What this file does: Provides the user interface for configuring data-related parameters, specifically target column and feature selection.
- What logic is implemented:
    - Displays dropdowns for selecting the target column and the number of features to select.
    - Triggers a JavaScript fetch request to `/run_feature_analysis` when the target column is changed.
    - Dynamically displays a feature importance plot and populates the feature selection dropdown based on the analysis results.
- What algorithms or techniques are used: Jinja2 templating, HTML form handling, Bootstrap for styling, JavaScript for dynamic updates and AJAX calls.
- What inputs it takes: User selections for the target column and the number of features to select.
- What outputs it produces: Submits the data configuration to the `/configure_data` route. Displays feature importance plot.
- How it connects to other files: Extends `base.html`. Renders the `DataConfigureForm` defined in `app/forms.py`. Interacts with the `run_feature_analysis` function in `app/routes.py` via JavaScript. Submits data to the `configure_data` function in `app/routes.py`.

### [APPEND | 2025-11-26 | app/templates/4_results.html]

- What this file does: Displays the results of the model training and evaluation.
- What logic is implemented:
    - Fetches the analysis results from the `/run_analysis` route via an AJAX call.
    - Displays information about the number of features used.
    - Shows the model evaluation report in a `<pre>` tag.
    - Conditionally displays a training loss plot for Deep Learning models.
    - Provides a link to view the model interpretation.
- What algorithms or techniques are used: Jinja2 templating, HTML structure, Bootstrap for styling, JavaScript for AJAX calls to fetch and display dynamic content.
- What inputs it takes: The response JSON from the `/run_analysis` endpoint.
- What outputs it produces: Rendered model evaluation report, feature information, optional loss plot, and a link to the interpretation page.
- How it connects to other files: Extends `base.html`. Fetches data from the `run_analysis` function in `app/routes.py`. Provides a link to the `show_interpretation` function in `app/routes.py`.

### [APPEND | 2025-11-26 | app/templates/5_interpretation.html]

- What this file does: Displays model interpretation plots and explanations for Machine Learning (ML) models.
- What logic is implemented:
    - Iterates through `plot_data` to display various interpretation plots (images, HTML content like SHAP force plots, image galleries for dependence plots).
    - Includes explanations for each plot to guide user understanding.
    - Conditionally displays a legend for categorical features if available.
- What algorithms or techniques are used: Jinja2 templating, HTML structure, Bootstrap for styling.
- What inputs it takes: `plot_data` (list of dictionaries containing plot details) and `legend_data` (dictionary for categorical feature mapping) passed from `app/routes.py`.
- What outputs it produces: Visualizations and textual explanations of model behavior.
- How it connects to other files: Extends `base.html`. Receives `plot_data` and `legend_data` from the `show_interpretation` function in `app/routes.py`.

### [APPEND | 2025-11-26 | app/templates/6_dl_interpretation.html]

- What this file does: Displays model interpretation plots and explanations specifically for Deep Learning (DL) models.
- What logic is implemented:
    - Iterates through `plot_data` to display various interpretation plots (images, HTML content like SHAP force plots, image galleries).
    - Includes explanations for each plot.
    - Conditionally displays a section for model architecture visualization and documentation.
- What algorithms or techniques are used: Jinja2 templating, HTML structure, Bootstrap for styling.
- What inputs it takes: `plot_data` (list of dictionaries containing plot details) and `architecture_data` (dictionary for architecture image and documentation) passed from `app/routes.py`.
- What outputs it produces: Visualizations and textual explanations of DL model behavior, including architecture.
- How it connects to other files: Extends `base.html`. Receives `plot_data` and `architecture_data` from the `show_interpretation` function in `app/routes.py`.

### [APPEND | 2025-11-26 | app/templates/base.html]

- What this file does: Defines the base structure and styling for all other HTML templates in the application.
- What logic is implemented:
    - Includes Bootstrap CSS and JavaScript for consistent styling and functionality.
    - Sets up a navigation bar with the application title.
    - Displays flashed messages (e.g., success, error messages) to the user.
    - Defines a `content` block where child templates can insert their specific content.
- What algorithms or techniques are used: Jinja2 templating (template inheritance), HTML5, Bootstrap 5 for responsive design and UI components, custom CSS for branding.
- What inputs it takes: `title` variable (optional, for page title), `messages` (for flashed messages).
- What outputs it produces: A consistent header, footer, and overall page layout for the application.
- How it connects to other files: All other `app/templates/*.html` files extend this `base.html` to inherit its structure and styling. `get_flashed_messages` from Flask provides messages.


DATA_PROCESS DIR STARTS


### [APPEND | 2025-11-26 | data_process/__init__.py]

- What this file does: Aggregates and exposes functions from individual data processing modules, making them easily importable from the `data_process` package.
- What logic is implemented: Imports specific functions/variables from other files within the `data_process` directory. Defines `__all__` to control what symbols are exported when `*` is used in an import statement.
- What algorithms or techniques are used: Python package initialization and modular programming.
- What inputs it takes: N/A (serves as an aggregator).
- What outputs it produces: Makes data processing functions available for external use.
- How it connects to other files: Imports from `encode_categorical.py`, `feature_selection.py`, `handle_imbalance.py`, `handle_missing_values.py`, `load_dataset.py`, `scale_numeric.py`, and `split_dataset.py`. It is imported by `app/routes.py` to access data processing functionalities.

### [APPEND | 2025-11-26 | data_process/encode_categorical.py]

- What this file does: Provides functions to encode categorical features into a numerical format suitable for machine learning models.
- What logic is implemented:
    - Identifies non-numeric columns in a DataFrame.
    - Supports two encoding types:
        - "onehot": Uses `pd.get_dummies` for one-hot encoding, dropping the first category to avoid multicollinearity.
        - "label": Uses `sklearn.preprocessing.LabelEncoder` for label encoding, assigning a unique integer to each category.
- What algorithms or techniques are used: One-Hot Encoding, Label Encoding.
- What inputs it takes:
    - `df` (pandas.DataFrame): The input DataFrame containing features.
    - `encoding_type` (str): Specifies either "onehot" or "label".
- What outputs it produces:
    - `pd.DataFrame`: The DataFrame with categorical columns encoded.
    - `dict`: A dictionary of `LabelEncoder` objects (if label encoding is used), or an empty dictionary (for one-hot encoding).
- How it connects to other files: Imported by `data_process/__init__.py` and used by `app/routes.py` during data preprocessing.

### [APPEND | 2025-11-26 | data_process/feature_selection.py]

- What this file does: Implements functions for feature selection and visualization of feature importance.
- What logic is implemented:
    - `feature_search`:
        - Samples the input data using stratified shuffle split (for classification) or shuffle split (for regression).
        - Removes low variance features using `VarianceThreshold`.
        - Drops non-informative columns based on keywords (e.g., "id", "name").
        - Fits a `LogisticRegression` (for classification) or `LinearRegression` (for regression) model to estimate feature importance based on coefficients.
        - Returns sorted feature names and their importance scores.
    - `feature_selection`:
        - Selects features based on user input (`top_n_features`):
            - If a number, selects the top N features.
            - If "auto", selects features based on importance scores (keeping those >= 20% of max importance or >= median, with a fallback to top 1).
    - `imp_plot`: Generates a horizontal bar plot of feature importance and saves it to a specified path.
- What algorithms or techniques are used:
    - `VarianceThreshold` for low variance feature removal.
    - `StratifiedShuffleSplit` and `ShuffleSplit` for sampling.
    - `LogisticRegression` and `LinearRegression` for feature importance estimation.
    - `matplotlib` for plotting.
- What inputs it takes:
    - `feature_search`: Feature matrix (`X`), target vector (`y`), and `task_type` ("Classification" or "Regression").
    - `feature_selection`: Feature matrix (`X`), `top_n_features` (str or int), `sorted_cols`, `sorted_scores`.
    - `imp_plot`: List of column names, list of importance scores, output file path.
- What outputs it produces:
    - `feature_search`: Sorted feature names and scores.
    - `feature_selection`: List of selected feature names, number of features selected.
    - `imp_plot`: A saved image file (PNG) of the feature importance plot.
- How it connects to other files: Imported by `data_process/__init__.py` and used by `app/routes.py` for feature engineering and visualization.

### [APPEND | 2025-11-26 | data_process/handle_imbalance.py]

- What this file does: Handles class imbalance for classification tasks and target imbalance for regression tasks.
- What logic is implemented:
    - For classification: Uses SMOTE (Synthetic Minority Over-sampling Technique) or SMOTENC (SMOTE for Nominal and Continuous features) to oversample the minority class.
    - For regression: Uses `smogn` (Synthetic Minority Over-sampling for Regression with Gaussian Noise) to address rare/imbalanced target values. Includes fallback to original data if `smogn` fails.
- What algorithms or techniques are used: SMOTE, SMOTENC, SMOGN.
- What inputs it takes:
    - `X` (pandas.DataFrame): Feature matrix.
    - `y` (pandas.Series): Target variable.
    - `task_type` (str): "Classification" or "Regression".
- What outputs it produces: Resampled feature matrix (`X_res`) and target vector (`y_res`) as pandas DataFrames/Series.
- How it connects to other files: Imported by `data_process/__init__.py` and used by `app/routes.py` during data preprocessing.

### [APPEND | 2025-11-26 | data_process/handle_missing_values.py]

- What this file does: Implements functions to handle missing values in a DataFrame.
- What logic is implemented:
    - Drops rows where the `target_col` has missing values.
    - Drops columns that have more than 40% missing values.
    - For remaining missing values:
        - Numeric columns: Fills missing values using "mean" or "median" strategy.
        - Categorical columns: Fills missing values using "most_frequent" or "constant" (with "Unknown") strategy.
- What algorithms or techniques are used:
    - Simple imputation using `SimpleImputer` from scikit-learn.
    - Mean/Median imputation for numerical data.
    - Most frequent/Constant imputation for categorical data.
- What inputs it takes:
    - `df` (pandas.DataFrame): Input dataset.
    - `target_col` (str): Name of the target column.
    - `strategy_num` (str): Strategy for numeric columns ("mean", "median").
    - `strategy_cat` (str): Strategy for categorical columns ("most_frequent", "constant").
- What outputs it produces: `pd.DataFrame`: Dataset with missing values handled.
- How it connects to other files: Imported by `data_process/__init__.py` and used by `app/routes.py` during data preprocessing.

### [APPEND | 2025-11-26 | data_process/load_dataset.py]

- What this file does: Provides a utility function to load a CSV dataset into a pandas DataFrame.
- What logic is implemented: Uses `pandas.read_csv` to read a CSV file from a given path.
- What algorithms or techniques are used: Data loading with pandas.
- What inputs it takes: `file_path` (str): Path to the CSV file.
- What outputs it produces: `pd.DataFrame`: The raw dataset.
- How it connects to other files: Imported by `data_process/__init__.py` and used by `app/routes.py` to load user-uploaded datasets.

### [APPEND | 2025-11-26 | data_process/scale_numeric.py]

- What this file does: Scales numerical features and conditionally scales the target column in a DataFrame.
- What logic is implemented:
    - Initializes either `StandardScaler` or `MinMaxScaler` based on `scaler_type`.
    - Scales all numerical feature columns (excluding the target column).
    - Conditionally scales the target column if the `domain_name` is "DL" (Deep Learning) or if the `model_name` is "SVM" and `prediction_type` is "Regression". This is to ensure target scaling is applied only when necessary for specific model types.
- What algorithms or techniques are used: Standardization (`StandardScaler`), Normalization (`MinMaxScaler`).
- What inputs it takes:
    - `df` (pandas.DataFrame): Input dataset.
    - `target_col` (str): Name of the target column.
    - `domain_name` (str): Model domain ("ML" or "DL").
    - `model_name` (str): Specific model name (e.g., "SVM").
    - `prediction_type` (str): Prediction task type ("Classification" or "Regression").
    - `scaler_type` (str): Type of scaler to use ("standard" or "minmax").
- What outputs it produces:
    - `pd.DataFrame`: The DataFrame with scaled numerical columns.
    - `object`: The fitted scaler object.
- How it connects to other files: Imported by `data_process/__init__.py` and used by `app/routes.py` during data preprocessing.

### [APPEND | 2025-11-26 | data_process/split_dataset.py]

- What this file does: Splits a dataset into training and testing sets.
- What logic is implemented: Uses `sklearn.model_selection.train_test_split` to divide the feature matrix (`X`) and target vector (`y`) into training and testing subsets. It supports optional stratified splitting for classification tasks.
- What algorithms or techniques are used: Train-test splitting, optional stratified sampling.
- What inputs it takes:
    - `X` (pandas.DataFrame): Feature matrix.
    - `y` (pandas.Series): Target series.
    - `test_size` (float): Fraction of the dataset to allocate for the test set.
    - `random_state` (int): Seed for reproducibility.
    - `s_split` (bool): Flag to enable stratified splitting.
- What outputs it produces: Four NumPy arrays: `X_train`, `X_test`, `y_train`, `y_test`.
- How it connects to other files: Imported by `data_process/__init__.py` and used by `app/routes.py` to prepare data for model training and evaluation.



MODEL_HUB DIR STARTS


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

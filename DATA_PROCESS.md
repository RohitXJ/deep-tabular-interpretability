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
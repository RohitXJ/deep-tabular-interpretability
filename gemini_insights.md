# Project Insights: Deep Tabular Interpretability

## 1. Project Overview

This project is a tool for interpreting deep learning models on tabular data. It provides two interfaces: a command-line interface (CLI) and a Flask-based web application. Both interfaces allow a user to upload a tabular dataset (in CSV format), configure a machine learning model, and then train and evaluate the model. The project's main goal is to provide insights into the model's predictions by performing feature importance analysis.

## 2. Project Structure

The project is organized into the following key directories and files:

-   **`/` (root directory)**:
    -   `main.py`: The entry point for the command-line interface.
    -   `run.py`: The entry point for the Flask web application.
    -   `config.py`: Configuration for the Flask application (e.g., secret key, upload folder).
    -   `requirements.txt`: A list of Python dependencies for the project.
    -   `.gitignore`, `.gitattributes`: Git-related files.
    -   `LICENSE`: The project's license.

-   **`/app`**: Contains the Flask web application.
    -   `__init__.py`: Initializes the Flask application and registers the blueprint.
    -   `routes.py`: Defines the routes and view functions for the web application. This is the core of the web app's logic.
    -   `forms.py`: Defines the forms used in the web application (e.g., for file upload and model configuration).
    -   `/static`: Contains static files like images.
        -   `/images`: Stores the generated feature importance plots.
    -   `/templates`: Contains the HTML templates for the web application.
        -   `base.html`: The base template that other templates extend.
        -   `1_upload.html`: The file upload page.
        -   `2_configure_model.html`: The model selection page.
        -   `3_configure_data.html`: The data configuration page.
        -   `4_results.html`: The results page.

-   **`/data_process`**: Contains modules for data preprocessing.
    -   `encode_categorical.py`: Encodes categorical features.
    -   `feature_selection.py`: Selects the most important features.
    -   `handle_imbalance.py`: Handles imbalanced datasets.
    -   `handle_missing_values.py`: Handles missing values in the dataset.
    -   `load_dataset.py`: Loads the dataset from a CSV file.
    -   `scale_numeric.py`: Scales numeric features.
    -   `split_dataset.py`: Splits the dataset into training and testing sets.

-   **`/model_hub`**: Contains modules for model training and evaluation.
    -   `evaluation.py`: Evaluates the trained model.
    -   `model_init.py`: Initializes the machine learning models.
    -   `training.py`: Trains the machine learning models.

-   **`/uploads`**: The directory where uploaded CSV files are stored.

-   **`/Test_data`**: Contains sample CSV datasets for testing.

## 3. Command-Line Interface (`main.py`)

The `main.py` script provides a command-line interface for the machine learning pipeline. Here's how it works:

1.  **Domain and Model Selection**: The user is prompted to choose a domain (currently only "ML" is fully implemented), a prediction type (Classification or Regression), and a specific model.
2.  **Data Loading**: The script loads a dataset from a default path (`Test_data\tested.csv`).
3.  **Target Column Selection**: The user is asked to select the target column for prediction.
4.  **Feature Analysis**:
    -   The data is preprocessed (missing values are handled, and features are temporarily scaled and encoded).
    -   Feature importance scores are calculated using the `feature_search` function.
    -   A plot of the feature importances is generated and saved.
    -   The user is asked to select the number of top features to use for training, or to choose 'auto' for automatic selection.
5.  **Data Preprocessing**:
    -   The selected features are extracted.
    -   The data is scaled and encoded again, this time for the actual model training.
    -   The dataset is split into training and testing sets.
    -   If the task is classification, class imbalance is handled.
6.  **Model Training and Evaluation**:
    -   The selected model is initialized.
    -   The model is trained on the training data.
    -   The trained model is evaluated on the test data, and the results are printed to the console.

## 4. Web Application (Flask)

The Flask web application provides a more user-friendly, step-by-step interface for the same machine learning pipeline. The entry point is `run.py`, which creates and runs the Flask app defined in the `app` package.

### Routes (`app/routes.py`)

-   **`@bp.route('/', methods=['GET', 'POST'])` (upload)**:
    -   This is the main page of the application.
    -   It displays a form (`UploadForm`) for uploading a CSV file.
    -   On successful upload, the file is saved to the `uploads` folder with a unique name, and the file path is stored in the user's session. The user is then redirected to the model configuration page.

-   **`@bp.route('/configure_model', methods=['GET', 'POST'])` (configure_model)**:
    -   This page displays a form (`ModelConfigureForm`) for selecting the prediction type and the model.
    -   The available models are dynamically updated based on the selected prediction type using a little bit of JavaScript that fetches the models from the `/models/<prediction_type>` endpoint.
    -   The selected configuration is stored in the session, and the user is redirected to the data configuration page.

-   **`@bp.route('/configure_data', methods=['GET', 'POST'])` (configure_data)**:
    -   This page displays a form (`DataConfigureForm`) for selecting the target column and the number of features to use.
    -   When the user selects a target column, an AJAX request is sent to the `/run_feature_analysis` endpoint to perform feature importance analysis.
    -   The feature importance plot is displayed on the page, and the feature selection dropdown is populated.
    -   The final configuration is stored in the session, and the user is redirected to the results page.

-   **`@bp.route('/run_feature_analysis', methods=['POST'])` (run_feature_analysis)**:
    -   This is an API endpoint that is called from the data configuration page.
    -   It performs feature importance analysis on the uploaded dataset and the selected target column.
    -   It saves the feature importance plot as a PNG file in the `app/static/images` directory.
    -   It returns a JSON response containing the URL of the plot and the available feature selection choices.

-   **`@bp.route('/models/<prediction_type>')` (get_models)**:
    -   This is another API endpoint that is called from the model configuration page.
    -   It returns a JSON list of available models for a given prediction type (Classification or Regression).

-   **`@bp.route('/results')` (results)**:
    -   This is the final page of the workflow.
    -   It retrieves the configuration and feature analysis results from the session.
    -   It runs the full machine learning pipeline: data loading, preprocessing, feature selection, model training, and evaluation.
    -   It captures the model evaluation report and displays it on the page.
    -   It also cleans up the uploaded file.

## 5. Machine Learning Pipeline

The core of the project is the machine learning pipeline, which is used by both the CLI and the web application. The pipeline consists of the following steps:

1.  **Load Dataset**: Loads a CSV file into a pandas DataFrame.
2.  **Handle Missing Values**: Fills missing values in the dataset.
3.  **Feature Search/Analysis**:
    -   Uses the `feature_search` function to calculate feature importance scores. The exact method for feature search is not detailed in the code I've read, but it's likely using a model-based approach (e.g., feature importances from a tree-based model).
    -   Generates a bar plot of the feature importances.
4.  **Feature Selection**: Selects the top N features based on the importance scores.
5.  **Scale Numeric Features**: Scales numerical features using a scaler (e.g., `StandardScaler` or `MinMaxScaler`).
6.  **Encode Categorical Features**: Encodes categorical features into a numerical format (e.g., using Label Encoding).
7.  **Handle Imbalance**: If the task is classification and the dataset is imbalanced, it uses a technique to handle the imbalance (e.g., SMOTE).
8.  **Split Dataset**: Splits the data into training and testing sets.
9.  **Model Training**:
    -   Initializes the selected machine learning model.
    -   Trains the model on the training data.
10. **Model Evaluation**:
    -   Evaluates the trained model on the test data.
    -   For classification, it likely calculates metrics like accuracy, precision, recall, and F1-score.
    -   For regression, it likely calculates metrics like R-squared, Mean Squared Error, and Mean Absolute Error.

## 6. Key Libraries and Dependencies

The project relies on a number of popular Python libraries:

-   **Flask**: For the web application framework.
-   **Flask-WTF**: For handling web forms.
-   **pandas**: For data manipulation and analysis.
-   **scikit-learn**: For machine learning models, preprocessing, and evaluation.
-   **matplotlib** and **seaborn**: For data visualization and plotting.
-   **xgboost**, **lightgbm**, **catboost**: For gradient boosting models.
-   **imbalanced-learn**: For handling imbalanced datasets.
-   **numpy**: For numerical operations.
-   **python-dotenv**: For managing environment variables.

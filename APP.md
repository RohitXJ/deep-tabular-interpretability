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
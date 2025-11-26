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
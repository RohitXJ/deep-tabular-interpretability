### Deep Tabular Interpretability Platform

**Goal**: To provide an interactive web-based platform for users to upload tabular datasets, train various Machine Learning (ML) and Deep Learning (DL) models, evaluate their performance, and understand their predictions.

**Key Features**:
*   **Data Upload & Preprocessing**: Users can upload CSV files, and the platform handles missing values, categorical encoding, and numerical scaling.
*   **Feature Selection**: Automatic feature selection based on importance.
*   **Model Training & Evaluation**: Supports both traditional ML models (Logistic/Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost) and selected Deep Learning models (FNN, TabNet, TabTransformer).
*   **Two-Mode Hyperparameter Tuning**: Offers "Automatic" hyperparameter selection based on dataset characteristics for ease of use, and a "Manual" mode for advanced users.
*   **Performance Reporting**: Provides detailed evaluation metrics for trained models.
*   **GPU & CPU Support**: Automatically utilizes a CUDA-enabled GPU for training if available, with a fallback to CPU to ensure both performance and compatibility.

**Deep Learning Models Focus**:
The project specifically emphasizes Deep Learning models for tabular data, aiming to showcase their capabilities and interpretability. The selected DL models are:
*   **FNN (Feedforward Neural Network)**: A foundational deep learning model serving as a strong baseline.
*   **TabNet**: A powerful and interpretable deep learning model designed for tabular data, featuring built-in attention mechanisms for explainability.
*   **TabTransformer**: A modern and powerful deep learning model that applies the Transformer architecture to tabular data. It offers strong performance and is interpretable through its attention mechanism.

**Architecture**:
*   **Frontend**: Flask templates with HTML, CSS, and JavaScript for user interaction.
*   **Backend**: Flask application handling data processing, model training, and serving results.
*   **Model Hub**: A dedicated module (`model_hub`) containing model initialization, training, and evaluation logic for both ML and DL models.
*   **Data Processing**: A separate module (`data_process`) for data handling functionalities.

**Project Evolution**:
*   **Dependency Resolution**: Resolved critical dependency conflicts by removing unused libraries (`rtdl-revisiting-models`, `tab-transformer-pytorch` from initial code) and pinning a modern, compatible version of `pytorch-tabnet`. This allows the project to run on the latest versions of PyTorch.
*   **Bug Fixes**: Corrected several bugs in the initial codebase, including a dimensionality mismatch in the `NODE` model and incorrect initialization of the `TabNet` model.
*   **GPU Acceleration**: Implemented automatic GPU detection and utilization (with a CPU fallback) to significantly accelerate the training of deep learning models.
*   **Model Replacement**: Replaced the problematic and buggy `NODE` model with the `TabTransformer` model, a more robust, performant, and interpretable alternative.

**Future Scope (Interpretability)**:
The project is designed with future expansion into model interpretation in mind, leveraging techniques like SHAP and the inherent interpretability of models like TabNet and TabTransformer.
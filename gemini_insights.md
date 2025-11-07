### Deep Tabular Interpretability Platform

**Goal**: To provide an interactive web-based platform for users to upload tabular datasets, train various Machine Learning (ML) and Deep Learning (DL) models, evaluate their performance, and understand their predictions through state-of-the-art interpretability techniques.

**Key Features**:
*   **Data Upload & Preprocessing**: Users can upload CSV files, and the platform handles missing values, categorical encoding, and numerical scaling.
*   **Feature Selection**: Automatic feature selection based on importance scores derived from the data.
*   **Model Training & Evaluation**: Supports both traditional ML models (Logistic/Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost) and selected Deep Learning models (FNN, TabNet, TabTransformer).
*   **Two-Mode Hyperparameter Tuning**: Offers "Automatic" hyperparameter selection based on dataset characteristics for ease of use, and a "Manual" mode for advanced users.
*   **Performance Reporting**: Provides detailed and user-friendly evaluation metrics for trained models.
*   **GPU & CPU Support**: Automatically utilizes a CUDA-enabled GPU for training if available, with a fallback to CPU to ensure both performance and compatibility.

**Deep Learning Models Focus**:
The project specifically emphasizes Deep Learning models for tabular data, aiming to showcase their capabilities and interpretability. The selected DL models are:
*   **FNN (Feedforward Neural Network)**: A foundational deep learning model serving as a strong baseline.
*   **TabNet**: A powerful and interpretable deep learning model designed for tabular data, featuring built-in attention mechanisms for explainability.
*   **TabTransformer**: A modern and powerful deep learning model that applies the Transformer architecture to tabular data.

---

### Advanced Model Interpretability

To move beyond *what* a model predicts and explain *why*, the platform provides a multi-faceted interpretation report with clear, easy-to-understand explanations for each chart.

**1. SHAP (SHapley Additive exPlanations)**

*   **Primary Method**: For most models (all ML models, FNN, and TabTransformer), the platform uses SHAP to explain the model's output.
*   **Multi-Faceted Visualizations**:
    *   **Overall Feature Importance (The "What")**: A SHAP Summary Plot shows which features have the biggest impact on predictions. It visualizes not only the importance but also the direction of the effects.
    *   **Individual Prediction Breakdown (The "Why")**: A SHAP Force Plot breaks down a single, individual prediction, showing the tug-of-war between features that pushed the prediction higher versus lower.
    *   **Feature Dependence (The "How")**: Dependence plots are generated for the top 3 most important features to show how the model's prediction changes as the value of a single feature changes.

**2. TabNet's Built-in Interpretability**

*   **What it is**: TabNet uses a sequential attention mechanism to select the most salient features. The platform visualizes the aggregate of these selections to show the global importance of each feature as determined by the TabNet model itself.

**Implementation Workflow**:
1.  After a model is trained, the model object and test data are saved to a temporary, session-specific directory.
2.  A "View Interpretation" button on the results page links to the dedicated interpretation page.
3.  This page triggers a backend process that loads the saved artifacts, generates all the relevant plots and explanations, and renders them in a user-friendly report.
4.  The temporary artifacts are automatically cleaned up after the report is viewed.
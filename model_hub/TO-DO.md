# Model Training & Evaluation Tasks (Review 1 - Part 2)

## 🔹 ML Pipeline Tasks

### 1. Model Selection (ML)

* **Task:** Choose a classical ML model for regression or classification.
* **Conditions/Choices:**

  * Regression → {Linear Regression, Ridge/Lasso, Random Forest Regressor, XGBoost/LightGBM/CatBoost}
  * Classification → {Logistic Regression, Random Forest Classifier, XGBoost/LightGBM/CatBoost, SVM (optional)}
* **Output:** Untrained model instance.

---

### 2. Model Training (ML)

* **Task:** Fit the selected model on training data.
* **Conditions/Choices:**

  * Supervised fit (`.fit(X_train, y_train)`)
  * No history object (unlike DL)
* **Output:** Trained model.

---

### 3. Model Evaluation (ML)

* **Task:** Evaluate trained model on test data.
* **Conditions/Choices:**

  * Regression metrics → MSE, RMSE, MAE, R²
  * Classification metrics → Accuracy, Precision, Recall, F1
* **Output:** Metrics dictionary.

---

### 4. Visualization (ML)

* **Task:** Visualize performance results.
* **Conditions/Choices:**

  * Regression → Residual plots, Predicted vs Actual plots
  * Classification → Confusion matrix, ROC curve, PR curve
* **Output:** Matplotlib/Seaborn plots.

---

## 🔹 DL Pipeline Tasks

### 1. Model Selection (DL)

* **Task:** Choose a DL architecture for regression or classification.
* **Conditions/Choices:**

  * Common → FNN/MLP
  * Advanced → Wide & Deep Network, TabNet, TabTransformer
* **Output:** Untrained PyTorch model.

---

### 2. Model Training (DL)

* **Task:** Train model with backpropagation and optimizer.
* **Conditions/Choices:**

  * Define optimizer (Adam, SGD, etc.)
  * Define loss function (MSE for regression, CrossEntropy for classification)
  * Track metrics per epoch
* **Output:** Trained model + Training history.

---

### 3. Model Evaluation (DL)

* **Task:** Evaluate trained model on test data.
* **Conditions/Choices:**

  * Regression metrics → MSE, RMSE, MAE, R²
  * Classification metrics → Accuracy, Precision, Recall, F1
* **Output:** Metrics dictionary.

---

### 4. Visualization (DL)

* **Task:** Plot performance curves.
* **Conditions/Choices:**

  * Training vs Validation Loss over epochs
  * Training vs Validation Accuracy (classification only)
* **Output:** Training history plots.

---

### 5. Additional Diagnostics (DL)

* **Task:** Provide deeper evaluation for interpretability.
* **Conditions/Choices:**

  * Classification → Confusion matrix, ROC, PR curve
  * Regression → Error distribution plots
* **Output:** Diagnostic visualizations.

---

## 🔹 Key Separation of ML vs DL Tasks

* **ML:** Simple fit, no epochs/history, lightweight metrics/plots.
* **DL:** Requires optimizer, epochs, history tracking, specialized plots.

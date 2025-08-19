# Model Training & Evaluation Functions (Review 1 - Part 2)

## 1. Select Model (model_init.py)

```python
def select_model(model_type: str = "mlp", **kwargs):
    """
    Initializes a model based on user choice.

    Args:
        model_type (str): Type of model ("mlp", "random_forest", "logistic", etc.).
        kwargs: Additional parameters for model initialization.
    Returns:
        object: Initialized model.
    """
```

**Input:** `model_type`, params
**Output:** `model object`

---

## 2. Train Model (training.py)

```python
def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains the given model on training data.

    Args:
        model (object): Initialized model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
    Returns:
        object: Trained model.
    """
```

**Input:** `(model, X_train, y_train)`
**Output:** `trained model`

---

## 3. Evaluate Model (evaluation.py)

```python
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluates model performance using accuracy, precision, recall, F1-score.

    Args:
        model (object): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    Returns:
        dict: Metrics dictionary.
    """
```

**Input:** `(model, X_test, y_test)`
**Output:** `{ "accuracy": float, "precision": float, "recall": float, "f1": float }`

---

## 4. Save Model (I_O.py)

```python
def save_model(model, file_path: str):
    """
    Saves the trained model to disk.

    Args:
        model (object): Trained model.
        file_path (str): Path to save model.
    Returns:
        None
    """
```

**Input:** `(model, file_path)`
**Output:** None (saved file)

---

## 5. Load Model (I_O.py)

```python
def load_model(file_path: str):
    """
    Loads a saved model from disk.

    Args:
        file_path (str): Path to saved model.
    Returns:
        object: Loaded model.
    """
```

**Input:** `file_path`
**Output:** `model object`

---

## 6. Plot Performance Metrics (evaluation.py)

```python
def plot_metrics(history: dict):
    """
    Plots training/validation loss and accuracy over epochs (for DL models).

    Args:
        history (dict): Dictionary with loss/accuracy values.
    Returns:
        None
    """
```

**Input:** `history dict`
**Output:** Visualization (matplotlib/seaborn plot)

---

## 7. Confusion Matrix & Classification Report (evaluation.py)

```python
def plot_confusion_matrix_and_report(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Plots confusion matrix and prints classification report.

    Args:
        model (object): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    Returns:
        None
    """
```

**Input:** `(model, X_test, y_test)`
**Output:** Confusion matrix plot + printed report

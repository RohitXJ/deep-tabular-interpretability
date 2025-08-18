# Data Preprocessing Functions (Review 1)

## 1. Load Dataset (Riyanka)

```python
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV dataset from the given file path.

    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Raw dataset as a pandas DataFrame.
    """
```

**Input:** `"path/to/file.csv"`
**Output:** `pd.DataFrame`

---

## 2. Handle Missing Values (Riyanka)

```python
def handle_missing_values(df: pd.DataFrame, strategy_num: str = "mean", strategy_cat: str = "most_frequent") -> pd.DataFrame:
    """
    Fills missing values for numeric and categorical columns.

    Args:
        df (pd.DataFrame): Input dataset.
        strategy_num (str): Strategy for numeric columns ("mean", "median").
        strategy_cat (str): Strategy for categorical columns ("most_frequent", "constant").
    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
```

**Input:** `pd.DataFrame`
**Output:** `pd.DataFrame` (cleaned)

---

## 3. Encode Categorical Columns (Ankur)

```python
def encode_categorical(df: pd.DataFrame, encoding_type: str = "onehot") -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical columns into numeric form.

    Args:
        df (pd.DataFrame): Input dataset.
        encoding_type (str): "onehot" or "label".
    Returns:
        tuple: (Encoded DataFrame, mapping dictionary for label encoding)
    """
```

**Input:** `pd.DataFrame`
**Output:** `(pd.DataFrame, dict)` (encoded df, label mapping if used)

---

## 4. Scale Numeric Columns (Ankur)

```python
def scale_numeric(df: pd.DataFrame, scaler_type: str = "standard") -> tuple[pd.DataFrame, object]:
    """
    Scales numeric columns.

    Args:
        df (pd.DataFrame): Input dataset.
        scaler_type (str): "standard" or "minmax".
    Returns:
        tuple: (Scaled DataFrame, fitted scaler object)
    """
```

**Input:** `pd.DataFrame`
**Output:** `(pd.DataFrame, scaler object)`

---

## 5. Handle Class Imbalance (Bidisha)

```python
def handle_imbalance(X: np.ndarray, y: np.ndarray, method: str = "smote") -> tuple[np.ndarray, np.ndarray]:
    """
    Handles class imbalance using resampling techniques.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        method (str): "smote", "undersample", or "oversample".
    Returns:
        tuple: (Balanced X, Balanced y)
    """
```

**Input:** `(X, y)` NumPy arrays
**Output:** `(X_balanced, y_balanced)` NumPy arrays

---

## 6. Split Dataset (Bidisha)

```python
def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits dataset into train and test sets.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        target_column (str): Column to predict.
        test_size (float): Fraction of dataset for testing.
        random_state (int): Seed for reproducibility.
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
```

**Input:** `pd.DataFrame`, target column name
**Output:** `(X_train, X_test, y_train, y_test)` as NumPy arrays

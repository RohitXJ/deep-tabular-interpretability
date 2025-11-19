"""
================================
Type 1: Model Architectures & Data Setup
================================

This file contains:
1.  A data processing function to convert cleaned data into PyTorch Tensors.
2.  The four custom ANN architectures (2 Regression, 2 Classification).

These functions and classes can be imported and instantiated in your main application.

User: Gemini CLI
Project: Deep Learning Interpretability
"""

# Required Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# DATA PRE-PROCESSING FORMAT (AFTER CLEANING)
# ----------------------------------------------------------------------

def create_pytorch_tensors_and_dataloaders(X_train_scaled, y_train, X_test_scaled, y_test, batch_size=32):
    """
    Converts cleaned/scaled numpy arrays and pandas Series into PyTorch 
    Tensors and a DataLoader for training.
    
    This function works for both regression and binary classification,
    as both require a float target tensor with shape [n_samples, 1].
    
    Args:
        X_train_scaled (np.ndarray): Scaled training features.
        y_train (pd.Series): Training target values.
        X_test_scaled (np.ndarray): Scaled test features.
        y_test (pd.Series): Test target values.
        batch_size (int): Batch size for the training DataLoader.

    Returns:
        tuple: (train_loader, X_train_t, X_test_t, y_test_t)
               - train_loader: DataLoader for the training set.
               - X_train_t: Tensor of training features.
               - X_test_t: Tensor of test features.
               - y_test_t: Tensor of test targets.
    """
    
    # --- Convert training data to PyTorch tensors ---
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    # Reshape y to [n_samples, 1] (required for both MSELoss and BCELoss)
    # Use .values to get numpy array from pandas series
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # --- Convert test data to PyTorch tensors ---
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # --- Create TensorDataset and DataLoader for training ---
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Return loaders and tensors (tensors are needed for SHAP and final eval)
    return train_loader, X_train_t, X_test_t, y_test_t


# ----------------------------------------------------------------------
# MODULE 1: REGRESSION ARCHITECTURES
# ----------------------------------------------------------------------

class ANN_Shallow_Regression(nn.Module):
    """
    A simple, shallow ANN for regression tasks.
    Architecture: Input -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Output(1)
    
    Matches 'RegressionModel1' from the notebook.
    """
    def __init__(self, input_shape):
        super(ANN_Shallow_Regression, self).__init__()
        self.layer1 = nn.Linear(input_shape, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1) # No activation for regression output

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

class ANN_Deep_Regression(nn.Module):
    """
    A deeper, regularized ANN for regression tasks using Dropout.
    Architecture: Input -> Dense(128) -> ReLU -> Dropout(0.2) -> 
                  Dense(64) -> ReLU -> Dropout(0.2) -> 
                  Dense(32) -> ReLU -> Output(1)
                  
    Matches 'RegressionModel2' from the notebook.
    """
    def __init__(self, input_shape):
        super(ANN_Deep_Regression, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Output layer
        )

    def forward(self, x):
        return self.layer_stack(x)


# ----------------------------------------------------------------------
# MODULE 2: CLASSIFICATION ARCHITECTURES
# ----------------------------------------------------------------------

class ANN_Shallow_Classification(nn.Module):
    """
    A simple, shallow ANN for binary classification tasks.
    Architecture: Input -> Dense(32) -> ReLU -> Dense(16) -> ReLU -> Output(1) -> Sigmoid
    
    Matches 'ClassificationModel1' from the notebook.
    """
    def __init__(self, input_shape):
        super(ANN_Shallow_Classification, self).__init__()
        self.layer1 = nn.Linear(input_shape, 32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        x = self.sigmoid(x)
        return x

class ANN_Deep_Classification(nn.Module):
    """
    A deeper, regularized ANN for binary classification tasks using Dropout.
    Architecture: Input -> Dense(64) -> ReLU -> Dropout(0.3) -> 
                  Dense(32) -> ReLU -> Dropout(0.3) -> 
                  Output(1) -> Sigmoid
                  
    Matches 'ClassificationModel2' from the notebook.
    """
    def __init__(self, input_shape):
        super(ANN_Deep_Classification, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer_stack(x)
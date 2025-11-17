import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import io

def DL_model_train(model: nn.Module, train_loader: DataLoader, prediction_type: str, epochs: int) -> nn.Module:
    """
    Trains a Deep Learning model.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training data.
        prediction_type (str): "Classification" or "Regression".
        epochs (int): Number of training epochs.

    Returns:
        nn.Module: The trained PyTorch model.
    """
    if prediction_type == "Regression":
        criterion = nn.MSELoss()
    elif prediction_type == "Classification":
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Capture stdout for live logging
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Print loss every 10 epochs or at the last epoch
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            sys.stdout.flush() # Ensure output is flushed immediately

    sys.stdout = old_stdout # Restore stdout

    return model, captured_output.getvalue()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import io

def DL_model_train(model: nn.Module, train_loader: DataLoader, prediction_type: str, epochs: int) -> tuple[nn.Module, list]:
    """
    Trains a Deep Learning model and returns the trained model and loss history.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training data.
        prediction_type (str): "Classification" or "Regression".
        epochs (int): Number of training epochs.

    Returns:
        tuple[nn.Module, list]: A tuple containing the trained PyTorch model and a list of loss values for each epoch.
    """
    if prediction_type == "Regression":
        criterion = nn.MSELoss()
    elif prediction_type == "Classification":
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        # Log to server console for monitoring
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

    return model, loss_history

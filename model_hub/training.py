import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from tab_transformer_pytorch import TabTransformer


def ML_model_train(model:object,data:list):
    X_train,y_train = data
    model.fit(X_train,y_train)
    return model

def DL_model_train(model:object, data:list, task_type:str, cat_features:list, num_features:list, epochs:int=10, batch_size:int=32, learning_rate:float=0.001, device:str='cpu'):
    X_train, y_train = data

    if isinstance(model, (TabNetClassifier, TabNetRegressor)):
        from sklearn.model_selection import train_test_split
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy().reshape(-1, 1)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_np, y_train_np, test_size=0.2, random_state=42)
        model.fit(
            X_train=X_train_split, y_train=y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            max_epochs=epochs,
            batch_size=batch_size,
            patience=10
        )
        return model

    device = torch.device(device)
    model.to(device)
    
    # Convert data to PyTorch tensors
    # Separate categorical and numerical features for TabTransformer
    if isinstance(model, TabTransformer):
        # Ensure categorical features are long type
        X_train_cat = torch.tensor(X_train[cat_features].values, dtype=torch.long)
        X_train_num = torch.tensor(X_train[num_features].values, dtype=torch.float32)
        X_train_tensor = (X_train_cat, X_train_num) # Tuple for TabTransformer
    elif isinstance(model, nn.Sequential): # Explicitly handle FNN
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    else:
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    if isinstance(model, TabTransformer):
        train_dataset = TensorDataset(X_train_tensor[0], X_train_tensor[1], y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    if task_type == "Classification":
        criterion = nn.CrossEntropyLoss()
        y_train_tensor = y_train_tensor.long()  # Convert target to long for CrossEntropyLoss
    else: # Regression
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_features if c in X_train.columns]
    num_feature_indices = [X_train.columns.get_loc(c) for c in num_features if c in X_train.columns]

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_data in train_loader:
            if isinstance(model, TabTransformer):
                batch_X_cat, batch_X_num, batch_y = batch_data
                batch_X_cat, batch_X_num, batch_y = batch_X_cat.to(device), batch_X_num.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X_cat, batch_X_num)
            else:
                batch_X, batch_y = batch_data
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)

            if task_type == "Classification":
                loss = criterion(outputs, batch_y.long())
            else:
                loss = criterion(outputs, batch_y.unsqueeze(1))
            
            loss.backward()
            optimizer.step()

    return model


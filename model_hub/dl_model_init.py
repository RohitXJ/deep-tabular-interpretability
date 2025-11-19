import torch.nn as nn
from ANN_architecture import ANN_Shallow_Regression, ANN_Deep_Regression, ANN_Shallow_Classification, ANN_Deep_Classification

def DL_models_call(type: str, model: str, input_shape: int) -> nn.Module:
    """
    Initializes and returns a Deep Learning model based on the specified type and model name.

    Args:
        type (str): The prediction type, either "Classification" or "Regression".
        model (str): The name of the DL model (e.g., "Shallow ANN", "Deep ANN").
        input_shape (int): The number of input features for the model.

    Returns:
        nn.Module: An instance of the specified PyTorch model.

    Raises:
        ValueError: If an unknown model type or name is provided.
    """
    if type == "Regression":
        if model == "Shallow ANN":
            return ANN_Shallow_Regression(input_shape)
        elif model == "Deep ANN":
            return ANN_Deep_Regression(input_shape)
        else:
            raise ValueError(f"Unknown Regression model: {model}")
    elif type == "Classification":
        if model == "Shallow ANN":
            return ANN_Shallow_Classification(input_shape)
        elif model == "Deep ANN":
            return ANN_Deep_Classification(input_shape)
        else:
            raise ValueError(f"Unknown Classification model: {model}")
    else:
        raise ValueError(f"Unknown prediction type: {type}")

import torch
from catenets.models.torch.representation_nets import DragonNet  # Adjust if your catenets install differs
import numpy as np

class DragonNetLearner:
    """
    A wrapper for catenets' DragonNet.
    Expects DragonNet to have a fit(X, y, w) method and a predict(X) method that 
    returns a tuple (y0, y1) of treatment-specific outcome predictions.
    """
    def __init__(self, input_dim, epochs=50, batch_size=32):
        self.input_dim = input_dim
        # Initialize DragonNet with input dimension only; do not pass reg_l2.
        self.model = DragonNet(input_dim)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, Y, T, X):
        # Convert numpy arrays to torch tensors.
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)  # shape (n, 1)
        T_tensor = torch.tensor(T, dtype=torch.float32).unsqueeze(1)  # shape (n, 1)
        # Call the DragonNet fit method.
        # (Assumes the model's fit method accepts X, y, and w, and handles training internally.)
        self.model.fit(X_tensor, Y_tensor, T_tensor)
        return self

    def effect(self, X):
        # Convert X to tensor.
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Assume predict returns a tuple (y0, y1) as torch tensors.
        return self.model.predict(X_tensor).detach().cpu().numpy()
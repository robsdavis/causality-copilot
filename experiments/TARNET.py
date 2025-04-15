import torch
from catenets.models.torch.representation_nets import TARNet  # Adjust if your catenets install differs
import numpy as np

class TarNetLearner:
    """
    A wrapper for catenets' TARNet.
    Expects TARNet to have a fit(X, y, w) method and a predict(X) method that 
    returns a tuple (y0, y1) of treatment-specific outcome predictions.
    """
    def __init__(self, input_dim, epochs=50, batch_size=32):
        self.input_dim = input_dim
        # Initialize TARNet with input dimension (do not pass reg_l2).
        self.model = TARNet(input_dim)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, Y, T, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
        T_tensor = torch.tensor(T, dtype=torch.float32).unsqueeze(1)
        self.model.fit(X_tensor, Y_tensor, T_tensor)
        return self

    def effect(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return self.model.predict(X_tensor).detach().cpu().numpy()
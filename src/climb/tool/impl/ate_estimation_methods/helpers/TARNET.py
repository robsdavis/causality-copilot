import torch
import numpy as np
from catenets.models.torch.representation_nets import TARNet
from catenets.models.torch.base import DEVICE

class TarNetLearner:
    """
    A wrapper for catenets' TARNet.
    Expects TARNet to have fit(X, y, w) and predict(X) methods.
    The fit method here takes features X, treatment t, and outcome y,
    and calls TARNet model.fit with (X, y, t).
    """
    def __init__(
        self,
        input_dim: int,
        seed: int = 0,
        val_split_prop: float = 0.0,
        n_iter: int = 50,
        batch_size: int = 32,
    ):
        # Initialize TARNet with proper training and validation settings
        self.model = TARNet(
            n_unit_in=input_dim,
            val_split_prop=val_split_prop,
            seed=seed,
            n_iter=n_iter,
            batch_size=batch_size,
        ).to(DEVICE)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "TarNetLearner":
        # Convert numpy arrays to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        t_tensor = torch.tensor(t, dtype=torch.long).squeeze().to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32).squeeze().to(DEVICE)
        # Fit the TARNet model (handles its own internal train/val split)
        self.model.fit(X_tensor, y_tensor, t_tensor)
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        # Convert features to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        # Predict returns the individual treatment effect (y1 - y0)
        with torch.no_grad():
            tau = self.model.predict(X_tensor)
        return tau.detach().cpu().numpy()

    # Alias predict to effect so external callers can use predict()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.effect(X)

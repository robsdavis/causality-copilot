import torch
import numpy as np
from catenets.models.torch.representation_nets import DragonNet
from catenets.models.torch.base import DEVICE

import torch
import numpy as np
from catenets.models.torch.representation_nets import DragonNet
from catenets.models.torch.base import DEVICE

class DragonNetLearner:
    """
    A wrapper for catenets' DragonNet.
    Expects DragonNet to have fit(X, y, w) and predict(X) methods.
    """
    def __init__(
        self,
        input_dim: int,
        seed: int = 0,
        val_split_prop: float = 0.0,
        n_iter: int = 50,
        batch_size: int = 32,
    ):
        # Initialize DragonNet with proper training and validation settings
        self.model = DragonNet(
            n_unit_in=input_dim,
            val_split_prop=val_split_prop,
            seed=seed,
            n_iter=n_iter,
            batch_size=batch_size,
        ).to(DEVICE)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "DragonNetLearner":
        # Convert numpy arrays to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        # outcome y should be float and shape (n_samples,)
        y_tensor = torch.tensor(y, dtype=torch.float32).squeeze().to(DEVICE)
        # treatment w as long for classification head
        w_tensor = torch.tensor(w, dtype=torch.long).squeeze().to(DEVICE)
        # Fit the DragonNet model (handles its own internal train/val split)
        self.model.fit(X_tensor, y_tensor, w_tensor)
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

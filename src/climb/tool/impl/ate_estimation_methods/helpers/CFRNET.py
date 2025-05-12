import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# A simple CFR loss (here we use MSE on the factual outcomes as a placeholder).
class CFRLoss(nn.Module):
    def __init__(self, alpha=3, metric="W1"):
        super(CFRLoss, self).__init__()
        self.alpha = alpha
        self.metric = metric

    def forward(self, y1_hat, y0_hat, y1_factual, y0_factual, treatment, phi):
        # Here, we ignore distributional loss for brevity.
        loss0 = F.mse_loss(y0_hat, y0_factual)
        loss1 = F.mse_loss(y1_hat, y1_factual)
        return loss0 + loss1

# CFR network implementation.
class CFR(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        rep_dim=200,
        hyp_dim=100,
        lr=1e-4,
        epochs=100,
        batch=128,
        decay=0,
        alpha=3,
        metric="W1",
        set_custom_seed_model: bool = False,
        seed: int = 0,
        binary_y: bool = False,
    ):
        super(CFR, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.decay = decay
        self.alpha = alpha
        self.metric = metric
        if set_custom_seed_model:
            torch.manual_seed(seed)
        # Representation network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU(),
        )
        # Potential outcome networks
        self.func0 = nn.Sequential(
            nn.Linear(rep_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, output_dim),
        )
        self.func1 = nn.Sequential(
            nn.Linear(rep_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, output_dim),
        )

    def forward(self, X):
        Phi = self.encoder(X)
        Y0 = self.func0(Phi)
        Y1 = self.func1(Phi)
        return Phi, Y0, Y1

    def fit(self, X, y_factual, treatment):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        # Convert inputs if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y_factual, torch.Tensor):
            y_factual = torch.tensor(y_factual, dtype=torch.float32)
        if not isinstance(treatment, torch.Tensor):
            treatment = torch.tensor(treatment, dtype=torch.float32)
        # Align shapes
        if y_factual.ndim == 1:
            y_factual = y_factual.unsqueeze(1)
        if treatment.ndim == 1:
            treatment = treatment.unsqueeze(1)
        # Training
        input_dim = X.shape[1]
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        loader = DataLoader(torch.cat([X, treatment, y_factual], dim=1), batch_size=self.batch, shuffle=True)
        mse = nn.MSELoss()
        for _ in range(self.epochs):
            for batch in loader:
                batch = batch.to(device)
                train_X = batch[:, :input_dim]
                train_t = batch[:, input_dim].unsqueeze(1)
                train_y = batch[:, input_dim+1: input_dim+2]
                _, y0_hat, y1_hat = self(train_X)
                loss = 0.0
                if torch.sum(train_t) > 0:
                    loss += mse(y1_hat[train_t.squeeze() == 1], train_y[train_t.squeeze() == 1])
                if torch.sum(1 - train_t) > 0:
                    loss += mse(y0_hat[train_t.squeeze() == 0], train_y[train_t.squeeze() == 0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)
        with torch.no_grad():
            _, Y0, Y1 = self(X)
        return (Y1 - Y0).cpu().numpy()

class CFRNetLearner:
    """
    Wrapper exposing a uniform fit/predict interface.
    """
    def __init__(self, input_dim, output_dim=1, rep_dim=200, hyp_dim=100,
                 lr=1e-4, epochs=100, batch=128, decay=0, alpha=3,
                 metric="W1", random_state=0):
        self.model = CFR(
            input_dim, output_dim, rep_dim, hyp_dim,
            lr, epochs, batch, decay, alpha, metric,
            set_custom_seed_model=True, seed=random_state
        )

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "CFRNetLearner":
        """Fit CFRNet with features X, treatment t, and outcome y."""
        # Delegate to model
        self.model.fit(X, y, t)
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Estimate treatment effects for X."""
        return self.model.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for effect()."""
        return self.effect(X)

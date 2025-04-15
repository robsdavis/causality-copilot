# CFRNET.py
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# A simple CFR loss (here we use MSE on the factual outcomes as a placeholder).
class CFRLoss(nn.Module):
    def __init__(self, alpha=3, metric="W1"):
        super(CFRLoss, self).__init__()
        self.alpha = alpha
        self.metric = metric

    def forward(self, y1_hat, y0_hat, y1_factual, y0_factual, treatment, phi):
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

        # Representation layer.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU(),
        )

        # Potential outcome for control (y0)
        self.func0 = nn.Sequential(
            nn.Linear(rep_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, hyp_dim),
            nn.ELU(),
            nn.Linear(hyp_dim, output_dim),
        )

        # Potential outcome for treated (y1)
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
        self.to(device)  # Move model to GPU if CUDA is available
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y_factual, torch.Tensor):
            y_factual = torch.tensor(y_factual, dtype=torch.float32)
        if not isinstance(treatment, torch.Tensor):
            treatment = torch.tensor(treatment, dtype=torch.float32).unsqueeze(1)

        # Ensure y_factual is 2D.
        if len(y_factual.shape) == 1:
            y_factual = y_factual.unsqueeze(1)

        # Concatenate X, treatment, and y_factual.
        data = torch.cat((X, treatment, y_factual), dim=1)

        input_dim = X.shape[1]
        cfr_loss = CFRLoss(alpha=self.alpha, metric=self.metric)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.decay)
        loader = DataLoader(data, batch_size=self.batch, shuffle=True)
        mse = nn.MSELoss()
        for epoch in range(self.epochs):
            for tr in loader:
                tr = tr.to(device)
                train_X = tr[:, :input_dim]
                train_t = tr[:, input_dim].unsqueeze(1)
                train_y = tr[:, input_dim+1: input_dim+2]
                phi, y0_hat, y1_hat = self(train_X)
                optimizer.zero_grad()
                # For simplicity, compute MSE on factual outcomes.
                loss = 0
                if torch.sum(train_t) > 0:
                    loss += mse(y1_hat[train_t == 1], train_y[train_t == 1])
                if torch.sum(1 - train_t) > 0:
                    loss += mse(y0_hat[train_t == 0], train_y[train_t == 0])
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
        Y0 = Y0.cpu().numpy()
        Y1 = Y1.cpu().numpy()
        return Y1 - Y0

class CFRNetLearner:
    def __init__(self, input_dim, output_dim=1, rep_dim=200, hyp_dim=100, lr=1e-4, epochs=100, batch=128, decay=0, alpha=3, metric="W1", random_state=0):
        self.model = CFR(input_dim, output_dim, rep_dim, hyp_dim, lr, epochs, batch, decay, alpha, metric, set_custom_seed_model=True, seed=random_state)
    def fit(self, Y, T, X):
        self.model.fit(X, Y, T)
        return self
    def effect(self, X):
        return self.model.predict(X)

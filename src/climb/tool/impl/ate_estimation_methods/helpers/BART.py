# src/climb/tool/impl/ate_estimation_methods/helpers/BART.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


class X_Learner_BART:
    """
    Implements the X-Learner framework using Random Forests to mimic BART-style two-stage estimation.
    """
    def __init__(self, n_trees: int = 100, random_state: int = 0, binary_y: bool = False):
        self.binary_y = binary_y
        # Outcome models for control and treatment groups
        self.M0 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M1 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        # Adjustment models for imputed treatment effects
        self.M2 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M3 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        # Propensity model
        self.g = LogisticRegression(max_iter=2000, random_state=random_state)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "X_Learner_BART":
        """
        Train the X-Learner:
        - Estimate outcomes for control (M0) and treated (M1)
        - Estimate propensity scores g
        - Impute individual treatment effects D_hat
        - Fit adjustment models M2 and M3

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        t : np.ndarray, shape (n_samples,) or (n_samples, 1)
            Binary treatment assignment (0 or 1)
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
            Outcome vector
        """
        # Flatten arrays
        t = t.ravel()
        y = y.ravel()

        # Split data
        X0, y0 = X[t == 0], y[t == 0]
        X1, y1 = X[t == 1], y[t == 1]

        # 1) Fit outcome models
        self.M0.fit(X0, y0)
        self.M1.fit(X1, y1)

        # 2) Fit propensity model
        self.g.fit(X, t)

        # 3) Impute treatment effects
        #    For controls: effect = E[Y|X, T=1] - Y
        #    For treated: effect = Y - E[Y|X, T=0]
        pred1 = self.M1.predict(X)
        pred0 = self.M0.predict(X)
        D_hat = np.where(
            t == 0,
            pred1 - y,
            y - pred0
        )

        # 4) Fit adjustment models
        self.M2.fit(X0, D_hat[t == 0])
        self.M3.fit(X1, D_hat[t == 1])

        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate conditional average treatment effects for new data X.

        Returns:
        --------
        tau_hat : np.ndarray, shape (n_samples,)
            Predicted individual treatment effects
        """
        # Propensity of control vs treated
        proba = self.g.predict_proba(X)
        p0 = proba[:, 0]
        p1 = proba[:, 1]

        # Predict adjusted effects
        adj0 = self.M2.predict(X)
        adj1 = self.M3.predict(X)

        # Weighted combination
        return p0 * adj0 + p1 * adj1


class BARTLearner:
    """
    High-level wrapper exposing a uniform fit/predict interface.
    """
    def __init__(self, n_trees: int = 100, random_state: int = 0, binary_y: bool = False):
        self.model = X_Learner_BART(n_trees=n_trees, random_state=random_state, binary_y=binary_y)

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "BARTLearner":
        """Fit the BART-based X-Learner."""
        self.model.fit(X, t, y)
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Predict treatment effects for X."""
        return self.model.effect(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for effect()."""
        return self.effect(X)

# BART.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

class X_Learner_BART:
    def __init__(self, n_trees: int = 100, random_state: int = 0, binary_y: bool = False):
        self.binary_y = binary_y
        self.M1 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M2 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M3 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.M4 = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
        self.g = LogisticRegression(max_iter=2000, random_state=random_state)

    def fit(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray):
        # Split data into control and treatment groups.
        X0 = X[(T == 0).ravel()]
        X1 = X[(T == 1).ravel()]
        Y0 = Y[(T == 0).ravel()].ravel()
        Y1 = Y[(T == 1).ravel()].ravel()
        self.M1.fit(X0, Y0)
        self.M2.fit(X1, Y1)
        self.g.fit(X, T.ravel())
        # Compute imputed treatment effect.
        D_hat = np.where(
            (T == 0).ravel(),
            self.M2.predict(X) - Y.ravel(),
            Y.ravel() - self.M1.predict(X),
        )
        self.M3.fit(X0, D_hat[(T == 0).ravel()])
        self.M4.fit(X1, D_hat[(T == 1).ravel()])
        return self

    def effect(self, X):
        # Combine predictions weighted by propensity probabilities.
        return (self.g.predict_proba(X)[:, 0] * self.M3.predict(X) +
                self.g.predict_proba(X)[:, 1] * self.M4.predict(X))


class BARTLearner:
    def __init__(self, n_trees: int = 100, random_state: int = 0, binary_y: bool = False):
        self.model = X_Learner_BART(n_trees=n_trees, random_state=random_state, binary_y=binary_y)
    def fit(self, Y, T, X):
        self.model.fit(X, Y, T)
        return self
    def effect(self, X):
        return self.model.effect(X)

from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .ate_estimation_base import ATEEstimationBase

class RegressionEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "random_forest_regression"

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Regression-based ATE estimation:
          - fit E[Y | W, X] with regression on (T,X)
          - predict potential outcomes at W=1 and W=0
          - return individual effects y1-y0
        """
        # build design matrix with treatment and covariates
        D = np.column_stack((T.reshape(-1, 1), X))
        model = RandomForestRegressor(random_state=seed)
        model.fit(D, Y)
        # counterfactual design matrices
        n = X.shape[0]
        D1 = np.column_stack((np.ones(n), X))
        D0 = np.column_stack((np.zeros(n), X))
        # predict outcomes
        y1 = model.predict(D1)
        y0 = model.predict(D0)
        # individual treatment effect
        return y1 - y0
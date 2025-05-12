from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import XLearner
from .ate_estimation_base import ATEEstimationBase

class XLearnerEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "x-learner"

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> np.ndarray:
        rf1 = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf2 = RandomForestRegressor(n_estimators=100, random_state=seed)
        learner = XLearner(models=[rf1, rf2])
        learner.fit(Y, T, X=X)
        return learner.effect(X)


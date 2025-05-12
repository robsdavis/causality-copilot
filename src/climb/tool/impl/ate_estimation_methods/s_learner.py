from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import SLearner
from .ate_estimation_base import ATEEstimationBase

class SLearnerEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "s-learner"

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> np.ndarray:
        model = RandomForestRegressor(n_estimators=100, random_state=seed)
        learner = SLearner(overall_model=model)
        learner.fit(Y, T, X=X)
        return learner.effect(X)
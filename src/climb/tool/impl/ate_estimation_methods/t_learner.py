from typing import Any
import numpy as np
from econml.metalearners import TLearner
from sklearn.ensemble import RandomForestRegressor
from .ate_estimation_base import ATEEstimationBase

class TLearnerEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "t-learner"

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> np.ndarray:
        # set up base learners with this seed
        rf1 = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf2 = RandomForestRegressor(n_estimators=100, random_state=seed)

        learner = TLearner(models=[rf1, rf2])
        learner.fit(Y, T, X=X)
        # return the individual treatment effect predictions
        return learner.effect(X)

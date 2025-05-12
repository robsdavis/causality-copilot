from typing import Any
import numpy as np
from .helpers.dragonnet import DragonNetLearner
from .ate_estimation_base import ATEEstimationBase

class DragonNetEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "dragonnet"

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> np.ndarray:
        input_dim = X.shape[1]
        learner = DragonNetLearner(input_dim)
        learner.fit(X, Y, T)
        return learner.predict(X)
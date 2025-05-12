from typing import Any
import numpy as np
from .helpers.CFRNET import CFRNetLearner
from .ate_estimation_base import ATEEstimationBase

class CFRNetEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "cfrnet"

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
        learner = CFRNetLearner(input_dim, random_state=seed)
        learner.fit(X, T, Y)
        return learner.predict(X)
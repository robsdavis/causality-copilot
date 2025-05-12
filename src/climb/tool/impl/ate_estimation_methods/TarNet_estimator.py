from typing import Any
import numpy as np
from .helpers.TARNET import TarNetLearner
from .ate_estimation_base import ATEEstimationBase

class TarNetEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "tarnet"

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
        learner = TarNetLearner(input_dim)
        learner.fit(X, T, Y)
        return learner.predict(X)
from typing import Any
import numpy as np
from .helpers.BART import BARTLearner
from .ate_estimation_base import ATEEstimationBase

class BARTEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "bart"

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> np.ndarray:
        learner = BARTLearner()
        learner.fit(X, T, Y)
        return learner.predict(X)
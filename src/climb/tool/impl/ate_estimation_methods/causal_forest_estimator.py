from typing import Any
import numpy as np
from econml.dml import CausalForestDML
from .ate_estimation_base import ATEEstimationBase

class CausalForestEstimation(ATEEstimationBase):
    @property
    def NAME(self) -> str:
        return "cf"

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any,
    ) -> np.ndarray:
        learner = CausalForestDML(
            n_estimators=500,
            min_samples_leaf=10,
            max_depth=15,
            cv=2,
            discrete_treatment=True,
            random_state=seed
        )
        learner.fit(Y, T, X=X)
        return learner.effect(X)
from typing import Any, Dict, List, Optional
import numpy as np

class ATEEstimationBase:
    """
    Base class for ATE estimation plugins. Handles looping over seeds,
    computing ATE and (optional) PEHE, and summarizing results.

    Subclasses must implement:
      - a @property NAME: str
      - a classmethod _estimate_effect_once(Y, T, X, seed, **kwargs) -> np.ndarray
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        instance = cls()
        if not isinstance(instance.NAME, str):
            raise TypeError(
                f"Subclass {cls.__name__!r} must define a NAME property returning a string"
            )

    @property
    def NAME(self) -> str:
        raise NotImplementedError("Subclasses must override the NAME property")

    @classmethod
    def run(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        n_runs: int = 10,
        tau_true: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        Orchestrates n_runs repetitions of effect estimation, aggregates ATE and PEHE.

        Parameters
        ----------
        Y : array-like
            Outcomes.
        T : array-like
            Binary treatment indicators.
        X : array-like
            Covariates.
        n_runs : int
            Number of bootstrap runs / random seeds.
        tau_true : array-like, optional
            True individual treatment effects; if provided, PEHE is computed.

        Returns
        -------
        Dict[str, float]
            'ate_mean', 'ate_std', 'pehe_mean', 'pehe_std'
        """
        ate_list: List[float] = []
        pehe_list: List[float] = []

        for seed in range(n_runs):
            tau_hat = cls._estimate_effect_once(
                Y, T, X, seed=seed, **kwargs
            )
            ate_list.append(float(np.mean(tau_hat)))

            if tau_true is not None:
                pehe_list.append(
                    float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
                )
            else:
                pehe_list.append(float('nan'))

        return {
            "ate_mean": round(float(np.mean(ate_list)), 4),
            "ate_std":  round(float(np.std(ate_list)), 4),
            "pehe_mean": round(float(np.nanmean(pehe_list)), 4),
            "pehe_std":  round(float(np.nanstd(pehe_list)), 4),
        }

    @classmethod
    def _estimate_effect_once(
        cls,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        seed: int,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Run one seed’s worth of training and effect estimation.

        Must return tau_hat: an array of individual treatment effects.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _estimate_effect_once()"
        )

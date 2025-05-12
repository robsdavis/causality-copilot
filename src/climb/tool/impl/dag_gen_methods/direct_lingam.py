from typing import Any
import numpy as np
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from .dag_discovery_base import DAGDiscoveryBase


class DirectLiNGAMDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "direct-lingam"

    @classmethod
    def _run(
        cls,
        data: np.ndarray,
        node_names: list[str],
        prune_threshold: float = 0.2,
        dir_threshold: float = 0.7,
        n_sampling: int = 100,
        min_causal_effect: float = 1e-6,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the signed adjacency matrix for DirectLiNGAM with bootstrap stability.
        """
        # 1) Fit base model
        model = DirectLiNGAM()
        model.fit(data)

        # 2) Bootstrap
        bs = model.bootstrap(data, n_sampling=n_sampling)

        # 3) Get edge‐presence probabilities
        probs = bs.get_probabilities(min_causal_effect=min_causal_effect)
        n = probs.shape[0]

        # 4) Build signed adjacency
        signed = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                f_ij = probs[i, j]
                f_ji = probs[j, i]

                # a) prune if both below threshold
                if f_ij < prune_threshold and f_ji < prune_threshold:
                    continue

                # b) strong i->j
                if f_ij >= dir_threshold and f_ij > f_ji:
                    signed[i, j] = -1
                    signed[j, i] = 1

                # c) strong j->i
                elif f_ji >= dir_threshold and f_ji > f_ij:
                    signed[j, i] = -1
                    signed[i, j] = 1

                # d) otherwise undirected
                else:
                    signed[i, j] = signed[j, i] = -1

        return signed
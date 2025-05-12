from typing import Any, List
import numpy as np
from causallearn.search.FCMBased.lingam import ICALiNGAM
from .dag_discovery_base import DAGDiscoveryBase


class ICALiNGAMDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "ica-lingam"

    @classmethod
    def _run(
        cls,
        data: np.ndarray,
        node_names: List[str],
        prune_threshold: float = 0.1,
        dir_threshold: float = 0.7,
        n_sampling: int = 100,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        ICA-LiNGAM with bootstrap stability + two-threshold rule:

          • prune_threshold: edges with bootstrap-presence < prune_threshold in BOTH directions are dropped.
          • dir_threshold: edges with freq ≥ dir_threshold (and stronger than reverse) become directed.
          • otherwise, surviving edges become undirected.
        """
        # 1) Fit base model
        model = ICALiNGAM()
        model.fit(data)

        # 2) Bootstrap the fitted model
        bs = model.bootstrap(data, n_sampling=n_sampling)

        # 3) Stack adjacency matrices and compute presence frequency
        all_mats = np.stack(bs.adjacency_matrices_, axis=0)
        freq = (np.abs(all_mats) > 0).mean(axis=0)

        n = freq.shape[0]
        signed = np.zeros((n, n), dtype=int)

        # 4) Build signed-CPDAG
        for i in range(n):
            for j in range(i + 1, n):
                f_ij = freq[i, j]
                f_ji = freq[j, i]

                # a) drop if both weak
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

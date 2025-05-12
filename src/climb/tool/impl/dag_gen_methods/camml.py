from typing import Any, List
import numpy as np
from tqdm import tqdm
from causallearn.search.FCMBased.lingam import CAMUV
from .dag_discovery_base import DAGDiscoveryBase

class CAMUVDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "camml"

    @classmethod
    def _run(
        cls,
        data: np.ndarray,
        node_names: List[str],
        prune_threshold: float = 0.1,
        dir_threshold: float = 0.99,
        n_sampling: int = 20,
        alpha: float = 0.05,
        num_explanatory_vals: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        CAM-UV with bootstrap stability + two-threshold rule:

          • prune_threshold: drop edges absent in ≥ (1-prune_threshold) of bootstraps.
          • dir_threshold: edges with freq ≥ dir_threshold become directed.
          • otherwise (survived but below dir_threshold) become undirected.
        """
        n = data.shape[1]
        k = num_explanatory_vals or n

        # 1) bootstrap adjacency counts
        mats = np.zeros((n_sampling, n, n), dtype=int)
        for b in tqdm(range(n_sampling)):
            idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
            Xb = data[idx, :]

            # run CAM-UV
            P, _ = CAMUV.execute(Xb, alpha, num_explanatory_vals=k)

            # fill binary adjacency: parent -> child
            for child_idx, parents in enumerate(P):
                for p in parents:
                    mats[b, p, child_idx] = 1

        # 2) compute frequency of each directed edge
        freq = mats.mean(axis=0)

        # 3) build signed-CPDAG
        signed = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                f_ij = freq[i, j]
                f_ji = freq[j, i]

                # prune
                if f_ij < prune_threshold and f_ji < prune_threshold:
                    continue

                # strong i->j
                if f_ij >= dir_threshold and f_ij > f_ji:
                    signed[i, j] = 1

                # strong j->i
                elif f_ji >= dir_threshold and f_ji > f_ij:
                    signed[j, i] = 1

                # undirected
                else:
                    signed[i, j] = signed[j, i] = 1

        return signed

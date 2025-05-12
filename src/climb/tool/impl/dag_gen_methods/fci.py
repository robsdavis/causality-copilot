from typing import Any, List
import numpy as np
from tqdm import tqdm
from causallearn.search.ConstraintBased.FCI import fci
from .dag_discovery_base import DAGDiscoveryBase


class FCIDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "fci"

    @classmethod
    def _run(
        cls,
        data: np.ndarray,
        node_names: List[str],
        prune_threshold: float = 0.1,
        dir_threshold: float = 0.7,
        n_sampling: int = 50,
        alpha: float = 0.05,
        indep_test: str = "fisherz",
        max_cond_set_size: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        FCI with bootstrap stability + two-threshold rule:

          • prune_threshold: drop edges present in < prune_threshold fraction of bootstraps
          • dir_threshold: edges with freq ≥ dir_threshold become directed
          • otherwise (survived but below dir_threshold) become undirected
        """
        n = data.shape[1]

        # validate max_cond_set_size
        if max_cond_set_size is not None:
            max_cond_set_size = int(max_cond_set_size)
            if max_cond_set_size < 0:
                raise ValueError("max_cond_set_size must be non-negative")
            if max_cond_set_size > n:
                raise ValueError("max_cond_set_size must be ≤ number of nodes")
            if max_cond_set_size == 0:
                raise ValueError("max_cond_set_size must be ≥ 1")

        # 1) bootstrap PAG adjacency
        mats = np.zeros((n_sampling, n, n), dtype=int)
        for b in tqdm(range(n_sampling)):
            idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
            Xb = data[idx, :]

            pag, _ = fci(
                Xb,
                alpha=alpha,
                node_names=node_names,
                indep_test=indep_test.lower(),
                max_cond_set_size=max_cond_set_size,
            )

            A = pag.graph  # adjacency encoding
            mats[b] = (A > 0).astype(int)

        # 2) compute frequencies
        freq = mats.mean(axis=0)

        # 3) build signed adjacency
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

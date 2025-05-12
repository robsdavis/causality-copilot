from typing import Any, List
import torch
import numpy as np
from tqdm import tqdm
from .helpers.SCORE_helpers import SCORE as stein_SCORE
from .dag_discovery_base import DAGDiscoveryBase


class SCOREDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "score"

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
        SCORE with bootstrap stability + two-threshold rule:

          • prune_threshold: drop edges with presence freq < prune_threshold in BOTH directions
          • dir_threshold: edges with freq ≥ dir_threshold (and stronger than reverse) become directed
          • intermediate (surviving but not strong) edges become undirected
        """
        n = data.shape[1]
        mats = np.zeros((n_sampling, n, n), dtype=int)

        # 1) Bootstrap loop
        for b in range(n_sampling):
            idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
            Xb = torch.tensor(data[idx, :], dtype=torch.float32)

            # run SCORE on bootstrap sample
            adj_t, _ = stein_SCORE(
                Xb,
                eta_G=kwargs.get("eta_G", 0.001),
                eta_H=kwargs.get("eta_H", 0.001),
                cutoff=kwargs.get("cutoff", 0.001),
                normalize_var=kwargs.get("normalize_var", False),
                dispersion=kwargs.get("dispersion", "var"),
                pruning=kwargs.get("pruning", "CAM"),
                threshold=kwargs.get("threshold", 0.1),
            )

            # to NumPy and binarize
            if isinstance(adj_t, torch.Tensor):
                adj_np = adj_t.detach().cpu().numpy()
            else:
                adj_np = np.array(adj_t)
            mats[b] = (adj_np != 0).astype(int)

        # 2) Empirical frequency of each edge
        freq = mats.mean(axis=0)

        # 3) Build signed-CPDAG matrix
        signed = np.zeros((n, n), dtype=int)
        for i in tqdm(range(n)):
            for j in range(i + 1, n):
                f_ij = freq[i, j]
                f_ji = freq[j, i]

                # a) prune if both directions below prune_threshold
                if f_ij < prune_threshold and f_ji < prune_threshold:
                    continue

                # b) strong i->j?
                if f_ij >= dir_threshold and f_ij > f_ji:
                    signed[i, j] = -1
                    signed[j, i] = 1

                # c) strong j->i?
                elif f_ji >= dir_threshold and f_ji > f_ij:
                    signed[j, i] = -1
                    signed[i, j] = 1

                # d) otherwise undirected
                else:
                    signed[i, j] = signed[j, i] = -1

        return signed

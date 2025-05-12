from typing import Any, List
import numpy as np
from tqdm import tqdm
import networkx as nx
from climb.tool.dag_helpers import run_notears
from .dag_discovery_base import DAGDiscoveryBase


class NoTearsDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "notears"

    @classmethod
    def _run(
        cls,
        data: np.ndarray,
        node_names: List[str],
        prune_threshold: float = 0.1,
        dir_threshold: float = 0.7,
        n_sampling: int = 100,
        lambda1: float = 0.05,
        loss_type: str = "l2",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        NO-TEARS with bootstrap stability + two-threshold rule:
          • prune_threshold: edges with presence freq < prune_threshold in BOTH directions are dropped.
          • dir_threshold: edges with freq ≥ dir_threshold become directed.
          • intermediate edges become undirected.
        """
        n = data.shape[1]

        # 1) Bootstrap the NO-TEARS weight estimates
        mats = np.zeros((n_sampling, n, n), dtype=int)
        for b in tqdm(range(n_sampling)):
            idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
            Xb = data[idx, :]

            # Solve NO-TEARS on bootstrap sample
            res = run_notears(Xb, node_names, lambda1=lambda1, loss_type=loss_type)
            Gb = res["G"]
            Wb_mat = nx.to_numpy_array(Gb, nodelist=node_names)
            mats[b] = (np.abs(Wb_mat) > 0).astype(int)

        # 2) Compute edge-presence frequencies
        freq = mats.mean(axis=0)

        # 3) Build signed-CPDAG
        signed = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                f_ij = freq[i, j]
                f_ji = freq[j, i]

                # a) drop if both directions weak
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

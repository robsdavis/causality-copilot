from typing import Any, List
import numpy as np
from causallearn.search.ScoreBased.GES import ges
from .dag_discovery_base import DAGDiscoveryBase


class GESDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "ges"

    @classmethod
    def _run(
        cls,
        data: np.ndarray,
        node_names: List[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        GES score-based search:
          • returns a CPDAG: directed edges where orientation is certain,
            undirected edges (i--j) otherwise.
        """
        # run GES and get the resulting graph
        res = ges(data, node_names=node_names)
        G = res["G"]

        # adjacency encoding: GeneralGraph.graph gives a numpy array A
        # where positive entry A[i,j]>0 indicates an edge mark tail->arrow
        A = G.graph
        n = len(node_names)
        signed = np.zeros((n, n), dtype=int)

        # interpret A to build signed adjacency
        for i in range(n):
            for j in range(i + 1, n):
                aij = A[i, j]
                aji = A[j, i]
                # directed i->j if only A[i,j]>0
                if aij > 0 and aji == 0:
                    signed[i, j] = 1
                # directed j->i if only A[j,i]>0
                elif aji > 0 and aij == 0:
                    signed[j, i] = 1
                # undirected if both marks exist
                elif aij > 0 and aji > 0:
                    signed[i, j] = signed[j, i] = 1
                # else: no edge
        return signed
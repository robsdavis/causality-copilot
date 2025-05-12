from typing import Any, List
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from .dag_discovery_base import DAGDiscoveryBase

class PCDiscovery(DAGDiscoveryBase):
    @property
    def NAME(self) -> str:
        return "pc"

    @classmethod
    def _run(
        cls,
        data: np.ndarray,
        node_names: List[str],
        alpha: float = 0.9,
        indep_test: str = "kci",
        max_cond_set_size: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        PC algorithm with pluggable CI test and optional depth limit.
        Returns a CPDAG: directed where oriented, undirected otherwise.
        """
        # normalize parameters
        indep_test = indep_test.lower()
        if max_cond_set_size is not None:
            max_cond_set_size = int(max_cond_set_size)

        # # run PC
        model = pc(
            data,
            alpha=alpha,
            node_names=node_names,
            indep_test=indep_test,
            max_cond_set_size=max_cond_set_size,
        )
        G = model.G

        # adjacency encoding from GeneralGraph
        A = G.graph
        n = len(node_names)
        signed = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(i+1, n):
                aij, aji = A[i,j], A[j,i]

                # i -> j
                if aij == -1 and aji == 1:
                    signed[i,j] = 1

                # j -> i
                elif aji == -1 and aij == 1:
                    signed[j,i] = 1

                # i -- j
                elif aij == 1 and aji == 1:
                    signed[i,j] = signed[j,i] = 1

        return signed
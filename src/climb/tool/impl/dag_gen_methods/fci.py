from typing import Any
import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from .helpers.simple_graph import _SimpleGraph

NAME = "fci"

def run(
    data: np.ndarray,
    node_names: list[str],
    prune_threshold: float = 0.1,
    dir_threshold: float = 0.7,
    n_sampling: int = 50,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    max_cond_set_size: int | None = None,
    **kwargs: Any,
):
    """
    FCI with bootstrap stability + two-threshold rule:

      • prune_threshold: drop edges present in < prune_threshold fraction of bootstraps  
      • dir_threshold:  edges with freq ≥ dir_threshold become directed  
      • otherwise (survived but below dir_threshold) become undirected  
    """
    n = data.shape[1]

    # validate max_cond_set_size as before…
    if max_cond_set_size is not None:
        max_cond_set_size = int(max_cond_set_size)
        if max_cond_set_size < 0:
            raise ValueError("max_cond_set_size must be non-negative")
        if max_cond_set_size > n:
            raise ValueError("max_cond_set_size must be ≤ number of nodes")
        if max_cond_set_size == 0:
            raise ValueError("max_cond_set_size must be ≥ 1")

    mats = np.zeros((n_sampling, n, n), dtype=int)

    for b in range(n_sampling):
        # 1) bootstrap sample
        idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
        Xb = data[idx, :]

        # 2) run FCI
        pag, _ = fci(
            Xb,
            alpha=alpha,
            node_names=node_names,
            indep_test=indep_test.lower(),
            max_cond_set_size=max_cond_set_size,
        )

        # 3) extract bootstrapped adjacency from the returned PAG:
        #    pag.graph is an (n×n) NumPy array where
        #      >0 means “tail at i, arrow at j” (i->j),
        #      <0 means “arrow at i, tail at j” (j->i),
        #       0 means no edge.
        A = pag.graph  # numpy ndarray 
        # We want mats[b,i,j]=1 if there is any edge pointing or undirected toward j:
        #   • directed i->j gives A[i,j]>0 → mats[b,i,j]=1
        #   • undirected or circle edges (which in PAG are encoded with positive entries)
        #     produce A[i,j]>0 and A[j,i]>0, so both directions get 1
        mats[b] = (A > 0).astype(int)

    # 4) compute edge‐presence frequencies
    freq = mats.mean(axis=0)  # shape (n, n)
    print(f"\n\nFCI frequency matrix:\n {freq}\n\n")

    # 5) apply double-threshold to build a signed‐CPDAG
    signed = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            f_ij = freq[i, j]
            f_ji = freq[j, i]

            # a) drop if both weak
            if f_ij < prune_threshold and f_ji < prune_threshold:
                continue

            # b) strong i -> j
            if f_ij >= dir_threshold and f_ij > f_ji:
                signed[i, j] = 1

            # c) strong j -> i
            elif f_ji >= dir_threshold and f_ji > f_ij:
                signed[j, i] = 1

            # d) otherwise undirected
            else:
                signed[i, j] = signed[j, i] = 1
    print(f"\n\nsigned:\n {signed}\n\n")

    # 6) wrap in your SimpleGraph
    graph_obj = _SimpleGraph(signed, node_names)
    return {"G": graph_obj}

from typing import Any
import numpy as np
from causallearn.search.FCMBased.lingam import CAMUV
from .helpers.simple_graph import _SimpleGraph

NAME = "camml"

def run(
    data: np.ndarray,
    node_names: list[str],
    prune_threshold: float = 0.1,
    dir_threshold: float = 0.99,
    n_sampling: int = 20,
    alpha: float = 0.05,
    num_explanatory_vals: int | None = None,
    **kwargs: Any,
):
    """
    CAM-UV with bootstrap stability + two-threshold rule:
    
      • prune_threshold: drop edges absent in ≥ (1-prune_threshold) of bootstraps.  
      • dir_threshold: edges with freq ≥ dir_threshold become directed.  
      • otherwise (pruned-survived but below dir_threshold) become undirected.
    """
    print(f"\n\n node names: {node_names}\n\n")
    n = data.shape[1]
    k = num_explanatory_vals or n

    # 1) bootstrap adjacency counts
    mats = np.zeros((n_sampling, n, n), dtype=int)
    for b in range(n_sampling):
        # resample rows with replacement
        idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
        Xb = data[idx, :]

        # run CAM-UV
        P, _ = CAMUV.execute(Xb, alpha, num_explanatory_vals=k)

        # fill binary adjacency: parent -> child
        for child_idx, parents in enumerate(P):
            for p in parents:
                mats[b, p, child_idx] = 1

    # 2) compute frequency of each directed edge across bootstraps
    freq = mats.mean(axis=0)  # shape (n, n)
    print("\n\nCAM-UV frequency matrix:\n", freq)
    
    # 3) build signed‐CPDAG encoding:
    #    - directed i->j: signed[i,j]=1, signed[j,i]=0
    #    - undirected:      signed[i,j]=signed[j,i]=1
    #    - dropped:         signed[...,]=0
    signed = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            f_ij = freq[i, j]
            f_ji = freq[j, i]

            # a) drop if both below prune_threshold
            if f_ij < prune_threshold and f_ji < prune_threshold:
                continue

            # b) strong i->j?
            if f_ij >= dir_threshold and f_ij > f_ji:
                signed[i, j] = 1

            # c) strong j->i?
            elif f_ji >= dir_threshold and f_ji > f_ij:
                signed[j, i] = 1

            # d) otherwise undirected
            else:
                signed[i, j] = signed[j, i] = 1
    print("\n\nCAM-UV signed-CPDAG matrix:\n", signed)

    # 4) wrap into your SimpleGraph (which interprets 1 as an edge)
    graph_obj = _SimpleGraph(signed, node_names)
    return {"G": graph_obj}

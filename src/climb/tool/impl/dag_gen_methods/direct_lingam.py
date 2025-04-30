from typing import Any
import numpy as np
from causallearn.search.FCMBased.lingam import DirectLiNGAM

from .helpers.simple_graph import _SimpleGraph

NAME = "direct-lingam"

def run(
    data: np.ndarray,
    node_names: list[str],
    prune_threshold: float = 0.2,
    dir_threshold: float = 0.7,
    n_sampling: int = 100,
    min_causal_effect: float = 1e-6,
    **kwargs: Any,
):
    """
    DirectLiNGAM with bootstrap stability and two‐threshold rule:
      • prune_threshold: below this frequency, edge is dropped.
      • dir_threshold: above this frequency, edge is directed.
      • in between: edge is undirected.
    """
    # 1) Fit base model
    model = DirectLiNGAM()
    model.fit(data)

    # 2) Bootstrap
    bs = model.bootstrap(data, n_sampling=n_sampling)

    # 3) Get edge‐presence probabilities
    probs = bs.get_probabilities(min_causal_effect=min_causal_effect)
    n = probs.shape[0]

    # 4) Build signed‐CPDAG
    signed = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            f_ij = probs[i, j]
            f_ji = probs[j, i]

            # 1) prune if both below prune_threshold
            if f_ij < prune_threshold and f_ji < prune_threshold:
                continue

            # 2) strong i->j?
            if f_ij >= dir_threshold and f_ij > f_ji:
                signed[i, j] = -1
                signed[j, i] = 1

            # 3) strong j->i?
            elif f_ji >= dir_threshold and f_ji > f_ij:
                signed[j, i] = -1
                signed[i, j] = 1

            # 4) otherwise undirected
            else:
                signed[i, j] = signed[j, i] = -1

    # 5) Wrap into _SimpleGraph and return
    graph_obj = _SimpleGraph(signed, node_names)
    return {"G": graph_obj}
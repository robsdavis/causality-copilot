from typing import Any
import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

NAME = "ges"

def run(
    data: np.ndarray,
    node_names: list[str],
    **kwargs: Any,
):
    """
    GES score‐based search.  
    score_type: e.g. "bic" or "aic" (if supported by your version of causal‐learn).
    """

    res = ges(
        data,
        node_names=node_names,
    )
    return {"G": res["G"]}

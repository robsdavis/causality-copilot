from typing import Any
import numpy as np
from causallearn.search.ConstraintBased.PC import pc

NAME = "pc"

def run(
    data: np.ndarray,
    node_names: list[str],
    **kwargs: Any,
):
    """
    PC algorithm with pluggable CI test and optional depth limit.
    """
    alpha = kwargs.get("alpha", 0.05)
    indep_test = kwargs.get("indep_test", "fisherz")
    max_cond_set_size = kwargs.get("max_cond_set_size", None)

    indep_test = indep_test.lower()
    model = pc(
        data,
        alpha=alpha,
        node_names=node_names,
        indep_test=indep_test,
        max_cond_set_size=max_cond_set_size,
    )
    return {"G": model.G}

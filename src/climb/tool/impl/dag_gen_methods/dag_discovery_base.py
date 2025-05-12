from typing import Any, List
import numpy as np
from .helpers.simple_graph import _SimpleGraph


class DAGDiscoveryBase:
    """
    Base class for DAG discovery plugins.

    Subclasses must implement the `NAME` property and the `_run` method,
    which returns a signed adjacency matrix (`np.ndarray` of shape [n,n])
    using the convention:
      - `-1` at [i,j] and `1` at [j,i] for directed i->j
      - `1` at both [i,j] and [j,i] for undirected i--j
      - `0` for absent edges.

    The base `run` method wraps the signed matrix into a `_SimpleGraph`.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        instance = cls()
        if not isinstance(instance.NAME, str):
            raise TypeError(f"Subclass {cls.__name__!r} must define a NAME property returning a string")

    @property
    def NAME(self) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} must implement the NAME property")

    @classmethod
    def run(cls, data: np.ndarray, node_names: List[str], **kwargs: Any) -> dict:
        # Delegate to subclass implementation to get signed adjacency matrix
        signed = cls._run(data, node_names, **kwargs)
        # Wrap into a SimpleGraph and return
        graph_obj = _SimpleGraph(signed, node_names)
        return {"G": graph_obj}

    @classmethod
    def _run(cls, data: np.ndarray, node_names: List[str], **kwargs: Any) -> np.ndarray:
        """
        Subclasses must implement this method to compute the signed adjacency matrix.
        """
        raise NotImplementedError(f"{cls.__name__} must implement the _run() method")


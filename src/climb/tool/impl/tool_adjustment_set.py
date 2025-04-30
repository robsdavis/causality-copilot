import os
from typing import Any, Optional, Dict
import pickle as pkl

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.linear_model import LinearRegression
from optimaladj.CausalGraph import CausalGraph

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

def load_causal_graph(dag_pickle_path: str | Path) -> CausalGraph:
    """
    Load whatever was pickled and return a fully-featured CausalGraph from optimaladj.
    Supports:
      • Already a CausalGraph
      • A causal-learn GeneralGraph wrapper with .G
      • Our _SimpleGraph wrapper with .graph & .node_names
      • A raw NumPy array
    """
    dag_pickle_path = Path(dag_pickle_path)
    with dag_pickle_path.open("rb") as f:
        obj = pkl.load(f)

    # If it *is* already the right CausalGraph, just return it.
    if isinstance(obj, CausalGraph):
        return obj

    # Otherwise, convert to a networkx.DiGraph first:
    if hasattr(obj, "G"):
        Gnx = obj.G  # causal-learn .G is already networkx
    elif hasattr(obj, "graph") and hasattr(obj, "node_names"):
        # signed CPDAG wrapper: rebuild a networkx
        Gnx = _nx_from_signed_cpdag(obj.graph, obj.node_names)
    elif isinstance(obj, np.ndarray):
        # raw array, assume node names X0..Xn
        names = [f"X{i}" for i in range(obj.shape[0])]
        Gnx = _nx_from_signed_cpdag(obj, names)
    else:
        raise ValueError(f"Unrecognized pickle contents in {dag_pickle_path}")
    
    # Now build an *empty* optimaladj.CausalGraph and add nodes/edges.
    cg = CausalGraph()
    for n in Gnx.nodes():
        cg.add_node(n)
    for u, v in Gnx.edges():
        cg.add_edge(u, v)
    cg._nx = Gnx

    return cg


def _nx_from_signed_cpdag(adj: np.ndarray, node_names: list[str]) -> nx.DiGraph:
    """
    Turn a signed-CPDAG matrix into a networkx DiGraph.
    Convention:
      - Directed i->j: adj[i,j] == -1 and adj[j,i] == 1
      - Undirected i--j: adj[i,j] == adj[j,i] == -1  (add both directions)
    """
    n = adj.shape[0]
    Gnx = nx.DiGraph()
    Gnx.add_nodes_from(node_names)

    for i in range(n):
        for j in range(n):
            if adj[i, j] == -1 and adj[j, i] == 1:
                Gnx.add_edge(node_names[i], node_names[j])
            elif adj[i, j] == -1 and adj[j, i] == -1:
                # Add both directions
                Gnx.add_edge(node_names[i], node_names[j])
                Gnx.add_edge(node_names[j], node_names[i])
    return Gnx

def treatment_in_causal_vertices(cg, treatment, outcome):
    Gnx = getattr(cg, "_nx")
    paths = nx.all_simple_paths(Gnx, source=treatment, target=outcome)
    causal_vertices = set().union(*paths) if paths else set()
    return treatment in causal_vertices

def compute_adj_sets(
    tc: ToolCommunicator,
    dag_pickle_path: str,
    treatment: str,
    outcome: str,
) -> None:
    """
    Loads a serialized DAG from a pickle file as a causal graph and computes optimal adjustment sets.

    This function performs the following steps:
      1) Loads the DAG from the provided dag_pickle_path and converts it into a CausalGraph.
      2) Sets L to an empty list, and N to all features (i.e. all nodes in the causal graph).
      3) Calls each of these functions safely (catching exceptions that may be raised as ConditionException errors):
            a) causal_graph.optimal_adj_set(treatment, outcome, L, N)
            b) causal_graph.optimal_minimal_adj_set(treatment, outcome, L, N)
            c) causal_graph.optimal_minimum_adj_set(treatment, outcome, L, N)
      4) Returns a dictionary with the following keys:
         {
             "optimal adjustment set": <list from optimal_adj_set or empty list on error>,
             "optimal minimal adjustment set": <list from optimal_minimal_adj_set or empty list on error>,
             "optimal minimum adjustment set": <list from optimal_minimum_adj_set or empty list on error>,
         }

    Parameters:
      - tc: ToolCommunicator for logging and setting returns.
      - dag_pickle_path (str): Path to the serialized DAG (pickle file).
      - treatment (str): Name of the treatment variable.
      - outcome (str): Name of the outcome variable.
    
    Note:
      - L is always set to an empty list.
      - N is set to all nodes (features) in the causal graph.
    """
    # Load the serialized DAG from the pickle file.
    try:
        causal_graph = load_causal_graph(dag_pickle_path)
    except Exception as e:
        tc.print("Error loading the DAG:", e)
        tc.set_returns(
            tool_return="Error loading the DAG. Please make sure you have run the DAG generation tool first and provided the correct path to the DAG pickle file.",
            user_report=["Error loading the DAG. Please make sure you have run the DAG generation tool first and provided the correct path to the DAG pickle file."],
        )
        return
    
    # raise error if treatment or outcome not in the causal vertices
    if not treatment_in_causal_vertices(causal_graph, treatment, outcome):
        msg = (
            f"Error: Treatment variable '{treatment}' is not in the causal graph. "
            "Please make sure you provided the correct treatment and that there's at least "
            f"one causal path between '{treatment}' and '{outcome}'."
        )
        tc.print(msg)
        tc.set_returns(
            tool_return=msg,
            user_report=[msg],
        )
        return


    # Set L to empty list and N to all nodes in the causal graph.
    L = []
    N = list(causal_graph.nodes())

    # Call each adjustment set function safely.
    success = True
    try:
        optimal_adj = causal_graph.optimal_adj_set(treatment, outcome, L, N)
        tc.print("Optimal adjustment set:", optimal_adj)
    except Exception as e:
        tc.print("Error in optimal_adj_set:", e)
        optimal_adj = {}
        success = False
    try:
        optimal_minimal_adj = causal_graph.optimal_minimal_adj_set(treatment, outcome, L, N)
        tc.print("Optimal minimal adjustment set:", optimal_minimal_adj)
    except Exception as e:
        tc.print("Error in optimal_minimal_adj_set:", e)
        optimal_minimal_adj = {}
        success = False
    try:
        optimal_minimum_adj = causal_graph.optimal_minimum_adj_set(treatment, outcome, L, N)
        tc.print("Optimal minimum adjustment set:", optimal_minimum_adj)
    except Exception as e:
        tc.print("Error in optimal_minimum_adj_set:", e)
        optimal_minimum_adj = {}
        success = False

    results = {
        "optimal adjustment set": optimal_adj,
        "optimal minimal adjustment set": optimal_minimal_adj,
        "optimal minimum adjustment set": optimal_minimum_adj,
    }

    tc.print("Optimal adjustment sets computed.")
    # check for empty sets
    if optimal_adj == {} and optimal_minimal_adj == {} and optimal_minimum_adj == {} and success:
        tc.print("""Optimal adjustment sets are empty. This likely means there are no non-causal backdoor paths linking the treatment and outcome.\
Check the causal graph to see if this is true by looking for causal paths between the treatment and the outcome. If there is only one causal path between \
treatment and outcome, we will simply regress on the treatment as it has a direct causal link.""")


    tc.set_returns(
        tool_return="Optimal adjustment sets computed.",
        user_report=[
            f"Optimal adjustment set: {optimal_adj}",
            f"Optimal minimal adjustment set: {optimal_minimal_adj}",
            f"Optimal minimum adjustment set: {optimal_minimum_adj}",
        ],
    )
    return results

class ComputeOptimalAdjSets(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        dag_pickle_path = os.path.join(self.working_directory, kwargs["dag_pickle_path"])
        treatment = kwargs["treatment"]
        outcome = kwargs["outcome"]
        thrd, out_stream = execute_tool(
            compute_adj_sets,
            dag_pickle_path=dag_pickle_path,
            treatment=treatment,
            outcome=outcome,
            # ---
            wd=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "compute_optimal_adj_sets"

    @property
    def description(self) -> str:
        return (
            "Loads a serialized DAG as a causal graph and computes optimal adjustment sets using three functions: "
            "optimal_adj_set, optimal_minimal_adj_set, and optimal_minimum_adj_set. "
            "This tool always uses an empty list for L and sets N to all features in the dataset."
        )

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dag_pickle_path": {
                            "type": "string",
                            "description": "Path to the serialized DAG (pickle file).",
                        },
                        "treatment": {
                            "type": "string",
                            "description": "Name of the treatment variable.",
                        },
                        "outcome": {
                            "type": "string",
                            "description": "Name of the outcome variable.",
                        },
                    },
                    "required": ["dag_pickle_path", "treatment", "outcome"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "Loads a serialized DAG as a causal graph and computes optimal adjustment sets. "
            "This tool always uses an empty list for L and considers all features in the dataset for N. "
            "It returns a dictionary with keys 'optimal adjustment set', 'optimal minimal adjustment set', and "
            "'optimal minimum adjustment set'."
        )

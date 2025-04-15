import os
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import pickle as pkl
import numpy as np
import json
from causallearn.graph import GeneralGraph


from .plot_dag import plot_dag_graphviz
from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

def is_acyclic_cpdag(adj_matrix: np.ndarray) -> bool:
    """
    Check if a directed graph (given by its adjacency matrix) is acyclic,
    ignoring undirected edges.

    In causal-learn's CPDAG encoding:
      - A directed edge i --> j is represented as:
            adj_matrix[i, j] == -1   and   adj_matrix[j, i] == 1.
      - An undirected edge between i and j is represented as:
            adj_matrix[i, j] == -1   and   adj_matrix[j, i] == -1.
      - To follow an edge from i to j, we look for a true directed edge.
    
    Returns True if the graph is acyclic, False otherwise.
    """
    n = adj_matrix.shape[0]
    visited = [False] * n
    rec_stack = [False] * n

    def dfs(v: int) -> bool:
        visited[v] = True
        rec_stack[v] = True
        for u in range(n):
            # Check for a directed edge: v -> u should have:
            # adj_matrix[v, u] == -1 and adj_matrix[u, v] == 1.
            if adj_matrix[v, u] == -1 and adj_matrix[u, v] == 1:
                if not visited[u]:
                    if dfs(u):
                        return True
                elif rec_stack[u]:
                    return True
        rec_stack[v] = False
        return False

    for node in range(n):
        if not visited[node]:
            if dfs(node):
                return False  # cycle found
    return True  # no cycles found

def find_undirected_edges(cpdag):
    n = cpdag.graph.shape[0]
    undirected_edges = []
    
    # Identify undirected edges: if both i->j and j->i are -1, treat as undirected
    for i in range(n):
        for j in range(i+1, n):
            if cpdag.graph[i, j] == -1 and cpdag.graph[j, i] == -1:
                undirected_edges.append((i, j))
    undirected_edges_names = [(cpdag.node_names[i], cpdag.node_names[j]) for i, j in undirected_edges]
    return undirected_edges_names, undirected_edges


def cpdag_to_json(cpdag: GeneralGraph) -> str:
    """
    Convert a CPDAG (instance of GeneralGraph) from causal-learn into a JSON structure.

    The JSON structure will have:
      - "nodes": a list of node names.
      - "directed_edges": a list of edges represented as {"from": source, "to": target}.
      - "undirected_edges": a list of undirected edges represented as {"node1": n1, "node2": n2}.

    We use the following logic:
      - If cpdag.is_directed_from_to(node1, node2) is True, then there is a directed edge from node1 to node2.
      - If cpdag.is_directed_from_to(node2, node1) is True, then there is a directed edge from node2 to node1.
      - If neither is true but cpdag.is_undirected_from_to(node1, node2) is True, then there is an undirected edge.
    """
    # Get the list of nodes' names
    nodes = cpdag.get_node_names()
    directed_edges = []
    undirected_edges = []

    num_nodes = cpdag.get_num_nodes()

    # Iterate over all unique pairs of nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            node1 = cpdag.nodes[i]
            node2 = cpdag.nodes[j]

            # Check for a directed edge between node1 and node2
            if cpdag.is_directed_from_to(node1, node2):
                directed_edges.append({"from": node1.get_name(), "to": node2.get_name()})
            elif cpdag.is_directed_from_to(node2, node1):
                directed_edges.append({"from": node2.get_name(), "to": node1.get_name()})
            # Check for an undirected edge (should be mutually exclusive with a directed edge)
            elif cpdag.is_undirected_from_to(node1, node2):
                undirected_edges.append({"node1": node1.get_name(), "node2": node2.get_name()})

    graph_dict = {
        "nodes": nodes,
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges
    }
    
    return json.dumps(graph_dict, indent=4)


def update_dag(
    tc: ToolCommunicator,
    dag_pickle_path: str,
    node1: str,
    node2: str,
    new_direction: str,  # Expected: "node1->node2" or "node2->node1" or "node1-x-node2"
    nodes_to_remove: Optional[List[str]] = None,
) -> None:
    """
    Loads a serialized CPDAG or PAG from dag_pickle_path, updates the edge between node1 and node2
    with the new direction, and optionally removes unnecessary nodes from the graph.
    
    For CPDAGs, a directed edge i -> j is encoded as:
        dag.graph[i, j] = -1   and   dag.graph[j, i] = 1.
    For PAGs, we assume that a fully directed edge is encoded the same way, but extra
    endpoint markings (e.g. circles) may be present on ambiguous edges.
    
    If the graph is a CPDAG, the function verifies acyclicity (using the underlying dag.graph array).
    For PAGs, acyclicity is not enforced.
    
    The updated graph is saved as a pickle file and plotted as a PNG.
    
    Parameters:
      - tc: ToolCommunicator for printing and setting returns.
      - dag_pickle_path: Path to the serialized CPDAG/PAG (pickle file).
      - node1: Name of the first node.
      - node2: Name of the second node.
      - new_direction: New orientation for the edge. Must be either "node1->node2" or "node2->node1" or "node1-x-node2".
      - nodes_to_remove: Optional list of node names to remove from the graph.
    """
    dag_pickle_path = Path(dag_pickle_path)
    
    # Load the serialized graph (CPDAG or PAG).
    with open(dag_pickle_path, "rb") as f:
        dag = pkl.load(f)
    
    # Retrieve node names.
    try:
        node_names = list(dag.node_names)
    except AttributeError:
        try:
            node_names = list(dag.graph.nodes())
            dag.node_names = node_names  # set for consistency
        except Exception as e:
            tc.print("Failed to obtain node names from dag.graph:", e)
            return

    # Get indices for the specified nodes.
    try:
        idx1 = node_names.index(node1)
        idx2 = node_names.index(node2)
    except ValueError as e:
        tc.print(f"Error: One or both nodes not found in the graph's node names: {e}")
        return

    # Check if the graph is a PAG. We assume that if dag has an attribute "pag" set to True,
    # then the graph is a PAG. Otherwise, it is assumed to be a CPDAG.
    is_pag = hasattr(dag, "pag") and dag.pag

    # Update the edge based on the provided new_direction.
    # For a fully directed edge (both CPDAG and PAG), we use:
    #   For node1 -> node2: set dag.graph[idx1, idx2] = -1 and dag.graph[idx2, idx1] = 1.
    if new_direction == f"{node1}->{node2}":
        dag.graph[idx1, idx2] = -1
        dag.graph[idx2, idx1] = 1
    elif new_direction == f"{node2}->{node1}":
        dag.graph[idx1, idx2] = 1
        dag.graph[idx2, idx1] = -1
    elif new_direction == f"{node1}-x-{node2}" or new_direction == f"{node2}-x-{node1}":
        dag.graph[idx1, idx2] = 0
        dag.graph[idx2, idx1] = 0
    else:
        tc.print("The new_direction parameter must be either 'node1->node2', 'node2->node1, or 'node1-x-node2', with node names matching.")
        return


    # Remove nodes if requested.
    if nodes_to_remove is not None:
        indices_to_remove = []
        for node in nodes_to_remove:
            try:
                idx = node_names.index(node)
                indices_to_remove.append(idx)
            except ValueError:
                tc.print(f"Node '{node}' not found in the graph's node names.")
        # Remove nodes in descending order to avoid index shifting.
        for idx in sorted(indices_to_remove, reverse=True):
            dag.graph = np.delete(dag.graph, idx, axis=0)
            dag.graph = np.delete(dag.graph, idx, axis=1)
            del node_names[idx]
        dag.node_names = node_names

    # For CPDAGs, check acyclicity using the underlying NumPy array.
    if not is_pag:
        if not is_acyclic_cpdag(dag.graph):
            tc.print("ERROR: The updated graph is not acyclic. Please ensure the updated edge does not introduce a cycle.")
            tc.set_returns(
                tool_return="ERROR: The updated graph is not acyclic. Please ensure the updated edge does not introduce a cycle.",
                user_report=["ERROR: The updated graph is not acyclic. Please ensure the updated edge does not introduce a cycle."]
            )
            return

    old_stem = dag_pickle_path.stem
    n = 1
    if old_stem.split("_")[-1].isdigit():
        n = int(old_stem.split("_")[-1]) + 1
        old_stem = "_".join(old_stem.split("_")[:-1])

    new_filename = f"{old_stem}_{n}.pkl"
    new_file_path = dag_pickle_path.parent / new_filename

    plot_filename = f"{old_stem}_{n}.png"
    plot_file_path = dag_pickle_path.parent / plot_filename

    tc.print(dag)

    # Plot the updated graph.
    # The updated plot_dag_graphviz function will examine the underlying matrix and any 'pag' flag.
    plot_dag_graphviz(dag.graph, node_names, str(plot_file_path))

    # Serialize the updated graph.
    with open(new_file_path, "wb") as f:
        pkl.dump(dag, f)

    dag_json = cpdag_to_json(dag)

    tc.print("Updated graph has been serialized and saved at:", new_file_path)
    tc.set_returns(
        tool_return=(
            f"Edge between {node1} and {node2} updated to {new_direction}. "
            f"Updated graph has been saved to {new_file_path} and plotted to {plot_file_path}"
        ),
        user_report=[
            f"Updated graph saved at: {new_file_path}",
            f"Plot saved at: {plot_file_path}"
            f"dag_json: {dag_json}",
        ],
    )

class UpdateDAG(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        dag_pickle_path = os.path.join(self.working_directory, kwargs["dag_pickle_path"])
        node1 = kwargs["node1"]
        node2 = kwargs["node2"]
        new_direction = kwargs["new_direction"]
        nodes_to_remove = kwargs.get("nodes_to_remove")
        thrd, out_stream = execute_tool(
            update_dag,
            dag_pickle_path=dag_pickle_path,
            node1=node1,
            node2=node2,
            new_direction=new_direction,
            nodes_to_remove=nodes_to_remove,
            wd=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "update_dag"

    @property
    def description(self) -> str:
        return (
            "Loads a serialized CPDAG/PAG, updates the specified edge with a new direction, "
            "optionally removes unnecessary nodes, and saves the updated graph. "
            "The updated file is saved as a pickle and a PNG plot is generated."
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
                            "description": "Path to the serialized CPDAG/PAG (pickle file).",
                        },
                        "node1": {
                            "type": "string",
                            "description": "Name of the first node of the edge to update.",
                        },
                        "node2": {
                            "type": "string",
                            "description": "Name of the second node of the edge to update.",
                        },
                        "new_direction": {
                            "type": "string",
                            "description": (
                                """
The new direction for the edge. Must be one of 'node1->node2', 'node2->node1', 'node1-x-node2' or 'node2-x-node1'.
'node1->node2' or 'node2->node1' will set a directed edge between node1 and node2 in the specified direction.
'node1-x-node2' or 'node2-x-node1' will set an undirected edge between node1 and node2.
"""
                            ),
                        },
                        "nodes_to_remove": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of node names to remove from the graph.",
                        },
                    },
                    "required": ["dag_pickle_path", "node1", "node2", "new_direction"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "Loads a serialized CPDAG/PAG, updates the specified edge's direction, "
            "optionally removes unnecessary nodes, and saves the updated graph. "
            "The updated file will be named as: "
            "f'{old_stem}_{node1}_{node2}_edge_updated{{nodes_removed_suffix}}.pkl', "
            "and a PNG plot of the updated graph will also be saved."
        )

    @property
    def logs_useful(self) -> bool:
        """Return `True` if the logs of this tool are *especially* useful for the LLM to understand what has been done.

        This will be used by the engine to determine whether to shorten the logs if needed for token reasons etc. This
        is up to the engine's discretion, this property just provides a hint.

        The user will always be able to see the full logs.

        Returns:
            bool: `True` if the logs of this tool are *especially* useful for the LLM to understand what has been done.
        """
        return True
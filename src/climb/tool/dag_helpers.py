from itertools import product
import json
from causallearn.graph import GeneralGraph


def is_acyclic(adj_matrix):
    """
    Check if a directed graph (given by its adjacency matrix) is acyclic.
    
    In causal-learn's encoding:
      - A directed edge i --> j is represented as:
            adj_matrix[i, j] == -1   and   adj_matrix[j, i] == 1.
      - Thus, to follow an edge from i to j, we look for adj_matrix[i, j] == -1.
    
    Returns True if the graph is acyclic, False otherwise.
    """
    n = adj_matrix.shape[0]
    visited = [False] * n
    rec_stack = [False] * n

    def dfs(v):
        visited[v] = True
        rec_stack[v] = True
        # For each neighbor u, if there is an edge v --> u (i.e. adj_matrix[v, u] == -1), follow it.
        for u in range(n):
            if adj_matrix[v, u] == -1:  # means v -> u
                if not visited[u]:
                    if dfs(u):
                        return True  # cycle detected
                elif rec_stack[u]:
                    return True  # cycle detected
        rec_stack[v] = False
        return False

    for node in range(n):
        if not visited[node]:
            if dfs(node):
                return False  # cycle found, so graph is not acyclic
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

def enumerate_dags(cpdag):
    """
    Enumerate all DAGs that are consistent with the CPDAG.
    
    cpdag: an object with an attribute 'graph', a NumPy array representing the CPDAG.
           In causal-learn's encoding:
             - For a directed edge i --> j: cpdag.graph[i, j] = -1 and cpdag.graph[j, i] = 1.
             - For an undirected edge i --- j: cpdag.graph[i, j] = cpdag.graph[j, i] = -1.
    
    Returns a list of adjacency matrices, each representing a valid DAG.
    """
    undirected_edges_names, undirected_edges = find_undirected_edges(cpdag)
    
    all_possible_dags = []
    # Iterate over all possible orientations (2^(number of undirected edges))
    for directions in product([0, 1], repeat=len(undirected_edges)):
        # Create a copy of the CPDAG's adjacency matrix
        new_graph = cpdag.graph.copy()
        # Assign a direction for each undirected edge
        for idx, (i, j) in enumerate(undirected_edges):
            if directions[idx] == 0:
                # Orient as i --> j:
                # Set: new_graph[i, j] = -1 and new_graph[j, i] = 1.
                new_graph[i, j] = -1
                new_graph[j, i] = 1
            else:
                # Orient as j --> i:
                # Set: new_graph[i, j] = 1 and new_graph[j, i] = -1.
                new_graph[i, j] = 1
                new_graph[j, i] = -1
        
        # Check if the resulting graph is a DAG (acyclic) using the corrected DFS.
        if is_acyclic(new_graph):
            all_possible_dags.append(new_graph)
    
    return all_possible_dags


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
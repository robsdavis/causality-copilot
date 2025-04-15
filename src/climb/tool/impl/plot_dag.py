import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout


def plot_dag_graphviz(adj_matrix, node_names, filepath):
    """
    Plot the graph (CPDAG or PAG) with color-coded edges.
    
    For CPDAGs (DAGs):
      - Directed edges (i -> j) are drawn in black with arrowheads.
      - Undirected edges (i --- j) are drawn in red, dashed, with no arrowheads.
    
    For PAGs:
      - Fully directed edges (e.g. i -> j: adj[i,j]==-1, adj[j,i]==1) are drawn as above.
      - Partially directed edges (one endpoint is ambiguous, e.g. represented by 0)
        are drawn in blue with a dotted line and an arrowhead.
      - Ambiguous/undirected edges (e.g. both endpoints ambiguous or other combinations)
        are drawn in green with a dotted line.
    
    Assumes that if the graph is a PAG, the adj_matrix has an attribute 'pag' set to True.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout

    # Detect whether we are dealing with a PAG.
    is_pag = hasattr(adj_matrix, "pag") and adj_matrix.pag

    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    n = adj_matrix.shape[0]

    directed_edges = []
    undirected_edges = []
    partial_edges = []  # for partially oriented edges (only used for PAGs)

    if not is_pag:
        # CPDAG (or DAG) encoding:
        # - Directed: i -> j if adj[i, j] == -1 and adj[j, i] == 1.
        # - Undirected: i --- j if adj[i, j] == adj[j, i] == -1.
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    directed_edges.append((node_names[i], node_names[j]))
                elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1:
                    directed_edges.append((node_names[j], node_names[i]))
                elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                    undirected_edges.append((node_names[i], node_names[j]))
    else:
        # PAG encoding.
        # Here we assume the following (customize these rules as needed):
        # - Fully directed: (-1, 1) or (1, -1) (as in CPDAG).
        # - Partially directed: one side has -1 (or 1) and the other is 0.
        # - Ambiguous (undirected): both endpoints are 0 or any other combination.
        for i in range(n):
            for j in range(i + 1, n):
                val_ij = adj_matrix[i, j]
                val_ji = adj_matrix[j, i]
                if val_ij == -1 and val_ji == 1:
                    directed_edges.append((node_names[i], node_names[j]))
                elif val_ij == 1 and val_ji == -1:
                    directed_edges.append((node_names[j], node_names[i]))
                elif val_ij == -1 and val_ji == -1:
                    undirected_edges.append((node_names[i], node_names[j]))
                elif (val_ij == -1 and val_ji == 0) or (val_ij == 0 and val_ji == 1):
                    # Here we assume the arrow is partially oriented.
                    partial_edges.append((node_names[i], node_names[j]))
                elif (val_ij == 0 and val_ji == -1) or (val_ij == 1 and val_ji == 0):
                    partial_edges.append((node_names[j], node_names[i]))
                elif val_ij == 0 and val_ji == 0:
                    undirected_edges.append((node_names[i], node_names[j]))
                # Add other conditions as needed.

    # Add all edges to G (for layout purposes).
    for edge in directed_edges + undirected_edges + partial_edges:
        G.add_edge(*edge)

    # Compute layout.
    try:
        pos = graphviz_layout(G, prog='dot')
    except Exception as e:
        print("Graphviz layout failed, using spring layout:", e)
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Draw directed edges (black, with arrowheads).
    nx.draw_networkx_edges(
        G, pos, edgelist=directed_edges,
        edge_color='black',
        arrows=True,
        arrowstyle='->',
        arrowsize=30,
        width=2,
        min_source_margin=22,
        min_target_margin=22,
        connectionstyle='arc3, rad=0.5'
    )

    # Draw undirected edges.
    if not is_pag:
        # CPDAG: undirected edges in red, dashed.
        nx.draw_networkx_edges(
            G, pos, edgelist=undirected_edges,
            edge_color='red', arrows=False, style='dashed', width=2
        )
    else:
        # PAG: ambiguous/undirected edges in green, dotted.
        nx.draw_networkx_edges(
            G, pos, edgelist=undirected_edges,
            edge_color='green', arrows=False, style='dotted', width=2
        )
        # Draw partially directed edges.
        nx.draw_networkx_edges(
            G, pos, edgelist=partial_edges,
            edge_color='blue', arrows=True, arrowstyle='-|>', arrowsize=25,
            style='dotted', width=2
        )

    title_str = "CPDAG" if not is_pag else "PAG"
    plt.title(title_str)
    dag_plot = plt.gcf()
    dag_plot.savefig(filepath, format='png')
    return dag_plot

# ORIGINAL FUNCTION
# def plot_dag_graphviz(adj_matrix, node_names, filepath):
#     """
#     Plot the CPDAG with color-coded edges:
#       - Directed edges (i -> j) are drawn in black with arrowheads.
#       - Undirected (bidirectional) edges are drawn in red, dashed, with no arrowheads.
    
#     In causal-learn's encoding:
#       - i -> j: adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1.
#       - i --- j: adj_matrix[i, j] == adj_matrix[j, i] == -1.
#     """
#     G = nx.DiGraph()
#     G.add_nodes_from(node_names)
#     n = adj_matrix.shape[0]
    
#     directed_edges = []
#     undirected_edges = []
    
#     # Classify edges by inspecting pairs (i, j) with i < j.
#     for i in range(n):
#         for j in range(i+1, n):
#             if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
#                 # Directed edge: i -> j.
#                 directed_edges.append((node_names[i], node_names[j]))
#             elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1:
#                 # Directed edge: j -> i.
#                 directed_edges.append((node_names[j], node_names[i]))
#             elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
#                 # Undirected edge: add once (choose i -> j arbitrarily).
#                 undirected_edges.append((node_names[i], node_names[j]))
    
#     # Add all edges to the graph (for computing the layout).
#     for edge in directed_edges + undirected_edges:
#         G.add_edge(*edge)
    
#     try:
#         pos = graphviz_layout(G, prog='dot')
#     except Exception as e:
#         print("Graphviz layout failed, using spring layout:", e)
#         pos = nx.spring_layout(G, seed=42)
    
#     plt.figure(figsize=(10, 8))
    
#     # Draw nodes and labels.
#     nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000)
#     nx.draw_networkx_labels(G, pos, font_size=12)
    
#     # Draw directed edges with arrowheads.
#     nx.draw_networkx_edges(
#         G, pos, edgelist=directed_edges,
#         edge_color='black',
#         arrows=True,
#         arrowstyle='->',   # alternative arrow style
#         arrowsize=30,
#         width=2,
#         min_source_margin=22,  # margin at source node
#         min_target_margin=22,   # margin at target node
#         connectionstyle='arc3, rad=0.5'  # adds slight curvature so arrows are visible
#     )
    
#     # Draw undirected edges without arrowheads, in red and dashed.
#     nx.draw_networkx_edges(
#         G, pos, edgelist=undirected_edges,
#         edge_color='red', arrows=False, style='dashed', width=2
#     )
#     plt.title(f"CPDAG")
#     dag_plot = plt.gcf()
#     dag_plot.savefig(filepath, format='png')
#     return dag_plot
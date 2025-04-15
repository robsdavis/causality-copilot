import os
from typing import Any, Optional, Dict, List, Tuple
import pickle as pkl

import numpy as np
import pandas as pd
from pathlib import Path
from networkx.drawing.nx_pydot import graphviz_layout
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges

# from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from climb.tool.dag_helpers import enumerate_dags, find_undirected_edges, cpdag_to_json
from climb.tool.impl.plot_dag import plot_dag_graphviz
from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


# Dictionary mapping algorithm names to lambda wrappers that normalize the return value.
DAG_GEN_METHODS = {
    "pc": lambda data, node_names, alpha, indep_test: {"G": pc(
        data, alpha=alpha, node_names=node_names, indep_test=indep_test
    ).G},
    "ges": lambda data, node_names, alpha, indep_test: {"G": ges(
        data, node_names=node_names
    )["G"]},
    "fci": lambda data, node_names, alpha, indep_test: {"G": fci(
        data, alpha=alpha, node_names=node_names, indep_test=indep_test
    )[0]},
}

def generate_dag(
    tc,
    data_file_path: str,
    dag_pickle_path: str,
    workspace: str,
    dag_gen_method: str = "pc",  # options: "pc", "ges", "fci"
    indep_test: str = "kci",  # used for PC and FCI
    alpha: float = 0.05,
) -> None:
    workspace = Path(workspace)
    data_file_path = Path(data_file_path)
    dag_pickle_path = Path(dag_pickle_path)
    # Use an appropriate filename based on the dag_gen_method.
    dag_file_path = workspace / ("cpdag.png" if dag_gen_method.lower() != "fci" else "pag.png")

    # Load the dataset.
    df = pd.read_csv(data_file_path)
    node_names = list(df.columns)

    # Convert non-numeric columns to numeric using ordinal encoding.
    non_numeric_cols = df.select_dtypes(include=["object", "category"]).columns
    df[non_numeric_cols] = df[non_numeric_cols].apply(lambda x: pd.factorize(x)[0])

    # Convert the DataFrame to a NumPy array.
    data = df.to_numpy()

    # Get the causal discovery method from the dictionary.
    method = DAG_GEN_METHODS.get(dag_gen_method.lower())
    if method is None:
        raise ValueError(f"Unsupported dag_gen_method: {dag_gen_method}")

    # Run the chosen dag_gen_method.
    result = method(data, node_names, alpha, indep_test)

    # The normalized output graph is now available under result["G"].
    graph = result["G"]
    tc.print(graph)

    # Ensure node names are attached to the graph if not already present.
    if not hasattr(graph, "node_names"):
        graph.node_names = node_names

    # If using PC or GES (which produce a CPDAG), enumerate all consistent DAGs.
    if dag_gen_method.lower() in {"pc", "ges"}:
        undirected_edges, _ = find_undirected_edges(graph)
        tc.print("Undirected edges found:", undirected_edges)
        dag_list = enumerate_dags(graph)
        num_dags = len(dag_list)
        tc.print("Number of potential DAGs:", num_dags)
    else:
        # For FCI (which produces a PAG), DAG enumeration isn't directly applicable.
        dag_list = None
        num_dags = "N/A"

    # Plot the graph and save it as an image.
    plot_dag_graphviz(graph.graph, node_names, dag_file_path)

    # Serialize (pickle) the graph object to a file.
    with open(dag_pickle_path, "wb") as f:
        pkl.dump(graph, f)

    dag_json = cpdag_to_json(graph)

    tc.print("Graph has been serialized and saved at:", dag_pickle_path)
    report_message = (
        f"Graph generation has completed. A graph has been generated using {dag_gen_method.upper()} and saved as an image at: {dag_file_path}\n"
        f"This graph is consistent with {num_dags} possible DAGs.\n"
        f"It has also been serialized to: {dag_pickle_path}"
    )
    tc.set_returns(
        tool_return=report_message,
        user_report=[
            "📊 **Graph Output**",
            f"Plot saved at: {dag_file_path}",
            f"Serialized graph saved at: {dag_pickle_path}",
            f"DAG: {dag_json}",
        ],
    )




class GenerateDAGs(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        data_file_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            generate_dag,
            data_file_path=data_file_path,
            workspace=self.working_directory,
            dag_pickle_path=os.path.join(self.working_directory, kwargs["dag_pickle_path"]),
            dag_gen_method=kwargs.get("dag_gen_method", "pc"),
            indep_test="kci",
            alpha=0.05,
            # ---
            wd=self.working_directory,
            # background_forbidden_edges=None,
            # background_required_edges=None,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "generate_dag"

    @property
    def description(self) -> str:
        return """Produces the markov equivalence class of DAGs for the given dataset in the for of a CPDAG.
This CPDAG is plotted with color-coded edges:
- Directed edges (i -> j) are drawn in black with arrowheads.
- Undirected (bidirectional) edges are drawn in red, dashed, with no arrowheads.
The function returns the path to the generated plot and the list of all possible DAGs consistent with the CPDAG.
"""

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
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "dag_pickle_path": {"type": "string", "description": "Path to the serialised DAG."},
                        "dag_gen_method": {"type": "string", "description": "The method to use for DAG generation. Defaults to 'pc'. Available methods: 'pc', 'fci', 'ges'."},
                        "indep_test": {
                            "type": "string",
                            "description": """
Independence test to use in the PC algorithm. Defaults to 'kci'. Always use \
the defualt value unless instructed by the user to use a different test. Available tests: \
    - "fisherz": Fisher's Z conditional independence test
    - "chisq": Chi-squared conditional independence test
    - "gsq": G-squared conditional independence test
    - "kci": Kernel-based conditional independence test""",
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Significance level for conditional independence tests. Defaults to 0.05.",
                        },
                        # "background_forbidden_edges": {
                        #     "type": "array",
                        #     "items": {
                        #         "type": "array",
                        #         "items": {"type": "integer"},
                        #         "minItems": 2,
                        #         "maxItems": 2,
                        #     },
                        #     "description": "List of forbidden edges in the form of (i, j).",
                        # },
                        # "background_required_edges": {
                        #     "type": "array",
                        #     "items": {
                        #         "type": "array",
                        #         "items": {"type": "integer"},
                        #         "minItems": 2,
                        #         "maxItems": 2,
                        #     },
                        #     "description": "List of required edges in the form of (i, j).",
                        # },
                    },
                    "required": [
                        "data_file_path",
                        "dag_pickle_path",
                        # "background_forbidden_edges",
                        # "background_required_edges",
                    ],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return """Produces the markov equivalence class of DAGs for the given dataset in the for of a CPDAG.
This CPDAG is plotted with color-coded edges:
- Directed edges (i -> j) are drawn in black with arrowheads.
- Undirected (bidirectional) edges are drawn in red, dashed, with no arrowheads.
The function returns the path to the generated plot and the list of all possible DAGs consistent with the CPDAG.
"""

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
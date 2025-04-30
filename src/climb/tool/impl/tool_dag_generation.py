import os
import pickle as pkl
import logging
import importlib
import pkgutil
import inspect
from pathlib import Path

import pandas as pd

from climb.tool.dag_helpers import enumerate_dags, find_undirected_edges, cpdag_to_json
from climb.tool.impl.plot_dag import plot_dag_graphviz
from ..tool_comms import execute_tool
from ..tools import ToolBase

# —– DISCOVER & VALIDATE PLUGINS —–
DAG_GEN_METHODS = {}
_pkg_name = f"{__package__}.dag_gen_methods"
_pkg = importlib.import_module(_pkg_name)

for finder, module_name, ispkg in pkgutil.iter_modules(_pkg.__path__):
    module = importlib.import_module(f"{_pkg_name}.{module_name}")
    NAME = getattr(module, "NAME", None)
    run_fn = getattr(module, "run", None)

    if not (isinstance(NAME, str) and callable(run_fn)):
        logging.warning(f"Skipping plugin {module_name!r}: missing NAME or run()")
        continue

    sig = inspect.signature(run_fn)
    params = sig.parameters
    if "data" not in params or "node_names" not in params:
        logging.warning(
            f"Skipping plugin {module_name!r}: run() must accept at least 'data' and 'node_names'"
        )
        continue

    # wrap to enforce {"G":...} return
    def _make_validator(fn, plugin_name):
        def validated(**kwargs):
            out = fn(**kwargs)
            if not (isinstance(out, dict) and "G" in out):
                raise RuntimeError(
                    f"Plugin '{plugin_name}' must return a dict with key 'G'; got {out!r}"
                )
            return out
        return validated

    DAG_GEN_METHODS[NAME.lower()] = _make_validator(run_fn, NAME)

PLUGIN_KEYS = sorted(DAG_GEN_METHODS.keys())


def generate_dag(
    tc,
    data_file_path: str,
    dag_pickle_path: str,
    workspace: str,
    dag_gen_method: str = "pc",
    **plugin_args
) -> None:
    ws = Path(workspace)
    df = pd.read_csv(ws / data_file_path)
    node_names = list(df.columns)

    # encode non‐numerics
    for col in df.select_dtypes(["object", "category"]):
        df[col], _ = pd.factorize(df[col])
    data = df.to_numpy()

    KEY = dag_gen_method.lower()
    run_fn = DAG_GEN_METHODS.get(KEY)
    if run_fn is None:
        print(f"Available methods: {PLUGIN_KEYS}")
        raise ValueError(f"No DAG‐generation plugin named {dag_gen_method!r}")

    # inspect plugin signature and build only the args it declared
    sig = inspect.signature(run_fn)
    
    call_args = {}
    for name in sig.parameters:
        if name in plugin_args:
            call_args[name] = plugin_args[name]

    # run and validate
    call_args.update(dict(data=data, node_names=node_names))
    result = run_fn(**call_args)
    graph = result["G"]
    if not hasattr(graph, "node_names"):
        graph.node_names = node_names

    # enumerate if it’s a CPDAG (pc/ges)
    if KEY in {"pc", "ges"}:
        undirected, _ = find_undirected_edges(graph)
        tc.print("Undirected edges:", undirected)
        dags = enumerate_dags(graph)
        num = len(dags)
        tc.print("Number of consistent DAGs:", num)
    else:
        num = "N/A"

    # plot + pickle
    img_name = f"{KEY}_dag.png"
    img_path = ws / img_name
    plot_dag_graphviz(graph.graph, node_names, img_path)

    with open(ws / dag_pickle_path, "wb") as f:
        pkl.dump(graph, f)

    dag_json = cpdag_to_json(graph)
    tc.set_returns(
        tool_return=(
            f"Generated via {dag_gen_method.upper()}; "
            f"image at {img_path}; {num} possible DAGs; serialized at {dag_pickle_path}"
        ),
        user_report=[
            "📊 **Graph Output**",
            f"Plot: {img_path}",
            f"Pickle: {dag_pickle_path}",
            f"DAG JSON: {dag_json}",
        ],
    )


class GenerateDAGs(ToolBase):
    def _execute(self, **kwargs):
        data_fp = os.path.join(self.working_directory, kwargs.pop("data_file_path"))
        pickle_fp = os.path.join(self.working_directory, kwargs.pop("dag_pickle_path"))

        # Pass along dag_gen_method + any plugin-specific args
        thrd, out = execute_tool(
            generate_dag,
            data_file_path=data_fp,
            dag_pickle_path=pickle_fp,
            workspace=self.working_directory,
            dag_gen_method=kwargs.pop("dag_gen_method"),
            wd=self.working_directory,
            **kwargs,
        )
        self.tool_thread = thrd
        return out

    @property
    def name(self):
        return "generate_dag"

    @property
    def description(self):
        return (
            "Auto-discovers and runs a DAG-generation plugin from dag_gen_methods/.  "
            "Only 'data' and 'node_names' are core; all other parameters are plugin-specific."
        )

    @property
    def description_for_user(self):
        return (
            "Auto-discovers and runs a DAG-generation plugin from dag_gen_methods/.  "
            "Only 'data' and 'node_names' are core; all other parameters are plugin-specific."
        )

    @property
    def specification(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string"},
                        "dag_pickle_path": {"type": "string"},
                        "dag_gen_method": {
                            "type": "string",
                            "description": "One of the discovered plugin NAMEs",
                            "enum": PLUGIN_KEYS,
                        },
                    },
                    "required": ["data_file_path", "dag_pickle_path"],
                    "additionalProperties": True,
                },
            },
        }

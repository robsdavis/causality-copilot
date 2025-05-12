import os
import pickle as pkl
import logging
import importlib
import pkgutil
import inspect
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..tool_comms import execute_tool, ToolCommunicator, ToolReturnIter
from ..tools import ToolBase
from .ate_estimation_methods.ate_estimation_base import ATEEstimationBase

# Discover and register ATE estimation plugins
ATE_METHODS: dict[str, Any] = {}
_pkg_name = f"{__package__}.ate_estimation_methods"
_pkg = importlib.import_module(_pkg_name)

for finder, module_name, ispkg in pkgutil.iter_modules(_pkg.__path__):
    module = importlib.import_module(f"{_pkg_name}.{module_name}")
    plugin_cls = None
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ATEEstimationBase) and obj is not ATEEstimationBase:
            plugin_cls = obj
            break
    if plugin_cls is None:
        logging.warning(f"Skipping plugin {module_name!r}: no ATEEstimationBase subclass found")
        continue
    try:
        plugin_name = plugin_cls().NAME.lower()
    except Exception as e:
        logging.warning(f"Skipping plugin {module_name!r}: invalid NAME property: {e}")
        continue
    ATE_METHODS[plugin_name] = plugin_cls.run

METHOD_KEYS = sorted(ATE_METHODS.keys())


def estimate_ate(
    tc: ToolCommunicator,
    data_file_path: str,
    outcome: str,
    treatment: str,
    ground_truth_column: Optional[str] = None,
    n_runs: int = 10,
    ate_method: str = "t-learner",
    workspace: str = ".",
    **plugin_args: Any
) -> None:
    ws = Path(workspace)
    df = pd.read_csv(ws / data_file_path)

    # Extract Y, T, X
    Y = df[outcome].values
    T = df[treatment].values
    X = df[[c for c in df.columns if c not in [outcome, treatment]]].values

    # Optionally pull out per-unit ground truth for PEHE
    tau_true: Optional[np.ndarray] = None
    if ground_truth_column and ground_truth_column in df.columns:
        tau_true = df[ground_truth_column].values
        tc.print(f"Ground truth column '{ground_truth_column}' found; will compute PEHE.")
    else:
        tc.print("No ground truth column provided or not found; PEHE will not be computed.")

    # Find and call the chosen plugin
    KEY = ate_method.lower()
    run_fn = ATE_METHODS.get(KEY)
    if run_fn is None:
        raise ValueError(f"No ATE estimation plugin named {ate_method!r}. Available: {METHOD_KEYS}")

    # Build call args for the plugin
    sig = inspect.signature(run_fn)
    call_args: dict[str, Any] = {}
    locals_map = dict(Y=Y, T=T, X=X, n_runs=n_runs, tau_true=tau_true)
    for name in sig.parameters:
        if name in locals_map:
            call_args[name] = locals_map[name]
        elif name in plugin_args:
            call_args[name] = plugin_args[name]

    # Execute
    result = run_fn(**call_args)

    # Persist results
    out_name = f"tmp_ate_{KEY}.pkl"
    out_path = ws / out_name
    with open(out_path, "wb") as f:
        pkl.dump(result, f)

    # Build return message & report
    tool_return = (
        f"ATE estimated via {ate_method.upper()}: "
        f"mean={result['ate_mean']:.4f}, std={result['ate_std']:.4f}"
    )
    # Append PEHE if available
    if result.get("pehe_mean") is not None and not np.isnan(result["pehe_mean"]):
        tool_return += f"; PEHE={result['pehe_mean']:.4f}"
    tool_return += f"; results saved at {out_name}"

    user_report = [
        "📊 **ATE Estimation**",
        f"Mean ATE: {result['ate_mean']:.4f}",
        f"Std ATE: {result['ate_std']:.4f}",
    ]
    if result.get("pehe_mean") is not None and not np.isnan(result["pehe_mean"]):
        user_report += [
            f"Mean PEHE: {result['pehe_mean']:.4f}",
            f"Std PEHE: {result['pehe_std']:.4f}",
        ]
    user_report.append(f"Results pickle: {out_name}")

    tc.set_returns(tool_return=tool_return, user_report=user_report)


class EstimateATE(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        data_fp = os.path.join(self.working_directory, kwargs.pop("data_file_path"))
        ate_method = kwargs.pop("ate_method", None)
        thrd, out = execute_tool(
            estimate_ate,
            data_file_path=data_fp,
            workspace=self.working_directory,
            ate_method=ate_method,
            wd=self.working_directory,
            **kwargs,
        )
        self.tool_thread = thrd
        return out

    @property
    def name(self) -> str:
        return "estimate_ate"

    @property
    def description(self) -> str:
        return (
            "Estimate average treatment effects (ATE) using various meta-learners. "
            "Auto-discovers plugins in ate_estimation_methods/ and supports optional "
            "ground truth column for PEHE calculation."
        )

    @property
    def description_for_user(self) -> str:
        return (
            "Estimate ATE with a chosen meta-learner. Specify `ate_method` (e.g. 't-learner'), "
            "and `ground_truth_column` if your data includes true per-sample treatment effects, "
            "so PEHE can be computed."
        )

    @property
    def specification(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string"},
                        "outcome": {"type": "string"},
                        "treatment": {"type": "string"},
                        "ground_truth_column": {
                            "type": "string",
                            "description": "Optional name of column containing true individual treatment effects for PEHE.",
                        },
                        "n_runs": {"type": "integer", "default": 10},
                        "ate_method": {
                            "type": "string",
                            "enum": METHOD_KEYS,
                            "default": "t-learner",
                        },
                    },
                    "required": ["data_file_path", "outcome", "treatment", "ate_method"],
                    "additionalProperties": True,
                },
            },
        }

    @property
    def logs_useful(self) -> bool:
        return True
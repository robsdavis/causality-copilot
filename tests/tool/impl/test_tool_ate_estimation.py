import pytest
from utils import get_tool_output
from pathlib import Path

from climb.tool.impl.tool_ate_estimation import estimate_ate
from climb.tool.tool_comms import ToolCommunicator

# List of ATE estimation methods (plugin NAMEs)
ATE_METHODS = [
    "t-learner",
    "s-learner",
    "x-learner",
    "dragonnet",
    "tarnet",
    "bart",
    "cfrnet",
    "cf",
    "regression",
]

@pytest.mark.parametrize("ate_method", ATE_METHODS)
def test_ate_estimation(ate_method):
    """
    Smoke test for the estimate_ate tool: ensures each plugin runs without errors
    and returns a tool_return message.
    """
    mock_tc = ToolCommunicator()
    data_file = Path.cwd() / "data/causality/sodium_sbp/synthetic_hypertension_sodium_binary_data.csv"

    # Run with minimal arguments; ground_truth_column left as None
    estimate_ate(
        mock_tc,
        data_file_path=str(data_file),
        outcome="sbp_in_mmHg",
        treatment="Sodium",
        ground_truth_column=None,
        n_runs=2,
        ate_method=ate_method,
        workspace="."
    )

    output = get_tool_output(mock_tc)
    tool_return = output.tool_return
    # The return message should mention the chosen method and the mean ATE
    assert ate_method.upper() in tool_return.upper()
    assert "MEAN=" in tool_return.upper() or "ATE=" in tool_return.upper()

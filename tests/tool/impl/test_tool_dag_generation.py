import pytest
from utils import get_tool_output
from pathlib import Path

from climb.tool.impl.tool_dag_generation import generate_dag
from climb.tool.tool_comms import ToolCommunicator

@pytest.mark.parametrize(
    "dag_generation_method",
    [
        "camml",
        "direct-lingam",
        "fci",
        "ges",
        "ica-lingam",
        "notears",
        "pc",
        "score",
    ],
)
def test_dag_generation(df_classification_path, df_regression_path, dag_generation_method):
    """This tests the feature_selection() function in tools. X1, X2, X5 are features that
    are correlated with the target in their respective tasks and should be selected.
    Task coverage include classification, regression, and survival (TODO)"""

    mock_tc = ToolCommunicator()
    data_dir = Path.cwd() / "data/causality/sodium_sbp"
    data_file = data_dir / "synthetic_hypertension_sodium_binary_data_no_gt.csv"

    # Execute function with mock_tc
    generate_dag(
        mock_tc,
        data_file_path=data_file,
        dag_pickle_path=f"tmp_dag_{dag_generation_method}.pkl",
        workspace="./",
        dag_gen_method=dag_generation_method,
    )

    tool_return = get_tool_output(mock_tc).tool_return
    print(tool_return)

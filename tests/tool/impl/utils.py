from climb.tool.tool_comms import ToolCommunicator, ToolOutput

# NOTE: There's some issue with mocking ToolCommunicator; potentially due to the mixed
# usage of stdout and set_returns. This is why we need to instantiate one over the
# function and get its output with this helper function
def get_tool_output(tc: ToolCommunicator) -> ToolOutput:
    """Helper function to extract the ToolOutput from a ToolCommunicator."""
    p = tc.comm_queue.get()
    while isinstance(p, str):
        p = tc.comm_queue.get()
    if isinstance(p, ToolOutput):
        return p
    else:
        raise ValueError("ToolCommunicator missing ToolOutput")

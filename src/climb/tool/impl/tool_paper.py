import os
from typing import Any, Dict, List, cast

import pdfplumber
from openai import AzureOpenAI, OpenAI

from climb.common import Session
from climb.common.data_structures import UploadedFileAbstraction
from climb.engine.const import MODEL_MAX_MESSAGE_TOKENS
from climb.tool.impl.sub_agents import create_llm_client, get_llm_chat

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase, UserInputRequest

def upload_and_summarize_example_paper(
    tc: ToolCommunicator,
    paper_file_path: str,
    feature1: str,
    feature2: str,
    session: Session,
    additional_kwargs_required: Dict[str, Any],
) -> None:
    tc.print("Paper file uploaded successfully.")

    tc.print("Ingesting the paper PDF text...")
    paper_text = ""
    with pdfplumber.open(paper_file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            paper_text += f"\n\n{text}"

    tc.print(paper_text)

    # NOTE: Left-aligned text below to avoid spurious spaces/tabs.
    SYSTEM_PROMPT = """
YOU ARE:
You are an expert in healthcare and medicine, with substantial knowledge of medical data analysis and writing medical papers.
You are reading a scientific paper in the field of healthcare and medicine.
You are tasked with summarizing the content of the paper to assess the direction of an edge in a causal graph.
This means you must look to see if the paper contains evidence for a cause, effect, or other relationship between two variables.

YOUR TASK:
Extract evidence for a causal relationships between the two variables.

WHAT'S TRICKY:
You will be given a text representation of the paper, derived from the PDF.
This text is not perfect, and may contain errors, strange characters, or other issues.
Such problems will happen when converting a PDF to text, especially tables, figures, and other non-text elements.
Use your best understanding of PDF formatting to work around these issues.
"""
    FIRST_USER_MESSAGE = f"""
PAPER TEXT:

---
{paper_text}
---
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FIRST_USER_MESSAGE},
    ]

    client = create_llm_client(session=session, additional_kwargs_required=additional_kwargs_required)
    out_text = get_llm_chat(
        client=client,
        session=session,
        additional_kwargs_required=additional_kwargs_required,
        chat_kwargs={"messages": messages, "stream": False},
    )

    tc.set_returns(
        tool_return=f"Information pertaining to a relationship between {feature1} and {feature2}:\n\n{out_text}",
    )


class UploadAndSummarizeExamplePaper(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        # Handle the file upload.
        if self.user_input is None:
            raise ValueError("No user input obtained")
        if self.working_directory == "TO_BE_SET_BY_ENGINE":
            raise ValueError("Working directory not set")
        # for user_input in self.user_input:
        if self.user_input.kind == "file":
            uploaded_file = cast(UploadedFileAbstraction, self.user_input.received_input)
            paper_path = os.path.join(self.working_directory, uploaded_file.name)
            with open(paper_path, "wb") as f:
                f.write(uploaded_file.content)
        # else:
        #     raise NotImplementedError("Expected user input of kind 'file' but got something else.")

        thrd, out_stream = execute_tool(
            upload_and_summarize_example_paper,
            wd=self.working_directory,
            paper_file_path=paper_path,
            session=kwargs["session"],
            additional_kwargs_required=kwargs["additional_kwargs_required"],
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "upload_and_summarize_example_paper"

    @property
    def description(self) -> str:
        return """
        The user will be prompted to upload an example medical paper in PDF format via the UI.
        An assistant AI model will then be called to summarize the structure and style of the paper.
        This description will be added as an 'assistant' message for you to use. Note that the example paper may not \
        be a perfect match for our paper, so use this information as a guide.
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
                        "paper_file_path": {"type": "string", "description": "Path to the pdf file."},
                        "feature1": {"type": "string", "description": "The feature to assess the presence of a causal relationship with feature2."},
                        "feature2": {"type": "string", "description": "The feature to assess the presence of a causal relationship with feature1."},
                    },
                },
            },
        }

    @property
    def user_input_requested(self) -> List[UserInputRequest]:
        return [
            UserInputRequest(
                key="file", kind="file", description="Please upload your file", extra={"file_types": ["pdf"]}
            ),
        ]

    @property
    def description_for_user(self) -> str:
        return "allows you to upload an example paper in PDF format."

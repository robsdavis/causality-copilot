import copy
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import pydantic
import rich.pretty
import rich.prompt

from climb.common import (
    Agent,
    EngineParameter,
    EngineState,
    KeyGeneration,
    Message,
    ResponseKind,
    Session,
    ToolSpecs,
    UIControlledState,
)
from climb.common.utils import check_extra_available, d2m, engine_log, m2d, update_templates
from climb.db import DB
from climb.tool import list_all_tool_names, list_all_tool_specs

from ._azure_config import (
    AzureOpenAIConfig,
)
from ._code_execution import code_extract, is_code_generated
from ._engine import (
    ChunkTracker,
    EngineAgent,
    LoadingIndicator,
    StreamLike,
    tree_helpers,
)
from ._engine_openai import AzureOpenAIEngineMixin, OpenAIEngineBase

# DEBUG__PRINT_GET_MESSAGE_HISTORY: In _engine_openai.py
# DEBUG__PRINT_APPEND_MESSAGE: In _engine_openai.py
DEBUG__PRINT_MESSAGES_PER_AGENT = False
# ---
DEBUG__PRINT_MESSAGES_TO_LLM = False
DEBUG__PRINT_TOOLS = False
# ---
DEBUG__PRINT_DELTA = False
# ---
DEBUG__USE_FILTER_TOOLS = True  # Only True is supported in this engine.
# ---
DEBUG__PRINT_TOOL_FILTERING_BASED_ON_TOOL_SET = False

if DEBUG__USE_FILTER_TOOLS is False:
    raise RuntimeError("DEBUG__USE_FILTER_TOOLS must be True for this engine.")

# region: === TOOL SETS ===
TOOL_SETS = ["default", "full"]

if check_extra_available():
    TOOL_SETS.append("extra")

ToolSetParameter = EngineParameter(
    name="tool_set",
    description=(
        # TODO: Add more information.
        "The set of tools to be used by the engine."
    ),
    kind="enum",
    enum_values=TOOL_SETS,
    default="default",
)

# endregion

# region: === Prompt templates ===

# TODO: Make an engine parameter:
N_LOOKAHEAD = 3
# --- Preamble prompts:
LOG_PREAMBLE = "HERE IS THE LOG OF ALL THE AGENT-USER CONVERSATIONS SO FAR"
LAST_WORKER_LOG_PREAMBLE = "BELOW IS THE LOG OF THE WORKER AGENT AND USER CONVERSATION FOR THE *LAST TASK*"
# --- Indicators:
CONTINUE_INDICATOR = "YOUR WORK FROM HERE"
TASK_COMPLETED_INDICATOR = "TASK COMPLETED"
TASK_STOPPED_INDICATOR = "TASK STOPPED"
LONG_MESSAGE_SPLIT_INDICATOR = "= CONTINUE ="
# --- Replace markers:
WORKER_ACTUAL_TASK_REPLACE_MARKER = "{WORKER_ACTUAL_TASK}"
WD_CONTENTS_REPLACE_MARKER = "{WD_CONTENTS}"

EPISODE_DB_REPLACE_MARKER = "{EPISODE_DB}"
PLAN_REPLACE_MARKER = "{PLAN}"
PLAN_COMPLETED_REPLACE_MARKER = "{PLAN_COMPLETED}"
PLAN_REMAINING_REPLACE_MARKER = "{PLAN_REMAINING}"
LAST_EPISODE_REPLACE_MARKER = "{LAST_EPISODE}"
REASONING_STEP_REPLACE_MARKER = "{REASONING_STEP}"

MAX_CHARS_REPLACE_MARKER = "{MAX_CHARS}"

PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES = "{PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES}"
PRIVACY_MODE_SECTION_INDICATOR_RULES_1 = "{PRIVACY_MODE_SECTION_INDICATOR_RULES_1}"
PRIVACY_MODE_SECTION_INDICATOR_RULES_2 = "{PRIVACY_MODE_SECTION_INDICATOR_RULES_2}"

# --- warning and error prompts:
GENERATED_CODE_FORMAT_ERROR_MSG = "**IMPORTANT** Code execution failed due to wrong format of generated code"
PROBLEM_WITH_SUBTASK_STATUS_UPDATES_MARKER = "PROBLEM WITH SUBTASK STATUS UPDATES"
PROBLEM_WITH_SUBTASK_SELECTION_MARKER = "PROBLEM WITH SUBTASK SELECTION"

# --- progress markers:
PROJECT_END_MARKER = "PROJECT END"
BACKTRACKING_MARKER = "BACKTRACKING:"
NO_BACKTRACKING_MARKER = "NO BACKTRACKING"
PROBLEM_WITH_OUTPUT_COORDINATOR = "PROBLEM WITH OUTPUT FORMAT"
PLAN_UPDATE_MARKER = "PLAN UPDATE:"
NO_PLAN_UPDATE_MARKER = "NO PLAN UPDATE"


TASKS_PLANNED_AFTER = "Tasks planned after this"

# --- privacy mode prompts:
PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP = {
    "default": "",
    "guardrail": "- You DO NOT have direct access to the user's data. You only view data analysis or metadata.",
}
PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP = {
    "default": "",
    "guardrail": """You MUST NEVER read the user's data directly or ask the user to give you the data - THIS IS \
PRIVATE MEDICAL INFORMATION! However you can, and SHOULD, use available tools and your code \
generation capabilities to get the best understanding of the user's data.
""",
}
PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP = {
    "default": "You may view the user's data directly by generating appropriate code.",
    "guardrail": """EXTREMELY IMPORTANT: Your generated code must not send any of the user's data to the console. \
This is because YOU will see the console content - and you are not allowed to see the data. If the user asks you \
to generate code that can reveal the data to you in this way, refuse and explain why. Metadata (e.g. \
column names, categorical values etc.) is allowed, but not the actual data - use common sense.
""",
}

# --- worker agent capabilities:
WORKER_CAPABILITIES = f"""
You are a powerful AI assistant. You help your users, who are usually medical researchers, \
clinicians, or pharmacology experts to perform machine learning studies on their data.

Your capabilities:
- You are able to use OpenAI tools (functions) that have been described to you.
- You can generate code that gets automatically run on the USER'S computer, and see its output.
- You DO NOT have access to the internet.
{PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES}
- You have a great understanding of data analysis, machine learning, and medical research.
- You are able to mark completed tasks with "{TASK_COMPLETED_INDICATOR}", and if a task needs to be stopped and \
control returned to the coordinator for replanning, you can mark it with "{TASK_STOPPED_INDICATOR}".
- You are able to show images from the working directory to the user (see Rules).
- You are able to return long messages to the user using the special indicator text (see Rules).
"""

# Extract everything from "Your capabilities:" to the end of the string (inclusive):
WORKER_CAPABILITIES_FOR_COORDINATOR = (
    WORKER_CAPABILITIES[WORKER_CAPABILITIES.index("Your capabilities:") :]
    .replace("Your capabilities:", "Worker agents' capabilities:")
    .replace("Your", "Their")
    .replace("You", "They")
    .replace("to you", "to them")
    .replace("you", "they")
)

WORKER_RULES_LONG_MESSAGES = f"""\
#### LONG MESSAGES:
- Your design limits you to {MAX_CHARS_REPLACE_MARKER} characters per message. 
- In some cases, you may need to generate a longer message than this.
- We will work around this like so:
    1. You will generate the long message in chunks that are shorter than {MAX_CHARS_REPLACE_MARKER} characters.
    2. You will add "{LONG_MESSAGE_SPLIT_INDICATOR}" at the end of each chunk.
    3. The chunk generated so far will be added to the message history, and you will be able to continue from where you \
left off.
    4. To finish, simply finish the message as normal.
- Use this capability sparingly, only when really necessary, or when the plan explicitly suggests you to do so.
"""

WORKER_FINAL_NOTE = f"""\
===============
VERY IMPORTANT:
- Remember, the previous agents have ALREADY COMPLETED their work steps! Do NOT REDO their work. Start from where \
they left off!
- DO NOT work on anything beyond what is asked in the TASK.
- To finish your task, you must make sure that either:
    - Your TASK has been completed successfully and issue the {TASK_COMPLETED_INDICATOR} message, OR
    - There is a problem with the TASK and it needs to be stopped (see Rules), in which case you should issue the \
{TASK_STOPPED_INDICATOR} message.
- Always refer to CURRENT WORKING DIRECTORY CONTENTS to see which files are available to you!
===============
"""


EPISODE_DB = [
    {
        "episode_id": "ENV_1",
        "selection_condition": None,
        "status_reason": None,
        "episode_name": "Upload data files",
        "episode_details": """ 
- Introduce yourself as an AI assistant that will help the user with their Causality study.
- Ask the user if they have their data file ready as a CSV file.
- If the user has files ready, proceed to summoning the tool. Otherwise, STOP the task.
- Finally, summon the `upload_data_multiple_files` tool so that the user can upload their data file.
""",
        "coordinator_guidance": None,
        "worker_guidance": "Do not process the data files in any way during this step. We will handle that in the next steps.",
        "tools": ["upload_data_multiple_files"],
    },
    {
        "episode_id": "ENV_2",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Check hardware",
        "episode_details": """
Use the `hardware_info` tool to get information about the user's hardware. Using the report, determine whether the user's hardware is suitable for the task. As a rough guide, we want a machine with a CPU with at least 4 cores, 16GB of RAM, and a GPU with at least 4GB of memory. If the user's hardware is not suitable, suggest they find a machine that meets these requirements or use a cloud service, but allow the option to proceed anyway.
""",
        "coordinator_guidance": None,
        "worker_guidance": None,
        "tools": ["hardware_info"],
    },
    {
        "episode_id": "ENV_3",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Check data file can be loaded",
        "episode_details": """
Generate code to check whether the data file can be loaded with `pd.read_csv(<file_path>)`, as that is how the tools expect it. CHECK that the loaded dataframe has MORE THAN ONE column and more than one row - otherwise it usually means the separator or delimiter is wrong. Try to find a way to load the file (e.g. try different delimiters), and then save the modified file in a way that can be loaded with `pd.read_csv(<file_path>)`. If not possible, suggest to the user that they fix the data and upload it again.
""",
        "coordinator_guidance": None,
        "worker_guidance": """
- You MUST NOT use any tool here. DO NOT SUMMON ANY TOOLS.
- You MUST generate code in this step!
- So, your response MUST have:
DEPENDENCIES:
```
pandas
```
CODE:
```
... your code to complete the episode ...
```

- If the user has provided both a training and a test dataset files, you must check *both*.
""",
        "tools": [],
    },
    {
        "episode_id": "INFO_1",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "High-level information from the user",
        "episode_details": """
Ask the user whether they would like to provide high-level information about the dataset, especially:
    - How is it structured (what does a row or a column represent)?
    - Any background information about the data.
""",
        "coordinator_guidance": None,
        "worker_guidance": None,
        "tools": [],
    },
#     {
#         "episode_id": "INFO_3", # TODO: Is this needed? Customise to causality or delete
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Assess data suitability and tool support",
#         "episode_details": """
# Given what the user has told you, ask yourself two things:
#     Q1: Is the data suitable for the task?
#         > Example problem: the data is in a format that is not supported by the tools (e.g. time series data).
#         > Example problem: more than one row per patient, but the tools expect one row per patient.
#         > Think of any such problems...
#     Q2: Does the AutoPrognosis set of tools that you have access to support the task?
# - If the answer to Q1 is NO, think whether the data can be somehow transformed to fit the task. If you think this is \
# possible, suggest this to the user. If not, suggest how the user can get the right data.
# - If the answer to Q2 is NO, apologize to the user, mention that your capabilities are still being enhanced, but for \
# now this task cannot be performed.
# - If on the basis of the above you think the task CAN be performed, proceed to the next step. Otherwise, ask the \
# user if you can help them with anything else.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": """
# DO NOT actually execute the AutoPrognosis tools here! Use their specifications for your information, but DO NOT \
# invoke them!
# """,
#         "tools": [
#             "",# TODO: Causality tools
#         ],
#     },
#     {
#         "episode_id": "EDA_4",
#         "status_reason": None,
#         "selection_condition": "Only if data analysis reveals fewer than 50 samples",
#         "episode_name": "Warn about small sample size if necessary",
#         "episode_details": """
# ONLY IF there are fewer than about 50 samples, warn the user that the results may not be reliable as there is not \
# enough data. Allow to continue if the user is happy with that. SKIP this step DIRECTLY and COMPLETELY if there are more than \
# 50 samples.

# Note: this refers to the number of samples in the training dataset.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": None,
#         "tools": [],
#     },
    {
        "episode_id": "COL_1",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Column background information",
        "episode_details": """
- Generate code to load the dataset and print the column names.
- Ask the user if the dataset contains a ground truth treatment effect column. Explain that this is a column with the ground truth treatment effect for each sample. If the dataset \
does have a ground truth treatment effect column, then generate code to do the following:
    1. Load the dataset into a pandas dataframe.
    2. Save the dataset with the new name "<old_file_name>_original_with_gt_treatment_effect.csv" This is the dataset you should use when summoning the `estimate_ate`. For all other \
purposes you should use the "<old_file_name>_no_gt_treatment_effect.csv" file.
    3. Drop the ground truth treatment effect column from the database.
    4. Save the dataset with the new name "<old_file_name>_no_gt_treatment_effect.csv". This is the dataset you should for all other purposes a part from with the `estimate_ate` tool.
If the dataset does have a ground truth treatment effect column, explain that this if OK, but that we will not be able to measure the quality of our estimator using PEHE, like we could \
if the dataset did include it.

Then regardless, Go through EACH  of the other columns (excluding the ground truth treatment effect column) and gather background information.
    - IF you have some idea what the column represents, provide the user with a short summary of this. Ask the user if this is correct, and if not, ask them to provide the correct information.
    - IF you are not sure what the column represents, ask the user to provide this information about the column straight away.
Finally, list all the columns you have discussed (this will be all the columns if there's < 30) with their descriptions and ask the user if they are happy with this summary and would like to proceed with the next step.
""",
        "coordinator_guidance": None,
        "worker_guidance": """
- Do *NOT* ask about one column at a time, but rather go through several columns at once, so that the process is more efficient.
- If there is a reasonable number of columns (roughly < 30), ensure that you have gone through all of them.
- If there are many columns, focus on the most important ones, or the ones that are most likely to be relevant to the task. Once you have done this, ask the user if they would like to continue with the remaining columns.
- *ALWAYS* go over the columns whose meaning is not clear to you - and ask the user for clarification.
""",
        "tools": [],
    },
#     {
#         "episode_id": "DP-M_1", # TODO: Do we need to handle missing data?
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Represent missing data as NaN",
#         "episode_details": """
# Any missing data must be first represented as `numpy.nan` values.

# 1. Check the results from the EDA step to figure out if there are non-standard NaNs. If you have reason to suspect \
# this is the case, list for the user, what specific values you suspect are used to represent missing data. Ask the \
# user to confirm if these are indeed used to represent missing data.
# **Note:** if you suspect there are no non-standard NaNs, tell this to the user and confirm. If the user confirms, \
# you can move on with the plan.

# 2. Generate code to replace the non-standard NaNs with `numpy.nan` values. Save this modified dataset with the \
# suffix `_nan` in the filename. IF there is a test set, you must apply the same transformation to it.
# """,
#         "coordinator_guidance": "This task should come before DP-M_2,3,4 tasks, as it affects them.",
#         "worker_guidance": """
# **IMPORTANT**
# - Do NOT assume some "typical" non-standard NaN placeholders.
# - Always check based on the data analysis results, and CONFIRM wit the user what the placeholders are.
# Otherwise you could end up replacing values that are not actually missing data!
# """,
#         "tools": [],
#     },
#     {
#         "episode_id": "DP-M_2",# TODO: Do we need to handle missing data?
#         "status_reason": None,
#         "selection_condition": "Only if missing data is present",
#         "episode_name": "Consider dropping columns with high missing values",
#         "episode_details": """
# If there is any missing data in the dataset:
#     - Generate code to show per-column % missing values. Show these in descending order of % missing values.
#     - Suggest that as a rule of thumb any columns with 80%+ missing values should be removed. Ask the user what their \
# acceptable threshold is for including a column in the analysis.
#     - Ask the user if they are happy to drop the columns that exceed this threshold? Do they want to keep some \
# columns that exceed this threshold? If so, which ones?
#     - Given the user responses, generate code to drop the appropriate columns. IF there is a test set, you *must* apply \
# the same transformation to it.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": """
# **IMPORTANT**: When generating code to show per-column % missing values:
# Remember that pandas shortens the printed output of the dataframe.
# Use the statements:
# pd.options.display.max_rows = None
# pd.options.display.max_columns = None
# in your code, to ensure that all the results are shown!
# """,
#         "tools": [],
#     },
#     {
#         "episode_id": "DP-M_3",# TODO: Do we need to handle missing data?
#         "status_reason": None,
#         "selection_condition": "Only if missing data is present",
#         "episode_name": "Consider dropping rows with missing values",
#         "episode_details": """
# Generate code to show:
#     - per-column % missing values,
#     - % of total rows that have missing values.
# If there are still missing values in the dataset:
#     - If the user's intended TARGET VARIABLE has missing values, we must make sure to handle this. Ask them if \
# they are happy to drop rows with missing values in the target variable. Point out to the user that imputing the \
# target variable is not recommended.
#     - Ask the user if they are happy to drop rows with missing data (in any column), or if they are happy to use \
# an imputation tool (HyperImpute). Suggest that imputation is usually the better option, especially if most rows \
# have missing values. If the percentage of rows with missing values is high, strongly suggest using imputation \
# rather than dropping rows, due to the risk of losing valuable data.
#     - If they want to drop the rows, generate code to do so. IF there is a test set, you *must* apply the same \
# transformation to it.
#     - If they want to use HyperImpute, complete your task, the next agent will handle this.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": "You MUST NOT use HyperImpute tool! The next agent will handle this.",
#         "tools": [],
#     },
#     {
#         "episode_id": "DP-M_4",# TODO: Do we need to handle missing data?
#         "status_reason": None,
#         "selection_condition": "Only if missing data is present",
#         "episode_name": "Impute missing values",
#         "episode_details": """
# Step 1. Generate code to show:
#     - per-column % missing values,
#     - % of total rows that have missing values.
# Step 2. Explain to the user what the hyperimpute tool will do and check they are happy to use it.
# Step 3. If there are still missing values in the dataset:
#     - Use the `hyperimpute_imputation_train_test` tool (do not write own code for imputation).
# After imputation, generate code to show the per-column % missing values again, and confirm with the user that there \
# are no more missing values.
# """,
#         "coordinator_guidance": """
# **IMPORTANT**:
# - If the dataset has ANY missing values at all, we MUST IMPUTE THEM before we can proceed to the predictive modelling \
# stage. Hence, you MUST issue this episode IF there are missing values.
# """,
#         "worker_guidance": """
# * Use the LATEST version of the dataset, after all the previous steps have been completed. Check the conversation \
# history and the modification date-time of the files in the working directory to ensure this.
# * Save the imputed dataset(s) with a suffix `_hyperimputed` in the filename.

# * IF there is a test set, you *must* provide BOTH the training and test dataset paths to the tool. It will fit on the \
# training data and transform the test data.
# """,
#         "tools": ["hyperimpute_imputation_train_test"],
#     },
#     {
#         "episode_id": "DP-AM_1",
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Discuss data preprocessing with the user",
#         "episode_details": """
# Discuss with the user whether they have any particular data preprocessing steps in mind. work with them to generate \
# code to perform these steps. **IMPORTANT** Since your tools down the line support standard preprocessing like \
# normalization, feature selection, etc., DO NOT suggest these here (unless the user explicitly asks to do these). \
# This step is about whether the user has any specific data transformations in mind, that have medical meaning.
# """,
#         "coordinator_guidance": """
# This is a LONG episode and it involves a lot of back-and-forth with the user.
# The worker may have many things to do here, to satisfy the user needs.
# """,
#         "worker_guidance": """
# ###### User interaction:
# - **IMPORTANT:** You MUST keep asking the user if they have any MORE preprocessing steps in mind. The user may \
# want to do multiple things here! You must keep asking until the user says (1) they are finished or (2) they do not have any \
# more preprocessing steps in mind or (3) they want to go on. Only then can you proceed to the next episode (if any).
# ###### Data:
# - **IMPORTANT:** You must work from the latest version of the dataset, after missing data handling!
# - **IMPORTANT:** If the user has provided both a training and a test dataset, you must ensure that whatever \
# preprocessing steps you do on the training dataset, you must also do on the test dataset. Note that you must not \
# use the test dataset to inform the preprocessing steps on the training dataset - this is a critical best practice.
# """,
#         "tools": ["EDA", "descriptive_statistics"],
#     },
#     {
#         "episode_id": "DP-AM_4", # TODO: Needed?
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Outlier detection",
#         "episode_details": """
# - Describe to the user what the `outlier_detection` tool does and what it's useful for.
# - Ask the user if they want to use this tool to discover and remove outliers in the dataset.
# - Only if the user says yes, summon the `outlier_detection` tool, otherwise skip this step.
# - Use the `outlier_detection` tool to find and remove the outliers in the dataset.
# - Tell the user how many outliers were found and removed.
# - Ask the user if they are happy with the removal of the outliers. If yes, this stage is complete. \
# If no, ask them if they would like to revert to using the dataset they had before summoning the tool, with the outliers.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": """
# **IMPORTANT**
# If the task is survival analysis, then you must do the following BEFORE running the feature selection tool:
# 1. Confirm what is the TIME variable (the variable that has the time index representing event time)
# 2. Confirm what is the EVENT variable (the variable that represents the event itself, usually binary)
# Do NOT remove these columns in the feature selection step!

# If the user has provided both a training and a test dataset, summon the tool twice, once for the training dataset \
# and once for the test dataset.
# """,
#         "tools": ["outlier_detection"],
#     },
    {
        "episode_id": "DAG_1",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Generate CPDAG using algorithm",
        "episode_details": """
Ask the user what algorithm they would like to use to generate the CPDAG. The user can choose from:
    - PC (PC algorithm)
    - GES (Greedy Equivalence Search)
    - FCI (Fast Causal Inference)
    - CAMML (Causal Additive Model)
    - Direct-LiNGAM (Direct linear non-gaussian acyclic model)
    - ICA-LiNGAM (Independent component analysis - linear non-gaussian acyclic model)
    - Notears (Neural Causal Inference)
    - Score (Score-based algorithm)

If the user does not have a preference, use the GES algorithm by default.

Summon the `generate_dag` tool to generate a CPDAG (Completed Partially Directed Acyclic Graph), using the method the user has specified.
This graph shows all the possible DAGs from the dataset columns based on conditional independence tests.

The `generate_dag` tool prints the graph in JSON with the following format:
```
{
    "nodes": ["A", "B", "C"],
    "directed_edges": [
        {"from": "A", "to": "B"},
        {"from": "B", "to": "C"},
        ...
    ],
    "undirected_edges": [
        {
            "node1": "A",
            "node2": "C"
        }
    ]
}
```

Re-state the undirected edges to the user in the format: {node1} --- {node2}.

Explain to the user the in the graph plot black lines represent directed edges, while red dashed lines represent undirected edges.
""",
        "coordinator_guidance": None,
        "worker_guidance": None,
        "tools": ["generate_dag"],
    },
# ----
# Find literature
# - Review the graph that we printed in the previous step and order the undirected edges by number of edges that are \
# connected to them. Remember undirected edges are printed with "{node1} --- {node2}". Directed edges have arrows, i.e. "--->" and should be ignored in this step. \
# Start with the edge that has the most connections. Show the user the ordered list of undirected edges and ask them \
# if they are happy to proceed with directing these edges in this order.
#     {
#         "episode_id": "LIT_1", # Retrieve papers from web search TODO: Episode in progress
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Retrieve papers to evidence edge directions",
#         "episode_details": """
# 1. Look at the tool logs from where the latest DAG was generated or updated. To find the undirected edges in the DAG look at the logs from the `update_dag` or `generate_dag` tools. \
# These tools print the graph in JSON with the following format:
# ```
# {
#     "nodes": ["A", "B", "C"],
#     "directed_edges": [
#         {"from": "A", "to": "B"},
#         {"from": "B", "to": "C"},
#         ...
#     ],
#     "undirected_edges": [
#         {
#             "node1": "A",
#             "node2": "C"
#         }
#     ]
# }
# ```
# 2. Show the user the list of undirected edges.
# 3. Explain to the user that you are going to establish the causal relationships between the features in the dataset that \
# are joined by an undirected edge in the DAG.
# 4. Summon the `retrieve_scientific_literature` tool to retrieve the literature for every undirected edge. This saves the literature into a JSON file for the next step.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": "Do not generate your own code to load the json file, you must use the `upload_json_file` tool to complete this task.",
#         "tools": ["retrieve_scientific_literature"],
#     },
    {
        "episode_id": "LIT_1", # Upload literature json file
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Upload literature json file",
        "episode_details": """
- Given the discussion so far, and the logs of the tools (generate_dag and update_dag) you have seen, decide whether there are any undirected edges in the DAG.
- If there are no undirected edges in the DAG, Ask the user if they are happy with the DAG and if they are ready to proceed to the next step. If they are happy, proceed \
to the next step. If they are not happy, ask them what they would like to change.
- If there are undirected edges, explain to the user that you are going to establish the causal relationships between the features in the dataset that \
are joined by an undirected edge in the DAG. To do this you will need to review the scientific literature.
- Ask them to upload the json file that contains the relevant literature about the dataset. To allow this,
summon the `upload_json_file` tool.
""",
        "coordinator_guidance": None,
        "worker_guidance": "Do not generate your own code to load the json file, you must use the `upload_json_file` tool to complete this task.",
        "tools": ["upload_json_file"],
    },
    {
        "episode_id": "LIT_2", # TODO: Episode in progress
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Iteratively update DAG with edge direction evidence scoring",
        "episode_details": """
In this episode we will repeat steps 1 to 10 until there are no more undirected edges in the DAG.
1. Look at the tool logs from where the latest DAG was generated or updated. To find the undirected edges in the DAG look at the logs from the `update_dag` or `generate_dag` tools. \
These tools print the graph in JSON with the following format:
```
{
    "nodes": ["A", "B", "C"],
    "directed_edges": [
        {"from": "A", "to": "B"},
        {"from": "B", "to": "C"},
        ...
    ],
    "undirected_edges": [
        {
            "node1": "A",
            "node2": "C"
        }
    ]
}
```
2. Show the user the list of undirected edges.
3. Generate code to load the json file that the user has uploaded in the previous step. Read the data in the file, and print it to the console.
4. Review the literature that the user has provided. Collate evidence for a causal effect that implies the edge should be directed in one direction or the other. \
To do this you are looking for evidence of a causal relationship between the two variables. Use your best understanding of the variable from the column descriptions discussed earlier.\
Evidence of a causal relationship is one where a change in one variable causes a change in another variable. Look at the evidence for each direction of each edge and decide which direction is more likely \
given the evidence you have found. You may also use your understanding of the data and the domain to help you decide on the direction of the edge in the DAG.
5. You must then suggest a direction for every undirected edge. Each direction must be provided with an associated evidence score.
6. Show the user all the suggested directions for all the undirected edges with there associated evidence score.
7. Look at your edge suggestions, the evidence you provide for them and the literature a second time. Make sure they are correct and meet all the necessary criteria below:
    - The evidence must contain assertions that one of the edge nodes being assessed has a causal effect on the other.
    - The evidence supports the causal direction you have suggested.
    - The evidence you present is the strongest evidence supplied in the evidence JSON. Bear in mind that there may be some equivocal sentences that could be interpreted either way, \
before you find much more definitive evidence later in the literature. 
Once you have reflected and you are sure all directions are correct step 7 is complete and you can proceed to step 8.
8. Ask the user if they are happy with each of the suggested directions for the undirected edges. 
9. Print all the previously undirected edges with their new directions that the user has confirmed in the format: {node1}->{node2}.
10. Summon the `update_dag` tool to update the DAG with the new directions that the user has confirmed in the previous step. \
You will need to summon the tool once with all the edge updates in a list of dictionaries (one edge per dictionary). When you update the DAG, you must make sure you update the most \
recent version of the DAG.
11. If the list of undirected edges is not empty, now go back to step 1 of this episode and repeat the process (this time in step 1 you will need to check for undirected Graph Edges \
in the update_DAG logs from step 10 of this episode) until there are no more undirected edges in the DAG. If the list of undirected edges is empty, \
you can proceed to the step 12.
12. Ask the user if they are happy that all the edges have been directed in the updated DAG and are ready to proceed to the next step.
""",
        "coordinator_guidance": None,
        "worker_guidance": """\
- The json file should be of the following format:
```
[
    {
        "title": "Title of the paper",
        "authors": "Authors of the paper",
        "year": "Year of publication",
        "doi": "DOI of the paper",
        "abstract": "Abstract of the paper",
        "Full Text": "Text of the paper",
        "Journal": "Journal the paper was published in",
        "Citation number": "Number of citations the paper has",
    },
    {...}
]
```
It is possible that not all fields are present in the json file. If a field is missing, you should still be able to \
load the file.

When creating a score for the evidence, consider the following:
- Evidence found in the title is stronger than evidence found in the abstract, which is stronger than evidence found in the full text.
- Independently, evidence found in a paper with a high number of citations is stronger than evidence found in a paper with a low number of citations.
- Independently, evidence found in a high impact journal is stronger than evidence found in a low impact journal.
- The score should be a number between 0 and 1, where 0 is weak evidence and 1 is strong evidence.
You MUST NOT generate your own code to complete this task, consider the evidence and generate a score yourself for each edge just based on the \
information provided in the literature and the criteria above.
NEVER ask the user if they are happy with the score, only if they are happy with the direction you have suggested for the edge.
BE ABLE to explain the score you have given to the user if and only if they ask.

To update the DAG:
Do not generate your own code to update the DAG, you must use the `update_dag` tool.
""",
        "tools": ["update_dag"],
    },
#     {
#         "episode_id": "LIT_3", # TODO: Episode in progress
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Update dag with new directions",
#         "episode_details": """
# - Summon the `update_dag` tool to update the CPDAG with the new directions that the user has confirmed in the previous step.
# You will need to summon the tool for each edge that the user has confirmed.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": "Do not generate your own code to complete this task, you must use the `update_dag` tool.",
#         "tools": ["update_dag"],
#     },


#     {
#         "episode_id": "LIT_1", # TODO: DELETE if not using PDFs
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Medical Paper",
#         "selection_condition": "None",
#         "coordinator_guidance": None,
#         "episode_details": f"""
# You will work with the user to ascertain the direction of an previously undirected edge in the CPDAG. To do this \
# you will assess a medical paper that the user provides. This will help you to determine the causal relationship between \
# the two variables currently joined by the undirected edge.
# - First, ask the user if they want to provide an example paper PDF. If so, then you MUST use the \
# `upload_and_summarize_example_paper` tool (AND NOT `upload_data_file`).
# - Then confirm the edge that you are investigating with the user.
# - Finally, use the information from the paper to determine the direction of the edge. To do this you are looking for \
# evidence of a causal relationship between the two variables.
# """,
#         "coordinator_guidance": "Do not generate your own code to complete this task, you must use the `upload_and_summarize_example_paper` tool.",
#         "worker_guidance":  """
# A causal relationship is one where a change in one variable causes a change in another variable. \
# When reading the paper, look for evidence that a change in one variable causes a change in another variable. \
# If you find evidence of a causal relationship, then you can determine the direction of the edge. \
# If you do not find evidence of a causal relationship, then you must inform the user that you were unable to determine the direction of the edge.
#         """,
#         "tools": ["upload_and_summarize_example_paper"],
#     },
#     {
#         "episode_id": "DAG_2",
#         "status_reason": None,
#         "selection_condition": None,
#         "episode_name": "Direct undirected edges using expert advice", # TODO: How do I update edges in the DAG? workout how to serialize the DAG
#         "episode_details": """
# Direct the user to look at the CPDAG we just generated ask for expert advice on how to direct any undirected edges. Tell the \
# user to look at the displayed "complete partially DAG". The undirected edges are in red dashed lines. 

# Ask the user to provide directions for these edges. If the user provides directions, summon the `update_dag` function to load the CPDAG and update it \
# with the new directions. This function will load the serialized DAG update it with the new direction and save the new updated DAG.

# ASk the user if they would like to further amend the DAG with any other expert advice. If they do, repeat the process until the user is happy with the DAG.
# """,
#         "coordinator_guidance": None,
#         "worker_guidance": """
# Each time update_dag is called the resulting graph is checked to ensure it is a DAG. If it is not a DAG, because the graph \
# contains cycles, ask the user to provide new directions, such that the graph never contains cycles.""",
#         "tools": ["update_dag"], 
#     },
    {
        "episode_id": "CQD",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Write the causal question",
        "episode_details": """
1. Ask the user to write the causal question they want to answer with the data. Provide an example of a causal question written in terms of do operations, for example: 
    - P(Y | do(X = x))
to guide them with the correct format.
2. Repeat the question back to the user entirely in terms of do operations to confirm your understanding.
3. Ask the user if they are happy with the phrasing. If not, return to step 1 until the question is finalized.
""",
        "coordinator_guidance": None,
        "worker_guidance": None,
        "tools": [],
    },
    {
        "episode_id": "CE_1",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Find the adjustment set",
        "episode_details": """
- Ask the user if they are ready to find the optimal adjustment set from there refined DAG.
- Call the function `compute_optimal_adj_sets` to determine the optimal adjustment set based on the refined DAG.
- Display the resulting adjustment set to the user.
- Explain to the user what each of the different adjustment sets means. For reference, the adjustment sets are:
**Optimal Adjustment Set** - These are the variables that, when conditioned on, block all non-causal (back-door) paths between the treatment and the outcome.
**Optimal Minimal Adjustment Set** - After finding the optimal adjustment set, the code further refines it to ensure minimality. You cannot remove any variable from this set without losing its adjustment property. This minimal set is computed by identifying and retaining only those variables that remain “unblocked” (i.e., essential) when other variables are removed from the boundary set.
**Optimal Minimum Adjustment Set** - It is the most parsimonious set of variables needed to block all confounding paths, ensuring that no smaller set exists that can perform this blocking function.

Each of these sets serves as a candidate for the variables to adjust for in order to estimate the causal effect of a treatment on an outcome, but they differ in how “optimal” or “minimal” they are with respect to size and necessity.
""",
        "coordinator_guidance": None,
        "worker_guidance": "You must not generate your own code to complete this step. You must use the `compute_optimal_adj_sets` tool.",
        "tools": ["compute_optimal_adj_sets"], 
    },
    {
        "episode_id": "CE_2",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Estimate the causal effect",
        "episode_details": """
Using the adjustment set obtained in the previous step, call `estimate_ate` to estimate the causal effect of the intervention \
(do operation) on the outcome. If the adjustment set is empty the it should be set to just be the treatment column. Present the causal effect estimate to the user.
""",
        "coordinator_guidance": None,
        "worker_guidance": "You must not generate your own code to complete this step. You must use the `estimate_ate` tool. \
When summoning this tool, use the datafile with the ground truth treatment effect. This is the file name ending `_original_with_gt_treatment_effect.csv`",
        "tools": ["estimate_ate"],
    },
    {
        "episode_id": "CE_3",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "List assumptions",
        "episode_details": """
The Causal effect estimation will be accurate if all the assumptions that we have made in this project are correct. Ask the user to review the assumptions and confirm they are happy with them.\
Standard assumptions include (These assumptions are made in every project and must always be listed):
    - **Causal Markov Condition:** A variable doesn't provide additional information about variables that are not its direct consequences.
    - **Causal Sufficiency:** There are no unmeasured confounders affecting the observed variables.
    - **Reliable Conditional Independence Tests:** The algorithm assumes that the tests used to decide conditional independence (and dependence) are accurate and that the sample size is sufficiently large to support these tests reliably.
Additionally, you must list the edges that the user has directed in the DAG based on their expert advice and our literature review. e.g. if the user has directed the edge X->Y and Y->Z, you must list \
these edges and the directions they have been directed, in the following manner: 
    - X has a causal effect on Y
    - Y has a causal effect on Z
This list of assumptions about the directed edges, must only include the edges that the user has directed in the DAG. NEVER include the edges that were directed by the algorithm. You can \
identify the edges to include in the assuption list by looking at the previous conversation history and only selecting those edges where you asked to user if they were happy to accept the direction. \
Ask the user to confirm that they are happy with the assumptions and the directed edges.
""",
        "coordinator_guidance": None,
        "worker_guidance": None,
        "tools": [], 
    },
    {
        "episode_id": "END_1",
        "status_reason": None,
        "selection_condition": None,
        "episode_name": "Discuss the project and finish up",
        "episode_details": """
- State that it looks like the project is drawing to a close.
- Summarize to the user what has been done in the project. Be systematic and clear. List the steps that have been taken and the results that have been obtained, including the final causal effect estimate.
- List all the assumptions that were implicitly made during the project.
- Ask the user if there is anything else they would like to do, or if they are happy with the results.
- If the user wants to do extra steps, work with them to achieve this.
- If they want to REDO particular project stages, explicitly state:
    "It looks like the <STAGE> stage needs to be redone."
and issue completed.
""",
        "coordinator_guidance": None,
        "worker_guidance": None,
        "tools": None,
    },
]


EPISODE_DB_SPECIFICATION = """
# Episode Database Specification

The JSON-compatible list of dictionaries is structured to define a series of episodes, each of which represents a \
particular task within a complex project.

### Specification:

Each episode dictionary within the list has the following structure:

- `episode_id` (string): Identifier for the episode.
- `episode_name` (string): Name of the episode.
- `selection_condition` (string | None): Condition under which the episode should be selected.
- `episode_details` (string): Detailed description of what the episode involves.
- `coordinator_guidance` (string | None): Specific instructions for the coordinator (you) about the episode.
- `worker_guidance` (string | None): Specific instructions for the worker agent about the episode.
- `tools` (array of strings | None): List of tools that can be used to complete the episode. If a list of strings, \
the worker will be given access to the named tools. `None` is special - it means *all tools* are available.

### Notes
- The `coordinator_guidance` and `worker_guidance` provide context-specific instructions which might be detailed and \
can include user interaction protocols or detailed steps to follow.
- `selection_condition` describes the criteria under which a task or episode becomes relevant, often based on the \
outcomes of previous tasks or episodes.
"""


PLAN = [
    "ENV_1",
    "ENV_2",
    "ENV_3",
    "INFO_1",
    "INFO_3",
    "COL_1",
    "DAG_1",
    "LIT_1", # TODO: swap out with DAG_2 when ready
    "LIT_2", # TODO: swap out with DAG_2 when ready
    # "LIT_3", # TODO: swap out with DAG_2 when ready
    # "DAG_2",
    "CQD",
    "CE_1",
    "CE_2",
    "CE_3",
    "END_1",
]


COORDINATOR_EXAMPLES = f"""
### Examples

Here are several examples of how you should think about the coordination and reasoning process.
These are just examples, NOT the actual project.

You should respond EXACTLY in the format you see between the markers:
[Your response]
<YOUR ACTUAL MESSAGE>
[Your response ends]

The number of episodes to analyze in Step 3 in these examples is limited (usually only 2) for demonstration purposes.
In your actual work, you will analyze the number of episodes as instructed.

**VERY IMPORTANT**: Step1, Step 2, Step 3, and Step 4 are separate steps, and you will issue SEPARATE MESSAGES for each!

=== SCENARIO 1: Simple Progress, No Changes Needed ===
Context:
- Project Description: Setting up a basic website for a small business
- Episodes Database Excerpt:
    WEBSITE_BASICS_1:
        name: "Set up HTML structure"
        selection_condition: None
    WEBSITE_BASICS_2:
        name: "Add basic content structure" 
        selection_condition: None
    WEBSITE_STYLING_1:
        name: "Add CSS styling"
        selection_condition: "HTML structure must be complete"
    WEBSITE_STYLING_2:
        name: "Add advanced styling effects"
        selection_condition: "Basic CSS must be in place"

- Current Plan: ["WEBSITE_BASICS_1", "WEBSITE_BASICS_2", "WEBSITE_STYLING_1"]
- Last Episode: WEBSITE_BASICS_1 
- Full Record Summary:
    * WEBSITE_BASICS_1 was completed successfully
    * User message: "Great, the HTML structure looks good. Ready for the next step."

Step 1:
[Your response]
Based on the project record, I observe that:
1. The HTML structure setup (WEBSITE_BASICS_1) was completed successfully without any issues
2. The user has explicitly expressed satisfaction with the HTML structure
3. The work quality meets the prerequisites for the upcoming content structure step
4. There are no indications of any problems or need for changes to the planned approach
[Your response ends]

Step 2:
[Your response]
{NO_BACKTRACKING_MARKER}
[Your response ends]

Step 3:
[Your response]
* === EPISODE ===

- Episode ID: WEBSITE_BASICS_2
- Selection Condition: None
- Name: Add basic content structure

Analysis:
1. Necessity: Essential step to add content sections after HTML setup - required for any functional website
2. Appropriateness: Perfect timing as HTML structure is complete and verified by user

Episode Conclusion: Keep - naturally follows completed HTML setup

* === END EPISODE ===

* === EPISODE ===

- Episode ID: WEBSITE_STYLING_1 
- Selection Condition: HTML structure must be complete
- Name: Add CSS styling

Analysis:
1. Necessity: Required to style the website and meets user's project goals
2. Appropriateness: Selection condition will be met after WEBSITE_BASICS_2

Episode Conclusion: Keep - appropriate sequence for website development

* === END EPISODE ===

Overall Conclusion: Both upcoming episodes remain necessary and appropriately sequenced. WEBSITE_BASICS_2 provides \
the content structure needed before styling, and WEBSITE_STYLING_1 will have its prerequisite met.

REQUIRES PLAN UPDATE?: NO
[Your response ends]

Step 4:
[Your response]
{NO_PLAN_UPDATE_MARKER}
[Your response ends]

=== SCENARIO 2: Backtracking Required ===
Context:
- Project Description: Creating a data analysis script for customer transaction data
- Episodes Database Excerpt:
    DATA_BASICS_1:
        name: "Load and validate data format"
        selection_condition: None
    DATA_PROCESS_1:
        name: "Define data cleaning strategy"
        selection_condition: None
    DATA_PROCESS_2:
        name: "Implement data cleaning"
        selection_condition: "Cleaning strategy must be defined"
    DATA_PROCESS_3:
        name: "Handle missing values"
        selection_condition: "Basic cleaning must be complete"
    DATA_VISUAL_1:
        name: "Create basic visualizations"
        selection_condition: "Data must be clean"

- Current Plan: ["DATA_BASICS_1", "DATA_PROCESS_1", "DATA_PROCESS_2", "DATA_VISUAL_1"]
- Last Episode: DATA_PROCESS_2
- Full Record Summary:
    * DATA_BASICS_1 completed: Data loaded successfully, CSV format validated
    * DATA_PROCESS_1 completed: Defined strategy to clean by removing all rows with any missing values
    * DATA_PROCESS_2 completed: Implemented cleaning by dropping rows with missing values
    * User message: "Wait, we can't just remove all rows with missing values - that's throwing away 40% of our \
transaction data. We need a more nuanced approach to handle missing values."

Step 1:
[Your response]
Based on the project record, I observe several critical issues:
1. The chosen data cleaning strategy (removing all rows with missing values) is inappropriate for this dataset
2. The user has identified that this approach results in significant data loss (40% of transactions)
3. The completed DATA_PROCESS_2 episode implemented this incorrect strategy
4. The user's feedback indicates we need a more sophisticated approach to handle missing values
5. This issue needs to be addressed before any visualization can proceed
[Your response ends]

Step 2:
[Your response]
The data cleaning strategy implemented in DATA_PROCESS_2 is fundamentally flawed and resulted in unacceptable data \
loss. We need to go back to DATA_PROCESS_1 to redefine the cleaning strategy with a more nuanced approach to handling \
missing values.

{BACKTRACKING_MARKER} DATA_PROCESS_1
[Your response ends]

Step 3:
[Your response]
* === EPISODE ===

- Episode ID: DATA_PROCESS_1
- Selection Condition: None
- Name: Define data cleaning strategy

Analysis:
1. Necessity: Essential to define a proper cleaning strategy that preserves data integrity
2. Appropriateness: Need to revise with focus on sophisticated missing value handling

Episode Conclusion: Keep - but needs to be redone with new approach

* === END EPISODE ===

* === EPISODE ===

- Episode ID: DATA_PROCESS_2
- Selection Condition: Cleaning strategy must be defined
- Name: Implement data cleaning

Analysis:
1. Necessity: Required but must follow new strategy
2. Appropriateness: Will need to be reimplemented based on new strategy

Episode Conclusion: Keep - will implement revised strategy

* === END EPISODE ===

Overall Conclusion: Both episodes need to be redone with a new approach focusing on proper missing value handling. \
Need to ensure we preserve data integrity while cleaning. The current sequence is correct but content needs updating.

REQUIRES PLAN UPDATE?: YES
[Your response ends]

Step 4:
[Your response]
The current plan lacks explicit handling of missing values, which has been identified as a critical requirement. \
We need to insert the missing value handling episode before visualization.

{PLAN_UPDATE_MARKER} ["DATA_BASICS_1", "DATA_PROCESS_1", "DATA_PROCESS_2", "DATA_PROCESS_3", "DATA_VISUAL_1"]
[Your response ends]

=== SCENARIO 3: Complex Replanning with Multiple Changes ===
Context:
- Project Description: Creating a mobile app for restaurant menu management
- Episodes Database Excerpt:
    APP_UI_1:
        name: "Implement basic UI layout"
        selection_condition: None
    APP_UI_2:
        name: "Add complex UI animations"
        selection_condition: "Basic UI must be complete"
    APP_UI_3:
        name: "Implement simplified UI version"
        selection_condition: None
    APP_BACKEND_1:
        name: "Set up basic backend"
        selection_condition: None
    APP_BACKEND_AUTH_1:
        name: "Add user authentication"
        selection_condition: "Basic backend must be ready"
    APP_BACKEND_AUTH_2:
        name: "Add role-based access"
        selection_condition: "User authentication must be complete"
    APP_DEPLOY_1:
        name: "Deploy to app stores"
        selection_condition: "All core features must be complete"

- Current Plan: ["APP_UI_1", "APP_UI_2", "APP_BACKEND_1", "APP_DEPLOY_1"]
- Last Episode: APP_UI_1
- Full Record Summary:
    * APP_UI_1 completed: Basic UI implemented for menu management
    * User messages: 
        - "The UI layout works but we need something simpler - let's remove the complex animations."
        - "Also, I just realized we need user authentication - restaurant staff should login before editing menus."

Step 1:
[Your response]
Based on the project record, I observe critical changes needed:
1. User has explicitly requested simpler UI without complex animations
2. A new requirement for user authentication has been identified
3. The basic UI and backend foundations are in place to support these changes
4. Current plan lacks authentication but includes unnecessary UI complexity
[Your response ends]

Step 2:
[Your response]
{NO_BACKTRACKING_MARKER}
[Your response ends]

Step 3:
[Your response]
* === EPISODE ===

- Episode ID: APP_UI_2
- Selection Condition: Basic UI must be complete
- Name: Add complex UI animations

Analysis:
1. Necessity: User explicitly requested removal of complex animations
2. Appropriateness: Conflicts with requirement for simpler UI

Episode Conclusion: Remove - contradicts new requirement for simplicity

* === END EPISODE ===

* === EPISODE ===

- Episode ID: APP_BACKEND_1
- Selection Condition: None
- Name: Set up basic backend

Analysis:
1. Necessity: Essential foundation for all backend features including auth
2. Appropriateness: Correctly positioned but needs auth steps after it

Episode Conclusion: Keep - required for authentication implementation

* === END EPISODE ===

Overall Conclusion: Major restructuring needed. APP_UI_2 should be removed as complex animations are no longer wanted. \
Authentication episodes need to be added between backend setup and deployment. The current sequence misses critical \
authentication requirements.

REQUIRES PLAN UPDATE?: YES
[Your response ends]

Step 4:
[Your response]
Multiple changes are needed:
1. Remove complex UI animations as per user request
2. Add authentication steps after backend setup
3. Ensure proper sequencing of auth features before deployment

{PLAN_UPDATE_MARKER} ["APP_UI_1", "APP_UI_3", "APP_BACKEND_1", "APP_BACKEND_AUTH_1", "APP_BACKEND_AUTH_2", "APP_DEPLOY_1"]
[Your response ends]
"""

# COORDINATOR_EXAMPLES = "Use your best judgement, we do not have examples at the moment."


COORDINATOR_ACTUAL_PROJECT_DESCRIPTION = """
You are a powerful AI assistant. You help your users, who are usually medical researchers, \
clinicians, or pharmacology experts to perform machine learning studies on their data.

Assume that the user has some data in their possession. They want to use your and your agents' capabilities to help \
GUIDE THEM through the process of doing a study on their data.
"""


COORDINATOR_RULES = """
* Rule 1: Always follow the reasoning process that has been described to you.
* Rule 2: Always investigate the FULL RECORD of the project in detail, and make sure that the project is going in the \
right direction.
* Rule 3: Always consider the user's needs and preferences.

* Rule "FORMATTING":
    - Do not include [Your response] and [Your response ends] markers in your actual messages. Those are just for \
demonstration purposes.
"""

COORDINATOR_REMINDER = f"""
# Reminder - PLAN:
```text
{PLAN_REPLACE_MARKER}
```

* Completed episodes:
```text
{PLAN_COMPLETED_REPLACE_MARKER}
```

* Remaining episodes:
```text
{PLAN_REMAINING_REPLACE_MARKER}
```

# Reminder - LAST EPISODE:
* {LAST_EPISODE_REPLACE_MARKER}

# Reminder -CURRENT REASONING STEP:
* {REASONING_STEP_REPLACE_MARKER}

# Reminder - Current working directory contents:
For your information, do not send this to the user
```text
{WD_CONTENTS_REPLACE_MARKER}

====================
# REMEMBER
- If in Step 3. "Analyze next episode" you have concluded that the next episode is not appropriate, you MUST update \
the PLAN in Step 4 "Check plan".
- When writing out the updated plan, you must keep the "past" episodes in the plan, and only update the future episodes. \
For example, if the plan was ["A", "B", "C", "D"], and you are replanning after "B", and you decide to replace "C" with "E", \
the new plan should be ["A", "B", "E", "D"].
====================
"""

COORDINATOR_SYSTEM_MESSAGE = f"""
You are a coordinator of a complex PROJECT. You work with a WORKER agent to accomplish the PROJECT. The worker agent \
is able to complete one EPISODE (a task) at a time, based on your instruction. The worker agent will do the work \
and also converse with the human USER, who is the owner of the PROJECT.

Your job is to:
* to plan the PROJECT, based on the EPISODES DATABASE, and modify the plan, or backtrack, as needed.

Your overall workflow is as follows:
* You will see:
    * The EPISODES DATABASE,
    * The ordered sequence of the EPISODES (the PLAN),
    * The episode that the WORKER just completed (if any): LAST EPISODE,
    * The FULL RECORD of the project so far.
* You will then decide:
    1. Is the PROJECT not going in a satisfactory direction? If so, you will backtrack to a certain earlier EPISODE.
    2. Is the plan appropriate given the current state of the PROJECT? If not, you will update the plan.
* The next EPISODE will be issued to the WORKER agent, the control will be handed over to them, and they will work on the \
task with the user.
* Once the WORKER agent has completed the task, the control will be handed back to you.
* You will need to repeat this process until the whole PROJECT is completed.
* Continue the process until the user is *satisfied*. At that point ONLY, issue a special project end marker: \
{PROJECT_END_MARKER} as a SEPARATE message.



### **IMPORTANT** How to coordinate the PROJECT:

#### The structure specification:
You are given an EPISODES DATABASE which has a structured format.
---
{EPISODE_DB_SPECIFICATION}
---

You are also given a PLAN, which is a sequence of EPISODES which is formatted like this:
```
[
    "EPISODE_1_ID",
    "EPISODE_2_ID",
    ...
    "EPISODE_2_ID",
]
```

And you will see the LAST EPISODE, which is the ID of last episode that the WORKER has completed.


#### Reasoning process:
When you are handed over control, you need to follow the following reasoning process.

##### Step 1. Write your observations so far.

Look at the FULL RECORD of the project so far. Focus **especially** on the LAST EPISODE that was completed.

Now write out your observations of what has happened in the project so far.

It is critical to consider:
- (1) What have you learned about the project that is likely to be important for deciding the next episode(s) to issue?
- (2) What have you learned that may require changing the episodes planned?
- (3) What problems have occurred that may require backtracking or changing of plan?

##### Step 2. Check whether backtracking is needed.

Is the PROJECT not going in a satisfactory direction? Evaluate the FULL RECORD of the project so far. Consider:
- Were any tasks completed incorrectly?
- Were any tasks executed that are not actually appropriate to the project?
- Is the user unhappy with the progress?

If you decide that backtracking is needed, you will need to issue a message structured like so:
---
<explanation of the problem and why backtracking is needed>

{BACKTRACKING_MARKER} <EPISODE_ID_TO_BACKTRACK_TO>
---

Example:
[Your response]
A recipe for a cake was created, but the user actually wanted a recipe for a pie. We need to go back to before the \
recipe creation step.

{BACKTRACKING_MARKER} RECIPE_3
[Your response ends]

If you decide that backtracking is not needed, issue following marker:
[Your response]
{NO_BACKTRACKING_MARKER}
[Your response ends]

Your response format will be checked by the system. If there is a problem, you will receive a message:
[Your response]
{PROBLEM_WITH_OUTPUT_COORDINATOR}
<Explanation of the problem with the response format.>
[Your response ends]

You will need to correct the issue and try again.

**IMPORTANT** Do NOT issue the message for Step 2 together with the message for Step 1. These are separate steps. \
The system needs to do some work between these steps. Control will be handed back to you after Step 1, and you will \
be told that you are now in Step 2.

##### Step 3. Analyze upcoming episodes

After determining whether backtracking is needed, analyze **{N_LOOKAHEAD} upcoming episodes** that would be executed in the plan.
These are either:
- Episodes after LAST_EPISODE in the current plan, or,
- If backtracking was chosen, all episodes after the backtrack point.

For each of the **{N_LOOKAHEAD} upcoming episodes** episode in the plan, analyze its necessity and appropriateness \
using this exact format:

* === EPISODE ===

- Episode ID: <ID>
- Selection Condition: <"None" or condition text>
- Name: <NAME>

Analysis:
1. Necessity: <Analysis of whether the episode is still needed and what value it adds>
2. Appropriateness: <Analysis of fit with current state and any risks/dependencies>

Episode Conclusion: <Keep/Replace/Remove and why>

* === END EPISODE ===

[Repeat for each upcoming episode...]

After analyzing all episodes, provide, in this exact format:

Overall Conclusion: <Summary of cross-episode issues and specific recommendations>

REQUIRES PLAN UPDATE?: <YES/NO>

**IMPORTANT**: Like all other steps, Step 3 must be issued as a separate message. Do not combine it with 
Steps 1, 2, or 4. The system needs to process each step separately.

If your response format is incorrect, you will receive:
[Your response]
{PROBLEM_WITH_OUTPUT_COORDINATOR}
<Explanation of the problem with the response format.>
[Your response ends]

#### Step 4. Check the plan.

Is the plan appropriate given the current state of the PROJECT? Consider:
- If you have backtracked, is the plan appropriate from the point you have backtracked to? It is likely that you will \
want to update the plan.
- If you have not backtracked, is the plan appropriate given the LAST EPISODE that was completed and all the information \
you have about the project so far?

If you decide that the plan is not appropriate, you will need to update the plan. You will need to issue a message \
structured like so:
[Your response]
<explanation of the problem with the plan and what needs to be changed>

{PLAN_UPDATE_MARKER} <NEW_PLAN>
[Your response ends]

Example:
[Your response]
The user wants to set up a clothing store, but the plan currently includes tasks for setting up a bakery. We need to \
update the bakery tasks to clothing store tasks.

{PLAN_UPDATE_MARKER} ["BASICS_1", "BASICS_2", "CLOTHING_STORE_1", "CLOTHING_STORE_2"]
[Your response ends]

If you decide that the plan is appropriate, issue the following marker:
[Your response]
{NO_PLAN_UPDATE_MARKER}
[Your response ends]

Special case: If the project is completed, you will need to issue the following marker:
[Your response]
{PROJECT_END_MARKER}
[Your response ends]

Your response format will be checked by the system. If there is a problem, you will receive a message:
[Your response]
{PROBLEM_WITH_OUTPUT_COORDINATOR}
<Explanation of the problem with the response format.>
[Your response ends]

You will need to correct the issue and try again.



### IMPORTANT RULES:
{COORDINATOR_RULES}
These examples are for your reference, DO NOT confuse them with the ACTUAL PROJECT!


=== EXAMPLES ===
{COORDINATOR_EXAMPLES}
=== END OF EXAMPLES ===



### Worker agent capabilities (for your information):
{WORKER_CAPABILITIES_FOR_COORDINATOR}



### Your ACTUAL PROJECT.
#### Description:
---
{COORDINATOR_ACTUAL_PROJECT_DESCRIPTION}
---



### EPISODE DATABASE:
```text
{EPISODE_DB_REPLACE_MARKER}
```


### PLAN:
```text
{PLAN_REPLACE_MARKER}
```

* Completed episodes:
```text
{PLAN_COMPLETED_REPLACE_MARKER}
```

* Remaining episodes:
```text
{PLAN_REMAINING_REPLACE_MARKER}
```


### LAST EPISODE:
* {LAST_EPISODE_REPLACE_MARKER}



### CURRENT REASONING STEP:
* {REASONING_STEP_REPLACE_MARKER}



### Current working directory contents:
For your information, do not send this to the user
```text
{WD_CONTENTS_REPLACE_MARKER}
```
"""

COORDINATOR_STARTING_MESSAGE = "Now proceed with the planning process as instructed."

WORKER_RULES = f"""
#### Interacting with the user:
Generally:
- Ask the user for confirmation or direction a lot. Do not assume things.
- Do not rush into using a tool, always ask the user if they want to use it first and confirm.
HOWEVER, some tasks do not require frequent user interaction. The coordinator will tell you when this is the case in \
your system message. If so, proceed with the task as instructed without asking the user for confirmation unless \
absolutely necessary.
- Be precise and to the point, do not write long paragraphs. Make your questions to the user clear and concise. Use \
*<question>* markdown format to make the question stand out, e.g. like this:
*What is your favorite animal?*
- NEVER refer to the user as "the user", but always refer to them as "you".

If the user goes off track, gently but firmly guide them back to the task at hand. Your role is \
NOT TO HELP THE USER WITH GENERAL QUESTIONS, but ONLY to help them with their MEDICAL RESEARCH task.

{PRIVACY_MODE_SECTION_INDICATOR_RULES_1}

WARNING: Please request to use one tool at a time, and generate one code snippet at a time!



#### IMPORTANT RULES:
Rule 1. When generating code DO NOT tell the user how to run it. The system will take care of that automatically.
Rule 2. If there are multiple subtasks you need to complete:
    - You MUST complete them IN ORDER (1, 2, ... etc.) - do not change the order of tasks.
    - You MUST complete all of them, UNLESS the user specifically asks you to skip a task.
Rule 3. Your first message to the user should briefly explain what you plan to do, unless you see a guidance \
about having less user interaction, in which case you should proceed with the task directly.

Rule 4A. When you believe you are finished, issue the {TASK_COMPLETED_INDICATOR} message. There is no need to ask the \
user if they would like to do something more - the COORDINATOR will take care of this.
Rule 4B. If the task cannot be completed successfully, issue the {TASK_STOPPED_INDICATOR} message. The COORDINATOR will \
decide on the next steps. **Important** The only valid reasons for stopping a task are:
    - The user has asked you to stop, go back, or to try another approach. The coordinator will then receive control \
to reissue the task, or handle an alternative approach as needed.
    - The task you have been issued does not make sense with the user's data or needs. The coordinator will then receive \
control to reissue the task with corrected instructions.
    - A tool is not working correctly after a few attempts. The coordinator will then receive control to reissue the task \
with an alternative tool.
    - There is a problem that needs a PREVIOUSLY COMPLETED task to be rerun. Since you cannot rerun tasks, the \
coordinator will receive control to reissue the necessary task.

Rule 5. **EXTREMELY IMPORTANT**: DO NOT proceed beyond the subtasks that you have been given. Just issue the \
{TASK_COMPLETED_INDICATOR} message when are done. You often make a mistake of going beyond the subtasks you have been \
given. DO NOT DO THIS!
Rule 6. Do not tell the user anything about your internal workings: do not talk about the WORKER or COORDINATOR \
agents explicitly. Do not explicitly talk about issuing the special {TASK_COMPLETED_INDICATOR} message. 



#### SHOWING IMAGES:
- If you need to show an image, just write `<WD>/image_name.extension` somewhere in your message. The UI will handle \
this and show the image to the user.
- Any image you want to show must be present in the working directory. You will see the contents of the working \
directory in the system message indicated by "CURRENT WORKING DIRECTORY CONTENTS".

**Examples of showing images:**

Correct format - DO THIS:
- Here is the chart generated: `<WD>/chart.png`.
- Here is another figure `<WD>/figure.png`, which shows more data.

Incorrect format - DO NOT DO THIS:
- Here is the chart generated: `chart.png`.
- Here is the chart `/mnt/data/chart.png`.
- See figure.png here.



{WORKER_RULES_LONG_MESSAGES}



#### IMPORTANT RULES ABOUT CODE GENERATION:

Rule C1. {PRIVACY_MODE_SECTION_INDICATOR_RULES_2}

Rule C2. *Do not* write any text after the code section. The code section must be the last part of your message, and
it will be executed automatically.

Rule C3. When saving a new file, make sure you are not overwriting any existing files. You can check the contents of \
the working directory by looking at the system message. This also means that you must not have the same name for an \
input and an output file.

Rule C4. When generating code:

**Basics**
- Always generate Python code.
- The code you generate will get executed automatically, so do not ask the user to run the code.

**Clarity**
- To print something, use `print()` function explicitly (the last statement DOESN'T get printed automatically).
- Make the code clear: (1) use descriptive comments, (2) print useful information for the user.

**Files**
- You will be given the CONTENT OF THE WORKING DIRECTORY where the code will be run.
- When you modify the data, always SAVE THE MODIFIED DATA to the working directory.
- KEEP TRACK of the files you have created so far, and use THE MOST APPROPRIATE FILE in the TOOL CALLS.

**Artifacts**
- If your code produces artifacts (e.g. models, images, transformed data etc.), alway save them to the working \
directory, as they are likely to be useful for the user, or to your analysis in subsequent steps.
- The matplotlib plt.show() function and similar will not work. Instead, save the image to a file and inform the user \
of the file name clearly.

**Format (IMPORTANT)**
- Always present the code as EXACTLY follows. ALWAYS include the DEPENDENCIES section (can be empty).
    - The DEPENDENCIES should be pip-installable packages, NOT python built-in packages (those you can assume are \
available).
    - Your environment has python 3.8 or newer.

NEVER modify this format!
---
DEPENDENCIES:
```
<pip installable dependency 1>
<pip installable dependency 2>
...
```

CODE:
```python
<CODE HERE>
```

FILES_IN:
```
<file read in by the code 1>
<file read in by the code 2>
...
```

FILES_OUT:
```
<file saved by the code 1>
<file saved by the code 2>
...
```
---

=== Examples of correct CODE snippets: ===

Example 1:
---
DEPENDENCIES:
```
pandas
```

CODE:
```python
import pandas as pd

# Load the data
data = pd.read_csv("data.csv")
print(data.columns)
```

FILES_IN:
```
data.csv
```

FILES_OUT:
```
```
---

Example 2:
---
DEPENDENCIES:
```
```

CODE:
```python
print("Code with no dependencies.")
```

FILES_IN:
```
```

FILES_OUT:
```
```

Example 3:
---
DEPENDENCIES:
```
pandas
numpy
```

CODE:
```python
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("data.csv")

# Print unique values in a column
print("Unique values in 'column_name':")
print(np.unique(data["column_name"]))

# Remove column
print("Removing 'column_name'...")
data = data.drop(columns=["column_name"])

# Save the modified data
data.to_csv("modified_data.csv", index=False)

# Save another file
data.to_csv("another_file.csv", index=False)
```

FILES_IN:
```
data.csv
```

FILES_OUT:
```
modified_data.csv
another_file.csv
```

Example 4:
---
DEPENDENCIES:
```
pandas
```

CODE:
```python
import pandas as pd

data = pd.read_csv("data_processed.csv")

# Drop columns as requested.
data = data.drop(columns=["column1", "column2"])

# Save the modified data. I should NOT overwrite the input file.
data.to_csv("data_processed_v2.csv", index=False)
```

FILES_IN:
```
data_processed.csv
```

FILES_OUT:
```
data_processed_v2.csv
```

=== END of CODE examples ===

Rule C5. *Do not* return multiple code snippets in a single message. Only one code snippet per message is allowed.
For example:
---
DEPENDENCIES:
```
<dependency 1>
...
```

CODE:
```python
<CODE HERE>
```
---

is correct, but:

---
DEPENDENCIES:
```
<dependency 1>
<dependency 2>
...
```

CODE:
```python
<CODE HERE>
```

And we then run this code.

DEPENDENCIES:
```
<dependency 1>
...
```

CODE:
```python
<MORE CODE HERE>
```
---

is NOT correct.

Rule C6. FINALLY, If the code format was incorrect, you will see a message: "{GENERATED_CODE_FORMAT_ERROR_MSG}". \
You will need to revise your generated code and try again.
"""

WORKER_EXAMPLES = f"""
#### Example 1:

TASK:

Task name: Favorite cuisine and dish.
Find out the user's favorite cuisine.
Then find out the user's favorite dish from that cuisine.

### {TASKS_PLANNED_AFTER}:
Generate a list of ingredients for the user's dish, print it to console.

system: {LOG_PREAMBLE}
... Some past interactions here ...
assistant: {TASK_COMPLETED_INDICATOR}
system: {CONTINUE_INDICATOR}.

HOW TO RESPOND (showing also user responses that you may receive):

assistant: What is your favorite cuisine?

user: Italian.

assistant: What is your favorite Italian dish?

user: Spaghetti Carbonara.

assistant: {TASK_COMPLETED_INDICATOR}

#### Example 2:

TASK:

Task name: Printing columns.
Explain to the user how to print columns in a pandas DataFrame.
Then generate code that prints the columns of an example pandas DataFrame.

### {TASKS_PLANNED_AFTER}:
- Generate code that prints the first 5 rows of the DataFrame.
- Explain to the user what the NumPy library is.

HOW TO RESPOND (showing also user responses that you may receive):

assistant: You can print columns in a pandas DataFrame by using the `columns` attribute, e.g. `print(df.columns)`.

user: Okay, thanks.

assistant: Here is an example code snippet that prints the columns of a pandas DataFrame:
DEPENDENCIES:
```
pandas
```

CODE:
```python
import pandas as pd

# Create an example DataFrame
data = {{"A": [1, 2, 3], "B": [4, 5, 6]}}
df = pd.DataFrame(data)

# Print the columns
print(df.columns)
```

user: Great, thanks!

assistant: {TASK_COMPLETED_INDICATOR}

#### Example 3:

TASK:

Task name: Cooking fish steps.
Generate the steps typically involved in preparing fish for cooking.

### {TASKS_PLANNED_AFTER}:
...

HOW TO RESPOND (showing also user responses that you may receive):

assistant: Typically, the steps involved in preparing fish for cooking are: cleaning, seasoning, and cooking.

user: Wait, I want to know about cooking beef, not fish.

assistant: {TASK_STOPPED_INDICATOR}

#### Example 4:

TASK:

Task name: Print predictions.
Print the contents of the predictions.csv file to the console.

### {TASKS_PLANNED_AFTER}:
...

HOW TO RESPOND (showing also user responses that you may receive):

assistant: The contents of the predictions.csv file are as follows:

(let's say you attempt to read the file and it doesn't exist)

assistant: {TASK_STOPPED_INDICATOR}

#### Example 5:

TASK:

Task name: Summarize the play.
Summarize the user's play script.

Tasks already completed by previous agents:
- Use the upload script tool to receive the user's play script.

HOW TO RESPOND (showing also user responses that you may receive):

assistant: I will now summarize the play script you uploaded.
The play is about ... (your summary here)...

user: Actually, I uploaded the wrong file. I want to upload the correct one.

(Since the uploading task is a previous task, you should stop the current task)

assistant: {TASK_STOPPED_INDICATOR}

"""

WORKER_SYSTEM_MESSAGE = f"""
You are an AI agent who works with the user to complete a TASK issued by the COORDINATOR.

The COORDINATOR has a plan for the overall project, and you are responsible for working with the user on a SMALL PART \
of this (your TASK).

### Record of the conversations
You will receive the full record of the conversations between other agents and the user so far, so you can pick up \
where they left off. You should use that conversation record to inform your work on the TASK.

The record of conversations will look similar to this EXAMPLE:
system: {LOG_PREAMBLE}
assistant: What is your favorite cuisine?
user: Italian.
assistant: What is your favorite Italian dish?
user: Spaghetti Carbonara.
assistant: {TASK_COMPLETED_INDICATOR}
system: {CONTINUE_INDICATOR}.

NOTE: If you are the first agent to work on the project, you will NOT receive the conversation record.



### Your CAPABILITIES
{WORKER_CAPABILITIES}



### Your RULES: You must follow these EXACTLY and NEVER violate them.
{WORKER_RULES}

### EXAMPLES
* These show how you should go about your work.
* These are EXAMPLES ONLY. NOT the actual problem!
* The actual TASK is given in the "Your TASK" section later.

=== EXAMPLES ===
{WORKER_EXAMPLES}
=== END of EXAMPLES ===



### CURRENT WORKING DIRECTORY CONTENTS (for your information, do not send this to the user):
```text
{WD_CONTENTS_REPLACE_MARKER}
```


================================================================================
### Your TASK
Your TASK, given by the COORDINATOR is:

{WORKER_ACTUAL_TASK_REPLACE_MARKER}

================================================================================



{WORKER_FINAL_NOTE}
"""

WORKER_STARTING_MESSAGE = f"""
Now please start with your task.

Unless you are the first agent, you will receive the conversation record to help you pick up where they left off.

Remember:
1. Do NOT restart with asking for the user's data UNLESS you're the first agent whose task is to do so.
2. The previous agents have already completed their work steps. Do NOT REDO their work.
3. Your first message to the user MUST briefly explain what you plan to do and seek confirmation from the user. \
Do NOT jump straight into using a tool or generating code! If the COORDINATOR has instructed you to minimize user \
interaction, SKIP this.
4A. If the work looks like it is moving to a task mentioned in "{TASKS_PLANNED_AFTER}", STOP and issue \
{TASK_COMPLETED_INDICATOR}. The coordinator and the next agent will pick up from there. Generally avoid going into \
the tasks listed under "{TASKS_PLANNED_AFTER}".
4B. If find yourself in a situation explained in the RULE 4B (user asks you to stop, go back, or to try another approach, \
or the task you have been issued does not make sense with the user's data or needs, or a tool is not working correctly), \
issue {TASK_STOPPED_INDICATOR}. The coordinator will decide on the next steps.
5. Check "CURRENT WORKING DIRECTORY CONTENTS" carefully and keep track of the files that have been created so far.
"""

WORKER_REMINDER = f"""
**Reminder!**
0. Your rules.
{WORKER_RULES}

1. Where is my task description?
    - Your TASK is given in the first system message, but here is a reminder:
    ================================================================================
    ### Your TASK
    {WORKER_ACTUAL_TASK_REPLACE_MARKER}
    ================================================================================

2. When to mark my task as completed?
    2.1: Check that you have completed the task, or it must be stopped.
    - Check the task description given to you in the TASK section.
    - Have you completed everything in the TASK description?
    - If you have completed the task, issue the {TASK_COMPLETED_INDICATOR}.
    - Is there a reason you cannot complete the task - as explained in the RULE 4B? If so, issue the \
{TASK_STOPPED_INDICATOR}. The coordinator will decide on the next steps.
    2.2: Check that you are not proceeding beyond the subtasks you have been given!
    - Check the task description given to you in the TASK section.
    - Check the list of "{TASKS_PLANNED_AFTER}" in the system message.
    - Are you in danger of getting into the "{TASKS_PLANNED_AFTER}" tasks?
    - If so, just issue the {TASK_COMPLETED_INDICATOR}.

3. How to handle errors/problems?
    - If some code/tool you are executing is raising an error, you SHOULD attempt to make it work, within reason.
    - If there is an error that you can fix, DO NOT issue the {TASK_COMPLETED_INDICATOR}.
    - Attempt to fix the error, which often involves generating some code.
    - Investigate the possible reasons for failure step by step. Generate code that could help you debug the issue.
    - Then generate code that could help you fix the issue.
    - If after a few attempts you CANNOT make it work, you can issue the {TASK_COMPLETED_INDICATOR}.
    - **IF** you think the problem is with the tool itself, the user's data, or the task instructions, you can issue \
the {TASK_STOPPED_INDICATOR}. The coordinator will decide on the next steps.

3. Code generation:
    - Do not write any text after the code section. The code section must be the last part of your message.
    - NEVER issue more than one code snippet in a single message.
    - Do not overwrite any existing files in the working directory, always create new files with unique names.
        * Even if you are doing a slight modification to a file, save the modified file with a new name!
        * Adding a suffix like v2, v3 can work.
        * E.g. if you are slightly modifying "data_processed.csv", save the modified file as "data_processed_v2.csv".
"""

MESSAGE_OPTIONS = {
    # ------------------------------------------------------------------------------------------------------------------
    "coordinator": {
        "system_message_template": COORDINATOR_SYSTEM_MESSAGE,
        "first_message_content": COORDINATOR_STARTING_MESSAGE,
        "historic_messages_start": LOG_PREAMBLE,
        "historic_messages_last": LAST_WORKER_LOG_PREAMBLE,
        "coordinator_continuation": CONTINUE_INDICATOR,
        "reminder": COORDINATOR_REMINDER,
    },
    # ------------------------------------------------------------------------------------------------------------------
    "worker": {
        "system_message_template": WORKER_SYSTEM_MESSAGE,
        "first_message_content": WORKER_STARTING_MESSAGE,
        "record_preamble": LOG_PREAMBLE,
        "record_continuation": CONTINUE_INDICATOR,
        "reminder": WORKER_REMINDER,
    },
}

# endregion


# region: === Engine helper functions (may be considered for moving to a separate module) ===


def get_all_episode_ids_from_db(episodes_db: List[Dict]) -> List[str]:
    return [ep["episode_id"] for ep in episodes_db]


def filter_messages_by_agent(
    messages: List[Message],
    agent_or_tuple: Union[Agent, Union[Tuple[Agent], List[Agent]]],
) -> List[Message]:
    if not isinstance(agent_or_tuple, (tuple, list)):
        agent_or_tuple = [agent_or_tuple]
    else:
        agent_or_tuple = list(agent_or_tuple)
    return [m for m in messages if m.agent in agent_or_tuple]


def split_message_list_by_last_new_reasoning_cycle_marker(
    messages: List[Message],
) -> Tuple[List[Message], List[Message]]:
    # all_messages = self.get_message_history()
    new_cycle_messages = [m for m in messages if m.new_reasoning_cycle is True]
    if not new_cycle_messages:
        return messages, []
    separating_message = new_cycle_messages[-1]  # The message that starts the new reasoning cycle.
    separating_message_idx = next(i for i, m in enumerate(messages) if m.key == separating_message.key)
    return (
        messages[:separating_message_idx],  # Historic messages up to last worker new cycle system message.
        messages[separating_message_idx:],  # All messages from last worker new cycle system message.
    )


def get_last_message_like(messages: List[Message], like: Callable[[Message], bool]) -> Optional[Message]:
    for message in reversed(messages):
        if like(message):
            return message
    return None


# Get all messages that have .current_structured_plan:
def _has_plan(m: Message) -> bool:
    return (
        m.engine_state is not None
        and m.engine_state.agent_state is not None
        and "coordinator" in m.engine_state.agent_state
        and d2m(m.engine_state.agent_state["coordinator"], CoordinatorCotState).current_plan is not None
    )


# Get last message that have worker delegated_content.
def _has_delegated_content(m: Message) -> bool:
    return (
        m.engine_state is not None
        and m.engine_state.agent_state is not None
        and "worker" in m.engine_state.agent_state
        and d2m(m.engine_state.agent_state["worker"], WorkerCotState).delegated_content is not None
    )


def parse_backtracking(
    input_text: str, backtracking_marker: str, no_backtracking_marker: str
) -> Tuple[bool, Optional[str]]:
    """
    Parses the input text to determine the backtracking status and ID.

    Args:
        input_text (str): Multi-line string to parse.
        backtracking_marker (str): Marker indicating backtracking is enabled.
        no_backtracking_marker (str): Marker indicating backtracking is disabled.

    Returns:
        tuple: (backtracking_true (bool), backtracking_id (str))

    Raises:
        ValueError: If both markers are present or the ID cannot be determined when required.
    """
    lines = input_text.splitlines()

    backtracking_present = any(backtracking_marker in line for line in lines)
    no_backtracking_present = any(no_backtracking_marker in line for line in lines)

    if backtracking_present and no_backtracking_present:
        raise ValueError("Both backtracking_marker and no_backtracking_marker are present in the input.")

    if backtracking_present:
        for line in lines:
            if backtracking_marker in line:
                parts = line.split()
                marker_index = parts.index(backtracking_marker)
                # Check if an ID exists after the marker
                if marker_index + 1 < len(parts):
                    backtracking_id = parts[marker_index + 1]
                    return True, backtracking_id
                else:
                    raise ValueError(f"Backtracking marker '{backtracking_marker}' found but no ID follows it.")

    elif no_backtracking_present:
        return False, None

    raise ValueError("Neither backtracking_marker nor no_backtracking_marker are present in the input.")


def validate_next_episode_analysis(text: str) -> bool:
    """
    Validates episode text format according to specified rules.
    Returns True if valid, raises ValueError with description if invalid.
    """
    # Strip any trailing whitespace and ensure text ends with actual content
    text = text.rstrip()

    # Define regex patterns for each section
    episode_id_pattern = r"Episode ID: ([A-Z][A-Z0-9_]*)\n"
    selection_condition_pattern = r"Selection Condition: (.+)\n"
    episode_name_pattern = r"Episode Name: (.+)\n"

    # Pattern for Analysis section with numbered points
    analysis_pattern = (
        r"\nAnalysis:\n"
        r"1\. Necessity: (.+?)\n"  # Non-greedy match for multi-line content
        r"2\. Appropriateness: (.+?)\n"  # Non-greedy match for multi-line content
    )

    # Pattern for Conclusion
    conclusion_pattern = r"\nConclusion: (.+?)$"  # End of string

    # Combine all patterns
    full_pattern = (
        f"^{episode_id_pattern}"
        f"{selection_condition_pattern}"
        f"{episode_name_pattern}"
        f"{analysis_pattern}"
        f"{conclusion_pattern}"
    )

    # Try to match the full pattern
    match = re.search(full_pattern, text, re.DOTALL)

    if not match:
        # If no match, try to identify specific issues
        if not re.match(f"^{episode_id_pattern}", text):
            raise ValueError(
                "Invalid Episode ID format. Must start with a capital letter and contain only uppercase letters, numbers, and underscores."
            )

        if not re.search(selection_condition_pattern, text):
            raise ValueError(
                "Invalid Selection Condition format. Must be a single line starting with 'Selection Condition: '"
            )

        if not re.search(episode_name_pattern, text):
            raise ValueError("Invalid Episode Name format. Must be a single line starting with 'Episode Name: '")

        if not re.search(r"\nAnalysis:\n1\. Necessity:", text):
            raise ValueError("Invalid Analysis section format. Must start with 'Analysis:' followed by '1. Necessity:'")

        if not re.search(r"2\. Appropriateness:", text):
            raise ValueError("Invalid Analysis section format. Missing or incorrect '2. Appropriateness:' section")

        if not re.search(r"\nConclusion:", text):
            raise ValueError("Invalid Conclusion format. Must start with 'Conclusion:'")

        raise ValueError("Invalid text format. Please check the overall structure and formatting.")

    # Additional validation for Episode ID format
    episode_id = match.group(1)
    if not re.match(r"^[A-Z][A-Z0-9_]*$", episode_id):
        raise ValueError(
            "Episode ID must start with a capital letter and contain only uppercase letters, numbers, and underscores"
        )

    return True


def parse_plan_update(input_text, plan_update_marker, no_plan_update_marker):
    """
    Parses the input text to determine the plan update status and updated plan.

    Args:
        input_text (str): Multi-line string to parse.
        plan_update_marker (str): Marker indicating a plan update.
        no_plan_update_marker (str): Marker indicating no plan update.

    Returns:
        tuple: (plan_update_true (bool), updated_plan (list[str]))

    Raises:
        ValueError: If both markers are present, or updated plan cannot be determined or parsed.
    """
    lines = input_text.splitlines()

    plan_update_present = any(plan_update_marker in line for line in lines)
    no_plan_update_present = any(no_plan_update_marker in line for line in lines)

    if plan_update_present and no_plan_update_present:
        raise ValueError("Both plan_update_marker and no_plan_update_marker are present in the input.")

    if plan_update_present:
        content = ""
        capture = False
        format_note = (
            "\nThe format of plan update must be **a list of EPISODE ID strings**, e.g.:\n"
            '["EP_1", "EP_8", "EP_X9"].\n'
            "Do not include the episode names or details, it is just a list of episode IDs."
        )

        for line in lines:
            if plan_update_marker in line:
                capture = True
                # Start capturing content after the marker
                marker_index = line.find(plan_update_marker) + len(plan_update_marker)
                content += line[marker_index:].strip() + " "
            elif capture:
                # Continue capturing until we find matching brackets
                content += line.strip() + " "
                if "]" in line:
                    break

        # Try to extract content within square brackets
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            try:
                updated_plan = eval(match.group())  # Unsafe for untrusted input; consider json.loads
                if not isinstance(updated_plan, list) or not all(isinstance(item, str) for item in updated_plan):
                    raise ValueError("Updated plan is not a list of strings." + format_note)
                return True, updated_plan
            except Exception as e:
                raise ValueError(f"Failed to parse updated plan: {e}." + format_note)

        raise ValueError("Plan update marker found but no valid list of strings in brackets." + format_note)

    elif no_plan_update_present:
        return False, None

    # If neither marker is present, raise an error
    raise ValueError("Neither plan_update_marker nor no_plan_update_marker is present in the input.")


def get_completed_and_remaining_episodes(
    plan: List[str], selected_episode: str, include_selected: bool = True
) -> Tuple[List[str], List[str]]:
    if selected_episode == "None":  # Special marker for no episode selected == start of the project.
        return [], plan

    completed_episodes = []
    for episode_id in plan:
        if episode_id != selected_episode:
            completed_episodes.append(episode_id)
        else:
            break
    if include_selected:
        completed_episodes.append(selected_episode)

    remaining_episodes = []
    for episode_id in plan:
        if episode_id not in completed_episodes and episode_id != selected_episode:
            remaining_episodes.append(episode_id)

    return completed_episodes, remaining_episodes


def format_episodes_id_name(
    episode_db: List[Dict[str, Any]],
    episode_ids: List[str],
) -> str:
    if len(episode_ids) == 0:
        return "No episodes."
    formatted = ""
    for episode_id in episode_ids:
        episode = [s for s in episode_db if s["episode_id"] == episode_id][0]
        formatted += f"- {episode['episode_id']}: {episode['episode_name']}\n"
    return formatted


def create_worker_actual_task(
    plan: List[str],
    episode_db: List[Dict[str, Any]],
    selected_episode: str,
) -> str:
    task_description_for_worker = """
You need to complete the task shown below.

"""
    episode_dict = [s for s in episode_db if s["episode_id"] == selected_episode][0]
    earlier_episodes = []
    for episode in plan:
        if episode != selected_episode:
            earlier_episodes.append(episode)
        else:
            break

    task_description_for_worker += f"""
### Task name: {episode_dict["episode_name"]}
{episode_dict["episode_details"]}
"""

    if episode_dict["worker_guidance"]:
        task_description_for_worker += f"""
**Guidance**:
{episode_dict["worker_guidance"]}
"""
    if DEBUG__USE_FILTER_TOOLS:
        if episode_dict["tools"] is None:
            task_description_for_worker += """\
***Potentially relevant tools***:
- All tools you have access to.
"""
        elif len(episode_dict["tools"]) > 0:
            task_description_for_worker += """\
***Potentially relevant tools***:
"""
            for tool in episode_dict["tools"]:
                task_description_for_worker += f"- {tool}\n"
        else:
            task_description_for_worker += """\
*You should not need to use any tools for this task.*
"""

    completed = """
### Tasks already completed by previous agents:
"""
    completed_count = 0
    for episode_id in plan:
        task = [s for s in episode_db if s["episode_id"] == episode_id][0]
        if episode_id in earlier_episodes:
            completed_count += 1
            completed += f"""
- {task['episode_name']}"""
    if completed_count == 0:
        completed += """
- None so far. You are the first agent to work on this project."""
    task_description_for_worker += completed

    still_remaining = f"""

### {TASKS_PLANNED_AFTER}:
**NOTE** NEVER PROCEED TO THESE TASKS. Just issue the {TASK_COMPLETED_INDICATOR} if it looks like the work is moving \
to these tasks.
"""
    still_remaining_count = 0
    for episode_id in plan:
        task = [s for s in episode_db if s["episode_id"] == episode_id][0]
        if episode_id not in earlier_episodes and episode_id != selected_episode:
            still_remaining_count += 1
            still_remaining += f"""
- {task['episode_name']}"""
    if still_remaining_count == 0:
        still_remaining += """
- None. You are at the end of the project plan."""
    task_description_for_worker += still_remaining

    return task_description_for_worker


@dataclass
class Episode:
    """Represents a single episode with its details and analysis.

    Args:
        episode_id (str): Unique identifier for the episode.
        selection_condition (Optional[str]): Condition for episode selection. Defaults to None.
        name (str): Name or title of the episode.
        analysis (List[str]): List of analysis points.
        conclusion (str): Conclusion drawn from the episode analysis.
    """

    episode_id: str
    selection_condition: Optional[str]
    name: str
    analysis: List[str]
    conclusion: str


@dataclass
class ParsedDocument:
    """Represents the parsed document containing episodes and overall conclusion.

    Args:
        episodes (List[Episode]): List of parsed episodes.
        overall_conclusion (str): Overall conclusion about all episodes.
        requires_plan_update (bool): Flag indicating if plan update is required.
    """

    episodes: List[Episode]
    overall_conclusion: str
    requires_plan_update: bool


class EpisodeAnalysisParser:
    """Parser for episode-based text documents."""

    # Constants for parsing.
    EPISODE_START = "* === EPISODE ==="
    EPISODE_END = "* === END EPISODE ==="

    EXC_NOTE = (
        "\n\n**IMPORTANT** You must MODIFY your previous response to FIX this problem. DO NOT return the same response! "
        "\nNote that the special characters like `?` and `:` are required exactly as shown."
        "\nNote also that it is case-sensitive."
    )

    def parse_document(self, text: str) -> ParsedDocument:
        """Parse the complete document containing episodes and conclusions.

        Args:
            text (str): Raw text document to parse.

        Returns:
            ParsedDocument: Parsed document with episodes and conclusions.

        Raises:
            ValueError: If the document format is invalid or required sections are missing.
        """
        # Split the document into episodes and conclusion sections.
        parts = text.strip().split(self.EPISODE_END)
        if len(parts) < 2:
            raise ValueError("Document must contain at least one episode and a conclusion section." + self.EXC_NOTE)

        # Parse episodes.
        episodes = []
        for part in parts[:-1]:  # Last part contains conclusion.
            if not part.strip():
                continue
            episodes.append(self._parse_episode(part.strip()))

        # Parse conclusion section.
        conclusion_section = parts[-1].strip()
        overall_conclusion, requires_update = self._parse_conclusion_section(conclusion_section)

        return ParsedDocument(
            episodes=episodes, overall_conclusion=overall_conclusion, requires_plan_update=requires_update
        )

    def _parse_episode(self, episode_text: str) -> Episode:
        """Parse a single episode section.

        Args:
            episode_text (str): Text content of a single episode.

        Returns:
            Episode: Parsed episode object.

        Raises:
            ValueError: If the episode format is invalid or required fields are missing.
        """
        # if not episode_text.startswith(self.EPISODE_START):
        #     raise ValueError("Episode must start with the correct marker.")

        # Extract details using regex.
        episode_id_match = re.search(r"Episode ID: (.+)", episode_text)
        condition_match = re.search(r"Selection Condition: (.+)", episode_text)
        name_match = re.search(r"Name: (.+)", episode_text)

        if not all([episode_id_match, condition_match, name_match]):
            raise ValueError("Episode is missing required fields (ID, Selection Condition, or Name)." + self.EXC_NOTE)

        # Extract and clean analysis points.
        analysis_section = re.search(r"Analysis:\s*((?:(?:\d+\. .+\n?)+))", episode_text)
        if not analysis_section:
            raise ValueError("Episode is missing Analysis section." + self.EXC_NOTE)

        analysis_text = analysis_section.group(1)

        # Check for required analysis components.
        if not re.search(r"\d+\.\s+Necessity:", analysis_text):
            raise ValueError("Analysis section must contain a 'Necessity:' point." + self.EXC_NOTE)
        if not re.search(r"\d+\.\s+Appropriateness:", analysis_text):
            raise ValueError("Analysis section must contain an 'Appropriateness:' point." + self.EXC_NOTE)

        analysis_points = [point.strip() for point in re.findall(r"\d+\. (.+)", analysis_text)]

        # Extract conclusion.
        conclusion_match = re.search(r"Episode Conclusion: (.+)", episode_text)
        if not conclusion_match:
            raise ValueError("Episode is missing `Episode Conclusion:`." + self.EXC_NOTE)

        return Episode(
            episode_id=episode_id_match.group(1).strip(),
            selection_condition=None
            if condition_match.group(1).strip().lower() == "none"
            else condition_match.group(1).strip(),
            name=name_match.group(1).strip(),
            analysis=analysis_points,
            conclusion=conclusion_match.group(1).strip(),
        )

    def _parse_conclusion_section(self, conclusion_text: str) -> tuple[str, bool]:
        """Parse the overall conclusion section.

        Args:
            conclusion_text (str): Text of the conclusion section.

        Returns:
            tuple[str, bool]: Tuple containing overall conclusion and plan update requirement.

        Raises:
            ValueError: If the conclusion format is invalid or required parts are missing.
        """
        # Extract overall conclusion.
        conclusion_match = re.search(
            r"Overall Conclusion: (.+?)(?=\n*REQUIRES PLAN UPDATE\?)", conclusion_text, re.DOTALL
        )
        update_match = re.search(r"REQUIRES PLAN UPDATE\?: (.+)", conclusion_text)

        if not conclusion_match:
            raise ValueError("Conclusion section is missing required part `Overall Conclusion:`" + self.EXC_NOTE)

        overall_conclusion = conclusion_match.group(1).strip()
        requires_update = update_match.group(1).strip().upper() == "YES"

        return overall_conclusion, requires_update


# endregion


class AgentStore(pydantic.BaseModel):
    coordinator: EngineAgent
    worker: EngineAgent

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


CoordinatorReasoningCotStage = Literal[
    "write_observations", "check_backtracking", "analyze_upcoming_episodes", "check_plan"
]
EpisodeDb = List[Dict[str, Any]]
Plan = List[str]


COORDINATOR_REASONING_COT_STAGE_MAP = {
    "write_observations": "1: Write observations.",
    "check_backtracking": "2: Check for backtracking.",
    "analyze_upcoming_episodes": "3. Analyze upcoming episodes.",
    "check_plan": "Step 4: Check the plan.",
}


# This class is only for convenience of typing.
# We transform it back and forth to a dictionary when storing it in the EngineState.
class CoordinatorCotState(pydantic.BaseModel):
    coordinator_reasoning_stage: CoordinatorReasoningCotStage
    # episode_db: EpisodeDb
    current_plan: Plan
    last_episode: str
    whole_project_completed: bool = False


# This class is only for convenience of typing.
# We transform it back and forth to a dictionary when storing it in the EngineState.
class WorkerCotState(pydantic.BaseModel):
    delegated_content: Optional[str] = None

    # NOTE: None = All tools!
    # If DEBUG__USE_FILTER_TOOLS is True, default to an empty list (no tools)
    # If DEBUG__USE_FILTER_TOOLS is False, default to None (all tools)
    delegated_tools: Optional[List[str]] = [] if DEBUG__USE_FILTER_TOOLS else None


# NOTE:
# Define the *additional* tools.
# If this tool_set -> then allow episodes with these tools.
# Otherwise, these tools, and episodes with these tools, are excluded!
# If tools are set to `None` in an episode EPISODE_DB, the additional tools are NOT going to be used unless the tool_set
# is one that allows them!
ADDITIONAL_TOOLS = dict()
ADDITIONAL_TOOLS["full"] = ["feature_extraction_from_text", "balance_data", "data_suite_insights"]
ADDITIONAL_TOOLS["extra"] = ["knn_shapley_data_valuation", "outlier_detection"]
# Therefore:
EXCLUDE_DEFAULT = ADDITIONAL_TOOLS["full"] + ADDITIONAL_TOOLS["extra"]
EXCLUDE_FULL = ADDITIONAL_TOOLS["extra"]


class OpenAICausalityEngine(OpenAIEngineBase):
    def __init__(
        self,
        db: DB,
        session: Session,
        conda_path: Optional[str] = None,
        *,
        api_key: str,
        # ---
        **kwargs: Any,
    ):
        super().__init__(
            db=db,
            session=session,
            conda_path=conda_path,
            api_key=api_key,
        )

        # Set up EngineState state:
        # CASE: First-time initialization only (not when loaded from DB):
        if self._new_session:
            # Will throw errors if the structured plan as defined is invalid:
            # TODO:
            # sanity_check_structured_plan(STRUCTURED_PLAN)

            # Set initial engine state.
            self.session.engine_state = EngineState(
                streaming=False,
                agent="coordinator",
                agent_switched=False,
                agent_state={
                    "coordinator": m2d(
                        CoordinatorCotState(
                            coordinator_reasoning_stage="write_observations",
                            current_plan=self.plan,
                            last_episode="None",
                        )
                    ),
                    "worker": m2d(
                        WorkerCotState(
                            delegated_content=None,
                        )
                    ),
                },
                ui_controlled=UIControlledState(interaction_stage="reason", input_request=None),
            )
        # CASE: When loaded from DB, restore the engine engine state:
        else:
            messages_with_engine_state = self._get_restartable_messages()
            if messages_with_engine_state:
                if messages_with_engine_state[-1].engine_state is None:
                    raise ValueError("EngineState was None.")
                self.session.engine_state = messages_with_engine_state[-1].engine_state

    def _before_define_agents_hook(self) -> None:
        super()._before_define_agents_hook()

        # === Set up tool & episode filtering. ===

        # Get all the allowed tools, that is, all tools except the excluded ones (based on tool_set engine parameter).
        all_available_tools = list_all_tool_names(filter_tool_names=None)
        if DEBUG__PRINT_TOOL_FILTERING_BASED_ON_TOOL_SET:
            print("All available tools:")
            rich.pretty.pprint(all_available_tools)
        tool_set = self.session.engine_params["tool_set"]
        if tool_set == "default":
            exclude_tools = EXCLUDE_DEFAULT
        elif tool_set == "full":
            exclude_tools = EXCLUDE_FULL
        else:
            exclude_tools = []
        allowed_tools = [t for t in all_available_tools if t not in exclude_tools]

        # Go through EPISODE_DB and the cases where tools are set to None, replacing with allowed tools.
        # This is because None represents ALL TOOLS (later in the logic) but we must ensure that ONLY allowed tools
        # # are used.
        if DEBUG__PRINT_TOOL_FILTERING_BASED_ON_TOOL_SET:
            print("Allowed tools:")
            rich.pretty.pprint(allowed_tools)
        self.allowed_tools = allowed_tools
        if DEBUG__PRINT_TOOL_FILTERING_BASED_ON_TOOL_SET:
            print("Allowed tools:")
            rich.pretty.pprint(allowed_tools)
        # Filter episodes.
        self.episode_db = copy.deepcopy(EPISODE_DB)
        # NOTE: Must use self.episode_db, not EPISODE_DB in this engine!
        for idx, episode in enumerate(EPISODE_DB):
            if episode["tools"] is None:
                # If episode has None tools, replace with **allowed tools** (NOT all available tools).
                self.episode_db[idx]["tools"] = allowed_tools
        # NOTE: Now we no longer have any episodes with None (=use all) tools. All tools are explicitly listed.
        # Filter episodes that have any of the excluded tools.

        # Filter out episodes that have any of the excluded tools.
        # Set the self.episode_db to only include episodes that do not have any of the excluded tools.
        filtered_episode_ids = []
        for episode in self.episode_db:
            # If episode has any tools that match exclude_tools, exclude it.
            if any(tool in exclude_tools for tool in episode["tools"]):
                if DEBUG__PRINT_TOOL_FILTERING_BASED_ON_TOOL_SET:
                    print(f">>>>>>>> Excluding episode {episode['episode_id']} because of tools: {episode['tools']}")
                continue
            filtered_episode_ids.append(episode["episode_id"])
        self.episode_db = [ep for ep in self.episode_db if ep["episode_id"] in filtered_episode_ids]
        if DEBUG__PRINT_TOOL_FILTERING_BASED_ON_TOOL_SET:
            print("Episode DB after excluded tools have been filtered out:")
            rich.pretty.pprint(self.episode_db)

        # Filter the PLAN (set self.plan) to only include episodes whose `episode_id`s are in self.episode_db.
        # (i.e. only include episodes that have not been excluded based on the tool_set parameter).
        self.plan = [ep for ep in PLAN if ep in get_all_episode_ids_from_db(self.episode_db)]
        # NOTE: Must use self.plan, not PLAN in this engine!
        if DEBUG__PRINT_TOOL_FILTERING_BASED_ON_TOOL_SET:
            print("PLAN after filtering out episodes with excluded tools:")
            rich.pretty.pprint(self.plan)

        # NOTE: Anywhere else in this engine, use self.episode_db and self.plan, not EPISODE_DB and PLAN!
        # === Set up tool & episode filtering. (END) ===

    @staticmethod
    def get_engine_parameters() -> List[EngineParameter]:
        parent_params = OpenAIEngineBase.get_engine_parameters()
        return parent_params + [ToolSetParameter]

    def _set_initial_messages(self, agent: EngineAgent) -> List[Message]:
        if agent.agent_type == "worker":
            msg_w_dc = get_last_message_like(self.get_message_history(), _has_delegated_content)
            delegated_content = d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerCotState).delegated_content  # type: ignore
        else:
            delegated_content = None

        completed_episodes, remaining_episodes = get_completed_and_remaining_episodes(
            self.plan, self.get_current_last_episode()
        )
        system_message_text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                EPISODE_DB_REPLACE_MARKER: rich.pretty.pretty_repr(self.episode_db),
                PLAN_REPLACE_MARKER: format_episodes_id_name(self.episode_db, self.plan),
                PLAN_COMPLETED_REPLACE_MARKER: format_episodes_id_name(self.episode_db, completed_episodes),
                PLAN_REMAINING_REPLACE_MARKER: format_episodes_id_name(self.episode_db, remaining_episodes),
                WORKER_ACTUAL_TASK_REPLACE_MARKER: str(delegated_content),  # Only applicable for worker.
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
                # Privacy mode templates:
                PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES: PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_1: PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_2: PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
            },
        )

        initial_messages = [
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                visibility="llm_only",
                new_reasoning_cycle=True if agent.agent_type != "simulated_user" else False,
                # TODO: ^ Needs to be refactored sensibly to avoid special casing.
                text=system_message_text,
                agent=agent.agent_type,
            )
        ]
        if agent.first_message_content is not None:
            assert (
                agent.first_message_role is not None
            ), "First message role must be set if first message content is set."
            initial_messages.append(
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role=agent.first_message_role,
                    text=agent.first_message_content,
                    agent=agent.agent_type,
                    visibility="llm_only_ephemeral",
                    engine_state=self.session.engine_state,
                )
            )
        tree_helpers.append_multiple_messages_to_end_of_tree(self.session.messages, initial_messages)

        self.db.update_session(self.session)

        return initial_messages

    def _gather_messages_coordinator(self, agent: EngineAgent) -> Tuple[List[Message], ToolSpecs]:
        engine_log("_gather_messages_coordinator")

        # coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState)
        messages_to_process = []
        tools = None

        historic_messages, last_coordinator_messages = split_message_list_by_last_new_reasoning_cycle_marker(
            self.get_message_history()
        )
        if not last_coordinator_messages:
            raise ValueError("No coordinator messages found for coordinator.")
        if not last_coordinator_messages[0].agent == "coordinator" and last_coordinator_messages[0].role == "system":
            raise ValueError("The last message set did not begin with a coordinator system message.")

        historic_messages, last_worker_messages = split_message_list_by_last_new_reasoning_cycle_marker(
            historic_messages
        )
        if last_worker_messages:
            if not last_worker_messages[0].agent == "worker" and last_worker_messages[0].role == "system":
                raise ValueError(
                    "The last message set in the historic messages component did not begin with "
                    "a worker system message."
                )

        historic_messages = self.exclude_system_messages(historic_messages)
        # ^ This will also remove the old system message.
        last_worker_messages = self.exclude_system_messages(last_worker_messages)

        completed_episodes, remaining_episodes = get_completed_and_remaining_episodes(
            self.get_current_plan(), self.get_current_last_episode()
        )
        coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState)
        coordinator_system_message = last_coordinator_messages[0]
        coordinator_system_message.text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                EPISODE_DB_REPLACE_MARKER: rich.pretty.pretty_repr(self.episode_db),
                PLAN_REPLACE_MARKER: format_episodes_id_name(self.episode_db, self.get_current_plan()),
                PLAN_COMPLETED_REPLACE_MARKER: format_episodes_id_name(self.episode_db, completed_episodes),
                PLAN_REMAINING_REPLACE_MARKER: format_episodes_id_name(self.episode_db, remaining_episodes),
                LAST_EPISODE_REPLACE_MARKER: self.get_current_last_episode(),
                REASONING_STEP_REPLACE_MARKER: COORDINATOR_REASONING_COT_STAGE_MAP[
                    coordinator_state.coordinator_reasoning_stage
                ],
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
                # Privacy mode templates:
                PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES: PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_1: PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_2: PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
            },
        )
        last_coordinator_messages = last_coordinator_messages[1:]

        # [system]
        messages_to_process.append(coordinator_system_message)

        if historic_messages:
            # <separator>
            messages_to_process.append(
                # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="system",
                    text=MESSAGE_OPTIONS["coordinator"]["historic_messages_start"],
                    agent="coordinator",
                )
            )
            # [historic record] - w/o system messages.
            messages_to_process.extend(historic_messages)

        if last_worker_messages:
            # <separator>
            # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
            messages_to_process.append(
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="system",
                    text=MESSAGE_OPTIONS["coordinator"]["historic_messages_last"],
                    agent="coordinator",
                )
            )
            messages_to_process.extend(last_worker_messages)

        # [current conversation]
        # <separator>
        # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
        messages_to_process.append(
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                text=MESSAGE_OPTIONS["coordinator"]["coordinator_continuation"],
                agent="coordinator",
            )
        )
        messages_to_process.extend(last_coordinator_messages)

        # [reminder]
        # Reminder message - to avoid going beyond the subtasks.
        completed_episodes, remaining_episodes = get_completed_and_remaining_episodes(
            self.get_current_plan(), self.get_current_last_episode()
        )
        reminder_message_text = update_templates(
            body_text=MESSAGE_OPTIONS["coordinator"]["reminder"],
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                EPISODE_DB_REPLACE_MARKER: rich.pretty.pretty_repr(self.episode_db),
                PLAN_REPLACE_MARKER: format_episodes_id_name(self.episode_db, self.get_current_plan()),
                PLAN_COMPLETED_REPLACE_MARKER: format_episodes_id_name(self.episode_db, completed_episodes),
                PLAN_REMAINING_REPLACE_MARKER: format_episodes_id_name(self.episode_db, remaining_episodes),
                LAST_EPISODE_REPLACE_MARKER: self.get_current_last_episode(),
                REASONING_STEP_REPLACE_MARKER: COORDINATOR_REASONING_COT_STAGE_MAP[
                    coordinator_state.coordinator_reasoning_stage
                ],
                # The below ones are just in case, not currently used in the reminder message.
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
                # Privacy mode templates:
                PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES: PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_1: PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_2: PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
            },
        )
        reminder_message = Message(
            key=KeyGeneration.generate_message_key(),
            role="system",
            text=reminder_message_text,
            agent="coordinator",
            visibility="llm_only_ephemeral",
            engine_state=self.session.engine_state,
        )
        self._append_message(reminder_message)  # Add to history.
        messages_to_process.append(reminder_message)  # And add to list of messages to send to LLM straight away too.
        # Reminder message [END].

        if DEBUG__PRINT_MESSAGES_PER_AGENT:
            engine_log("--- COORDINATOR messages_to_process ")
            rich.pretty.pprint(messages_to_process)
            engine_log("--- COORDINATOR messages_to_process [END]")

        return messages_to_process, tools

    def _gather_messages_worker(self, agent: EngineAgent) -> Tuple[List[Message], ToolSpecs]:
        engine_log("_gather_messages_worker")
        messages_to_process = []

        msg_w_dc = get_last_message_like(self.get_message_history(), _has_delegated_content)
        if msg_w_dc is None:
            raise ValueError("Delegated content was None.")

        if not DEBUG__USE_FILTER_TOOLS:
            tools = list_all_tool_specs()
        else:
            tool_names = d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerCotState).delegated_tools  # type: ignore
            if tool_names is None:
                raise ValueError("Delegated tools must not be None as we have an explicit tool filtering mechanism.")
            tools = list_all_tool_specs(filter_tool_names=tool_names)

        worker_messages = filter_messages_by_agent(
            self.get_message_history(),
            agent_or_tuple=(agent.agent_type, "simulated_user"),
            # ^ TODO: Refactor this to avoid special casing.
        )
        historic_worker_messages, last_worker_messages = split_message_list_by_last_new_reasoning_cycle_marker(
            worker_messages
        )
        if not last_worker_messages:
            raise ValueError("No worker messages found for worker.")

        worker_agent_system_message = last_worker_messages[0]
        worker_agent_system_message.text = update_templates(
            body_text=agent.system_message_template,
            templates={
                WD_CONTENTS_REPLACE_MARKER: self.describe_working_directory_str(),
                WORKER_ACTUAL_TASK_REPLACE_MARKER: str(
                    d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerCotState).delegated_content  # type: ignore
                ),
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
                # Privacy mode templates:
                PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES: PRIVACY_MODE_SECTION_INDICATOR_CAPABILITIES_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_1: PRIVACY_MODE_SECTION_INDICATOR_RULES_1_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
                PRIVACY_MODE_SECTION_INDICATOR_RULES_2: PRIVACY_MODE_SECTION_INDICATOR_RULES_2_MAP[
                    str(self.session.engine_params["privacy_mode"])
                ],
            },
        )

        historic_worker_messages = self.exclude_system_messages(historic_worker_messages)
        last_worker_messages = self.exclude_system_messages(last_worker_messages)
        # ^ This will also remove the old system message.

        # [system]
        messages_to_process.append(worker_agent_system_message)

        if historic_worker_messages:
            # <separator>
            messages_to_process.append(
                # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="system",
                    text=MESSAGE_OPTIONS["worker"]["record_preamble"],
                    agent="worker",
                )
            )
            # [historic record] - w/o system messages.
            messages_to_process.extend(historic_worker_messages)

        # <separator>
        messages_to_process.append(
            # NOTE: This is a FULLY EPHEMERAL message, not stored in the DB.
            Message(
                key=KeyGeneration.generate_message_key(),
                role="system",
                text=MESSAGE_OPTIONS["worker"]["record_continuation"],
                agent="worker",
            )
        )
        # [current conversation]
        messages_to_process.extend(last_worker_messages)

        # Reminder message - to avoid going beyond the subtasks.
        reminder_message_text = update_templates(
            body_text=MESSAGE_OPTIONS["worker"]["reminder"],
            templates={
                WORKER_ACTUAL_TASK_REPLACE_MARKER: str(
                    d2m(msg_w_dc.engine_state.agent_state["worker"], WorkerCotState).delegated_content  # type: ignore
                ),
                MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
            },
        )
        reminder_message = Message(
            key=KeyGeneration.generate_message_key(),
            role="system",
            text=reminder_message_text,
            agent="worker",
            visibility="llm_only_ephemeral",
            engine_state=self.session.engine_state,
        )
        self._append_message(reminder_message)  # Add to history.
        messages_to_process.append(reminder_message)  # And add to list of messages to send to LLM straight away too.
        # Reminder message [END].

        if DEBUG__PRINT_MESSAGES_PER_AGENT:
            engine_log("--- WORKER messages_to_process ")
            rich.pretty.pprint(messages_to_process)
            engine_log("--- WORKER messages_to_process [END]")

        return messages_to_process, tools

    def get_current_plan(self) -> List[str]:
        msg_w_plan = get_last_message_like(self.get_message_history(), _has_plan)
        if not msg_w_plan:
            return copy.deepcopy(self.plan)
        else:
            return d2m(msg_w_plan.engine_state.agent_state["coordinator"], CoordinatorCotState).current_plan  # type: ignore

    def get_current_last_episode(self) -> str:
        msg_w_plan = get_last_message_like(self.get_message_history(), _has_plan)
        if not msg_w_plan:
            return "None"
        else:
            return d2m(msg_w_plan.engine_state.agent_state["coordinator"], CoordinatorCotState).last_episode  # type: ignore

    def _dispatch_coordinator(self, agent: EngineAgent) -> EngineState:
        last_message = self.get_last_message()

        if last_message.text is None:
            raise ValueError("Last message text must not be None.")
        if last_message.agent != "coordinator":
            raise ValueError("Last message agent must be coordinator.")
        if last_message.engine_state is None:
            raise ValueError("Last message EngineState must not be None.")
        last_message_coordinator_state = d2m(last_message.engine_state.agent_state["coordinator"], CoordinatorCotState)

        if last_message_coordinator_state.coordinator_reasoning_stage is None:
            raise ValueError("Last message EngineState must have reasoning stage.")

        print("================================================================================")
        print(last_message)
        print(last_message_coordinator_state.coordinator_reasoning_stage)
        print("================================================================================")

        if PROJECT_END_MARKER in last_message.text:
            try:
                _, remaining_episodes = get_completed_and_remaining_episodes(
                    self.get_current_plan(), self.get_current_last_episode()
                )
                if len(remaining_episodes) != 0:
                    raise ValueError(
                        f"Project end marker found, but there are remaining episodes: {remaining_episodes}."
                        "You may not complete the project before the end of the plan."
                    )

                last_message_coordinator_state.whole_project_completed = True

                coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState)
                # coordinator_state.coordinator_reasoning_stage = "done"
                coordinator_state.whole_project_completed = True
                self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)

                self.update_state()

                return self.session.engine_state

            except ValueError as e:
                exc_str = str(e)
                self._append_message(
                    message=Message(
                        key=KeyGeneration.generate_message_key(),
                        role="system",
                        visibility="llm_only",
                        agent="coordinator",
                        text=f"{PROBLEM_WITH_OUTPUT_COORDINATOR}:\n{exc_str}",
                    )
                )

        if last_message_coordinator_state.coordinator_reasoning_stage == "write_observations":
            # We aren't doing any special parsing in this step.
            message_text = last_message.text

            # Update EngineState state.
            coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState)
            coordinator_state.coordinator_reasoning_stage = "check_backtracking"
            # coordinator_state.last_episode = self.get_current_last_episode()
            self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)
            self.update_state()

        elif last_message_coordinator_state.coordinator_reasoning_stage == "check_backtracking":
            try:
                message_text = last_message.text

                # Check for backtracking output.
                backtracking_true, backtracking_id = parse_backtracking(
                    message_text, BACKTRACKING_MARKER, NO_BACKTRACKING_MARKER
                )

                # Check if backtracking ID is valid.
                if backtracking_true and backtracking_id not in self.get_current_plan():
                    # TODO: What if the whole plan revamped? Edge case.
                    raise ValueError(f"Backtracking ID '{backtracking_id}' is not a valid episode ID.")

                # TODO: Backtracking action itself.

                # Update EngineState state.
                coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState)
                coordinator_state.coordinator_reasoning_stage = "analyze_upcoming_episodes"
                if backtracking_true:
                    coordinator_state.last_episode = backtracking_id
                else:
                    coordinator_state.last_episode = self.get_current_last_episode()
                self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)
                self.update_state()

            except ValueError as e:
                exc_str = str(e)
                self._append_message(
                    message=Message(
                        key=KeyGeneration.generate_message_key(),
                        role="system",
                        visibility="llm_only",
                        agent="coordinator",
                        text=f"{PROBLEM_WITH_OUTPUT_COORDINATOR}:\n{exc_str}",
                    )
                )

        elif last_message_coordinator_state.coordinator_reasoning_stage == "analyze_upcoming_episodes":
            try:
                message_text = last_message.text

                # Parse.
                parsed_document = EpisodeAnalysisParser().parse_document(message_text)
                rich.pretty.pprint(parsed_document)

                _, remaining_episodes = get_completed_and_remaining_episodes(
                    self.get_current_plan(), self.get_current_last_episode()
                )
                parsed_episode_ids = [episode.episode_id for episode in parsed_document.episodes]
                if not parsed_episode_ids == remaining_episodes[:N_LOOKAHEAD]:
                    raise ValueError(
                        f"Analyzed episodes do not match the expected upcoming episodes: {parsed_episode_ids} vs. "
                        f"{remaining_episodes[:N_LOOKAHEAD]}"
                    )

                # Update EngineState state.
                coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState)
                coordinator_state.coordinator_reasoning_stage = "check_plan"
                # coordinator_state.last_episode = self.get_current_last_episode()
                self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)
                self.update_state()

            except ValueError as e:
                exc_str = str(e)
                print(f"{PROBLEM_WITH_OUTPUT_COORDINATOR}:\n{exc_str}")
                self._append_message(
                    message=Message(
                        key=KeyGeneration.generate_message_key(),
                        role="system",
                        visibility="llm_only",
                        agent="coordinator",
                        text=f"{PROBLEM_WITH_OUTPUT_COORDINATOR}:\n{exc_str}",
                    )
                )

        elif last_message_coordinator_state.coordinator_reasoning_stage == "check_plan":
            try:
                message_text = last_message.text
                coordinator_state = d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState)

                # Check for plan update output.
                plan_update_true, updated_plan = parse_plan_update(
                    message_text, PLAN_UPDATE_MARKER, NO_PLAN_UPDATE_MARKER
                )

                # TODO: Validate the updated plan.

                # Get the next episode given the last episode and the plan.
                if plan_update_true:
                    plan = updated_plan
                else:
                    plan = self.get_current_plan()

                def get_next_episode(plan: List[str], last_episode: str) -> str:
                    # print(plan)
                    # print(last_episode)
                    if last_episode == "None":
                        return plan[0]
                    try:
                        last_episode_idx = plan.index(last_episode)
                    except ValueError:
                        raise ValueError(
                            f"Last episode '{last_episode}' not found in the plan. Have you written out the "
                            "ENTIRE PLAN, including past episodes?\nFor example, if the original plan was ['START_1', "
                            "'START_2', 'MIDDLE_1', 'MIDDLE_3', 'FINAL_1'], and we have finished 'START_2', and you "
                            "want to change the 'MIDDLE_3' to 'MIDDLE_X', you must write out the entire new plan, including "
                            "the past episodes, like so: ['START_1', 'START_2', 'MIDDLE_1', 'MIDDLE_X', 'FINAL_1']."
                        )
                    if last_episode_idx + 1 < len(plan):
                        return plan[last_episode_idx + 1]
                    else:
                        raise ValueError("Last episode is the last episode in the plan.")

                next_episode = get_next_episode(plan, coordinator_state.last_episode)

                # Create the actual task description for the worker.
                try:
                    worker_actual_task = create_worker_actual_task(
                        plan=plan,
                        episode_db=self.episode_db,
                        selected_episode=next_episode,
                    )
                    # Any replacements:
                    # NOTE: If the structured plan features any "replace markers". these must be replaced here!
                    worker_actual_task = update_templates(
                        body_text=worker_actual_task,
                        templates={
                            MAX_CHARS_REPLACE_MARKER: str(self.max_tokens_per_message),
                        },
                    )
                except Exception as e:
                    # engine_log("EXCEPTION IN create_worker_actual_task")
                    # engine_log(e)
                    raise RuntimeError(f"Could not create the worker actual task description:\n{str(e)}") from e
                # engine_log("worker_actual_task")
                # engine_log(worker_actual_task)

                # Update EngineState state.
                coordinator_state.current_plan = plan
                coordinator_state.last_episode = next_episode
                self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)
                self.session.engine_state.agent = "worker"
                self.session.engine_state.agent_switched = True

                # Get the list of relevant tools.
                if DEBUG__USE_FILTER_TOOLS:
                    subtask_dict = [subtask for subtask in self.episode_db if subtask["episode_id"] == next_episode][0]
                    if subtask_dict["tools"] is None:
                        relevant_tools = None
                    else:
                        relevant_tools = subtask_dict["tools"]
                    delegated_tools = list(set(relevant_tools)) if relevant_tools is not None else None
                else:
                    raise NotImplementedError("DEBUG__USE_FILTER_TOOLS is False.")

                # Prepare everything for the worker.
                worker_state = d2m(self.get_state().agent_state["worker"], WorkerCotState)
                worker_state.delegated_content = worker_actual_task
                if DEBUG__USE_FILTER_TOOLS:
                    worker_state.delegated_tools = delegated_tools
                self.session.engine_state.agent_state["worker"] = m2d(worker_state)
                self._set_initial_messages(agent=self.agents_.worker)

                # Update state.
                self.update_state()

                engine_log(worker_actual_task)
                # engine_log("EngineState at the end of `dispatch_next_subtasks`.")
                # engine_log("*" * 100)
                # engine_log(self.session.engine_state)
                # engine_log("*" * 100)

            except ValueError as e:
                exc_str = str(e)
                self._append_message(
                    message=Message(
                        key=KeyGeneration.generate_message_key(),
                        role="system",
                        visibility="llm_only",
                        agent="coordinator",
                        text=f"{PROBLEM_WITH_OUTPUT_COORDINATOR}:\n{exc_str}",
                    )
                )

        else:
            raise ValueError(
                f"Invalid coordinator reasoning stage: {last_message_coordinator_state.coordinator_reasoning_stage}"
            )

        return self.session.engine_state

    def _dispatch_worker(self, agent: EngineAgent) -> EngineState:
        last_message = self.get_last_message()

        if TASK_COMPLETED_INDICATOR in str(last_message.text) or TASK_STOPPED_INDICATOR in str(last_message.text):
            self.session.engine_state.agent_switched = True
            self.session.engine_state.agent = "coordinator"

            # Update coordinator state:
            coordinator_state = d2m(self.session.engine_state.agent_state["coordinator"], CoordinatorCotState)
            coordinator_state.coordinator_reasoning_stage = "write_observations"
            self.session.engine_state.agent_state["coordinator"] = m2d(coordinator_state)

            self._set_initial_messages(agent=self.agents_.coordinator)
            self.update_state()

        else:
            self.session.engine_state.user_message_requested = True

            # Exceptions to asking for user input.
            # Control will be handed over back to the worker in these cases.
            if last_message.text is not None:
                if GENERATED_CODE_FORMAT_ERROR_MSG in last_message.text:
                    self.session.engine_state.user_message_requested = False
                if LONG_MESSAGE_SPLIT_INDICATOR in last_message.text:
                    self.session.engine_state.user_message_requested = False

        return self.session.engine_state

    def define_agents(self) -> Dict[Agent, EngineAgent]:
        # AgentStore just to act as a dotdict for convenient access.
        self.agents_ = AgentStore(
            coordinator=EngineAgent(
                "coordinator",
                first_message_content=MESSAGE_OPTIONS["coordinator"]["first_message_content"],
                system_message_template=MESSAGE_OPTIONS["coordinator"]["system_message_template"],
                first_message_role="assistant",
                set_initial_messages=OpenAICausalityEngine._set_initial_messages,  # type: ignore
                gather_messages=OpenAICausalityEngine._gather_messages_coordinator,  # type: ignore
                dispatch=OpenAICausalityEngine._dispatch_coordinator,  # type: ignore
            ),
            worker=EngineAgent(
                "worker",
                system_message_template=MESSAGE_OPTIONS["worker"]["system_message_template"],
                first_message_content=MESSAGE_OPTIONS["worker"]["first_message_content"],
                first_message_role="assistant",
                set_initial_messages=OpenAICausalityEngine._set_initial_messages,  # type: ignore
                gather_messages=OpenAICausalityEngine._gather_messages_worker,  # type: ignore
                dispatch=OpenAICausalityEngine._dispatch_worker,  # type: ignore
            ),
        )
        as_dict = self.agents_.model_dump()  # {"coordinator": coordinator EngineAgent, ...}
        return as_dict  # type: ignore

    def define_initial_agent(self) -> Agent:
        return "coordinator"

    @staticmethod
    def get_engine_name() -> str:
        return "openai_causality"

    def project_completed(self) -> bool:
        return d2m(self.get_state().agent_state["coordinator"], CoordinatorCotState).whole_project_completed

    def get_worker_messages_historic_last(self) -> Tuple[List[Message], List[Message]]:
        # Return [the historic messages] and [the last reasoning cycle messages] for the worker.
        all_worker_messages = self.get_all_worker_messages()
        worker_new_cycle_messages = [m for m in all_worker_messages if m.new_reasoning_cycle is True]
        if not worker_new_cycle_messages:
            raise ValueError("No new reasoning cycle messages found for worker.")
        separating_message = worker_new_cycle_messages[-1]  # The message that starts the new reasoning cycle.
        separating_message_idx = next(i for i, m in enumerate(all_worker_messages) if m.key == separating_message.key)
        return (
            all_worker_messages[:separating_message_idx],  # Historic worker messages.
            all_worker_messages[separating_message_idx:],  # Last reasoning cycle worker messages.
        )

    def get_all_messages_historic_last(self) -> Tuple[List[Message], List[Message]]:
        all_messages = self.get_message_history()
        worker_new_cycle_messages = [m for m in all_messages if m.agent == "worker" and m.new_reasoning_cycle is True]
        if not worker_new_cycle_messages:
            return all_messages, []
        separating_message = worker_new_cycle_messages[-1]  # The message that starts the new reasoning cycle.
        separating_message_idx = next(i for i, m in enumerate(all_messages) if m.key == separating_message.key)
        return (
            all_messages[:separating_message_idx],  # Historic messages up to last worker new cycle system message.
            all_messages[separating_message_idx:],  # All messages from last worker new cycle system message.
        )

    def exclude_system_messages(self, messages: List[Message]) -> List[Message]:
        return [m for m in messages if m.role != "system"]

    def exclude_non_coordinator_system_messages(self, messages: List[Message]) -> List[Message]:
        return [m for m in messages if (m.role != "system" or m.agent == "coordinator")]

    def get_all_worker_messages(self) -> List[Message]:
        return [m for m in self.get_message_history() if m.agent in ("worker", "supervisor")]

    def _reset_engine_state_after_api_call(self) -> None:
        # Reset the engine state after an API call is made.
        worker_state = d2m(self.session.engine_state.agent_state["worker"], WorkerCotState)
        coordinator_state = d2m(self.session.engine_state.agent_state["coordinator"], CoordinatorCotState)
        self.session.engine_state = EngineState(
            streaming=True,
            # Do not reset agent:
            agent=self.session.engine_state.agent,
            agent_switched=False,
            agent_state={
                # We do not need to reset anything in the coordinator state.
                "coordinator": m2d(coordinator_state),
                # We do not need to reset anything in the worker state.
                "worker": m2d(worker_state),
            },
            ui_controlled=UIControlledState(),
        )

    def _llm_call(self, messages: List[Message], tools: ToolSpecs) -> StreamLike:
        # Some sanity checks:
        if not messages:
            raise ValueError("No messages to process.")
        else:
            system_messages = [m for m in messages if m.role == "system"]
            if not system_messages:
                raise ValueError("No system messages found in the messages to process.")
            if messages[0].role != "system":
                raise ValueError("First message must be a system message.")

        messages_to_send_to_openai = [
            self._handle_openai_message_format(m)  # type: ignore
            for m in messages
            if m.visibility not in ("ui_only", "system_only")
        ]

        if DEBUG__PRINT_MESSAGES_TO_LLM:
            engine_log("=========================")
            engine_log("messages_to_send_to_openai")
            rich.pretty.pprint(messages_to_send_to_openai)
            engine_log("=========================")
        if DEBUG__PRINT_TOOLS:
            engine_log("tools\n", tools)
            engine_log("=========================")
        self._log_messages(
            messages=messages_to_send_to_openai,
            tools=tools,
            metadata={"agent": self.session.engine_state.agent},
        )
        completion_kwargs = dict(
            messages=messages_to_send_to_openai,
            stream=True,
        )
        if tools is not None and len(tools) > 0:
            completion_kwargs["tools"] = tools
        stream = self.initialize_completion()(**completion_kwargs)
        if self.supports_streaming_token_count() is False:
            self._count_tokens(messages_in=messages_to_send_to_openai, tools=tools)

        # Process the stream and yield the results.
        message_chunks: List[str] = []
        is_tool_call = False
        tool_call_content: Dict[str, Any] = {"content": None, "role": "assistant", "tool_calls": {}}

        # Reset the engine state.
        self._reset_engine_state_after_api_call()

        chunk_tracker = ChunkTracker()
        for chunk in stream:
            action = self.stream_start_hook(chunk)
            if action == "continue":
                continue

            if len(chunk.choices) == 0:
                if chunk.usage is not None:
                    # Special final message that contains only usage. Its choices will be [] and usage will be provided.

                    if self.supports_streaming_token_count() is True:
                        self._count_tokens(chunk.usage)
                    else:
                        # Sanity check:
                        raise ValueError(
                            "Engine `supports_streaming_token_count()` was False `usage` was provided by the API."
                        )

                    # We do not need to proceed to the stream handling logic when we receive this special usage-only
                    # chunk, therefore set delta to None so that it triggers a `continue` in the loop.
                    delta = None

                else:
                    # Otherwise, a chunk with no choices is considered invalid.
                    delta = None

            else:
                # A normal chunk with choices.
                delta = chunk.choices[0].delta  # type: ignore

            if DEBUG__PRINT_DELTA:
                engine_log("delta", delta)

            if delta is None:
                continue

            if delta.content is not None:
                # Case: response is text message, stream it to UI.
                self.session.engine_state.response_kind = ResponseKind.TEXT_MESSAGE

                # Handle previous message if needed ---
                chunk_tracker.update(sentinel="text")
                self._append_last_message_if_required(
                    chunk_tracker,
                    last_message_is_tool_call=is_tool_call,
                    last_message_text_chunks=message_chunks,
                    last_message_tool_call_content=tool_call_content,
                    agent=self.session.engine_state.agent,
                )
                is_tool_call = False
                tool_call_content = {"content": None, "role": "assistant", "tool_calls": {}}
                # ---

                message_chunk = delta.content  # type: ignore
                message_chunk = message_chunk if isinstance(message_chunk, str) else ""
                message_chunks.append(message_chunk)
                yield message_chunk

            else:
                # Case: response is tool call, process it.
                # Solution from:
                # https://community.openai.com/t/has-anyone-managed-to-get-a-tool-call-working-when-stream-true/498867/11

                if delta.tool_calls:
                    self.session.engine_state.response_kind = ResponseKind.TOOL_REQUEST

                    # Handle previous message if needed ---
                    chunk_tracker.update(sentinel="tool_call")
                    self._append_last_message_if_required(
                        chunk_tracker,
                        last_message_is_tool_call=is_tool_call,
                        last_message_text_chunks=message_chunks,
                        last_message_tool_call_content=tool_call_content,
                        agent=self.session.engine_state.agent,
                    )
                    is_tool_call = True
                    message_chunks = []
                    # ---

                    piece = delta.tool_calls[0]  # type: ignore
                    tool_call_content["tool_calls"][piece.index] = tool_call_content["tool_calls"].get(
                        piece.index, {"id": None, "function": {"arguments": None, "name": ""}, "type": "function"}
                    )
                    if piece.id:
                        tool_call_content["tool_calls"][piece.index]["id"] = piece.id
                    if piece.function.name:  # type: ignore
                        tool_call_content["tool_calls"][piece.index]["function"]["name"] = piece.function.name  # type: ignore
                    if piece.function.arguments:
                        if tool_call_content["tool_calls"][piece.index]["function"]["arguments"] is None:
                            tool_call_content["tool_calls"][piece.index]["function"]["arguments"] = ""
                        tool_call_content["tool_calls"][piece.index]["function"]["arguments"] += (
                            piece.function.arguments
                        )  # type: ignore
                    # engine_log(">>>> Yielding LoadingIndicator")
                    yield LoadingIndicator

                else:
                    # End of streaming reached.
                    # The final chunk is always `content`=None and `tool_calls`=None.
                    chunk_tracker.update(sentinel="end_of_stream")
                    self._append_last_message_if_required(
                        chunk_tracker,
                        last_message_is_tool_call=is_tool_call,
                        last_message_text_chunks=message_chunks,
                        last_message_tool_call_content=tool_call_content,
                        agent=self.session.engine_state.agent,
                    )
                    # engine_log(">>> SETTING STREAMING DONE")
                    self.session.engine_state.streaming = False

                    # Check if code execution is needed.
                    last_message_text = self.get_last_message().text
                    if last_message_text is not None:
                        engine_log("Checking if code execution is needed.")
                        if is_code_generated(last_message_text):
                            engine_log("Code execution IS needed.")

                            self.session.engine_state.response_kind = ResponseKind.CODE_GENERATION
                            # TODO: Spot this earlier.

                            code_extract_fail = False
                            try:
                                dependencies, code, files_in, files_out = code_extract(last_message_text)
                            except Exception as e:
                                code_extract_fail = True
                                code_extract_fail_msg = (
                                    f"{GENERATED_CODE_FORMAT_ERROR_MSG}. Try again. Error details:\n```\n{e}\n```"
                                )
                            if not code_extract_fail:
                                self._append_message(
                                    Message(
                                        key=KeyGeneration.generate_message_key(),
                                        role="assistant",
                                        text=None,
                                        generated_code_dependencies=dependencies,
                                        generated_code=code,
                                        files_in=files_in,
                                        files_out=files_out,
                                        visibility="system_only",
                                        engine_state=self.session.engine_state,
                                    )
                                )
                            else:
                                self.session.engine_state.response_kind = ResponseKind.TEXT_MESSAGE
                                self._append_message(
                                    Message(
                                        key=KeyGeneration.generate_message_key(),
                                        role="assistant",
                                        text=code_extract_fail_msg,
                                        visibility="all",
                                    )
                                )

    def create_new_message_branch(self) -> bool:
        # Create a new message branch.
        last_branch_point = tree_helpers.get_last_branch_point_node(self.session.messages)
        if last_branch_point is None:
            raise ValueError("No branch point found in the message history.")
        if last_branch_point != tree_helpers.get_last_terminal_child(self.session.messages):
            last_branch_point.add(
                Message(
                    key=KeyGeneration.generate_message_key(),
                    role="new_branch",
                    text=None,
                    agent="worker",
                    visibility="system_only",
                )
            )
            return True
        return False

    def _update_system_message(self, message: Message, template: str) -> Message:
        message = copy.deepcopy(message)
        message.text = template.replace("{WD_CONTENTS}", self.describe_working_directory_str())
        return message


class AzureOpenAICausalityEngine(
    AzureOpenAIEngineMixin,  # Mixing needs to come first to override the methods correctly.
    OpenAICausalityEngine,
):
    def __init__(
        self,
        db: DB,
        session: Session,
        conda_path: Optional[str] = None,
        *,
        api_key: str,
        azure_openai_config: AzureOpenAIConfig,
        # ---
        **kwargs: Any,
    ):
        # Initialize the Mixin.
        AzureOpenAIEngineMixin.__init__(
            self,
            azure_openai_config=azure_openai_config,
        )
        # Initialize the Base class.
        OpenAICausalityEngine.__init__(
            self,
            db=db,
            session=session,
            conda_path=conda_path,
            api_key=api_key,
            # ---
            **kwargs,
        )

    @staticmethod
    def get_engine_parameters() -> List[EngineParameter]:
        parent_params = AzureOpenAIEngineMixin.get_engine_parameters()
        return parent_params + [ToolSetParameter]

    @staticmethod
    def get_engine_name() -> str:
        return "azure_openai_causality"

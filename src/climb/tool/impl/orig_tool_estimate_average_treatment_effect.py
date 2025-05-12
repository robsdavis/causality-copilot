import os
from typing import Any, Optional, Dict

from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
from pathlib import Path

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

def estimate_ate(
    tc: ToolCommunicator,
    df_file_path: str,
    treatment: str,
    outcome: str,
    adjustment_covariates: list,
    ground_truth_column: Optional[str] = None,
    model: Optional[Any] = None,
) -> None:
    """
    Estimate the average treatment effect (ATE) using a regression-based approach.
    Additionally, if a ground truth treatment effect column is provided, compute PEHE.
    
    The ATE is defined as:
        ATE = E[Y(1) - Y(0)] = E_X[E[Y | W=1, X] - E[Y | W=0, X]]
    
    This function:
    - Loads the dataset from df_file_path.
    - Uses the provided treatment, outcome, and adjustment_covariates to fit a model for E[Y | W, X].
    - Creates two counterfactual copies of the dataset with the treatment set to 1 and 0, respectively.
    - Predicts the outcome under both interventions and computes the mean difference (ATE).
    - If ground_truth_column is provided and exists in the dataset, computes PEHE as:
          PEHE = sqrt(mean((tau_hat - tau_true)^2))
    
    Parameters:
      - tc: ToolCommunicator for logging and setting returns.
      - df_file_path (str): Path to the CSV file containing the dataset.
      - treatment (str): Column name for the binary treatment variable.
      - outcome (str): Column name for the outcome variable.
      - adjustment_covariates (list of str): List of column names for the adjustment covariates.
      - ground_truth_column (str, optional): Name of the column containing the ground truth treatment effects.
      - model: A regression model instance (if None, RandomForestRegressor will be used).
    
    Sets:
      - tool_return: A message summarizing the estimated ATE and PEHE (if computed).
      - user_report: A list with details about the estimated ATE, PEHE (if computed), and the adjustment_covariates used.
    """

    df_file_path = Path(df_file_path)
    df = pd.read_csv(df_file_path)

    # Use a default model if none is provided.
    model = RandomForestRegressor()

    # Define features: treatment and adjustment_covariates.
    tc.print(f"The adjustment set used for the ATE calculation: {adjustment_covariates}.")
    features = [treatment] + adjustment_covariates

    # Fit the model f(W, X) to predict Y.
    X_train = df[features]
    y_train = df[outcome]
    model.fit(X_train, y_train)

    # Create counterfactual copies of the dataset.
    df_treat1 = df.copy()
    df_treat0 = df.copy()
    df_treat1[treatment] = 1  # Intervention: treatment = 1.
    df_treat0[treatment] = 0  # Intervention: treatment = 0.

    # Ensure the order of features matches training.
    X_treat1 = df_treat1[features]
    X_treat0 = df_treat0[features]

    # Predict outcomes under both interventions.
    y_pred_treat1 = model.predict(X_treat1)
    y_pred_treat0 = model.predict(X_treat0)

    # Compute individual-level differences and then the ATE.
    differences = y_pred_treat1 - y_pred_treat0
    ate = np.mean(differences)

    # Here differences is our tau_hat.
    tau_hat = differences

    # Initialize PEHE as None.
    pehe = None
    if ground_truth_column is not None and ground_truth_column in df.columns:
        tau_true = df[ground_truth_column].values
        pehe = np.sqrt(np.mean((tau_hat - tau_true)**2))
        tc.print("Ground truth treatment effects found in the data. Using them to calculate PEHE.")
    else:
        tc.print("No ground truth treatment effect column provided; PEHE will not be calculated.")

    # round the ATE and PEHE for better readability.
    ate = round(ate, 4) 
    tc.print(f"Estimated ATE: {ate}")
    if pehe is not None:
        pehe = round(pehe, 4)
        tc.print(f"Estimated PEHE: {pehe}")
    
    # Build the messages for tool_return and user_report.
    message = f"ATE estimation completed. The estimated average treatment effect is {ate}."
    report = [f"Estimated ATE: {ate}", f"Adjustment covariates used: {adjustment_covariates}"]
    if pehe is not None:
        message += f" PEHE is {pehe}."
        report.append(f"Estimated PEHE: {pehe}")
    
    tc.set_returns(
        tool_return=message,
        user_report=report,
    )
    return {
        'ate': ate,
        'pehe': pehe,
        'model': model,
        'adjustment_covariates': adjustment_covariates,
    }


class EstimateATE(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        df_file_path = os.path.join(self.working_directory, kwargs["df_file_path"])
        treatment = kwargs["treatment"]
        outcome = kwargs["outcome"]
        adjustment_covariates = kwargs["adjustment_covariates"]
        # Optional: pass a model instance if needed.
        model = kwargs.get("model", None)
        # Optional ground truth column name.
        ground_truth_column = kwargs.get("ground_truth_column", None)
        thrd, out_stream = execute_tool(
            estimate_ate,
            df_file_path=df_file_path,
            treatment=treatment,
            outcome=outcome,
            adjustment_covariates=adjustment_covariates,
            ground_truth_column=ground_truth_column,
            model=model,
            wd=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "estimate_ate"

    @property
    def description(self) -> str:
        return (
            "Estimates the average treatment effect (ATE) using a regression approach. "
            "The function fits a model to predict the outcome based on treatment and adjustment_covariates, "
            "then computes the difference in predicted outcomes under interventions W=1 and W=0. "
            "If a ground truth treatment effect column is provided, it also calculates PEHE."
        )

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
                        "df_file_path": {
                            "type": "string",
                            "description": "Path to the CSV file containing the dataset.",
                        },
                        "treatment": {
                            "type": "string",
                            "description": "Column name for the binary treatment variable.",
                        },
                        "outcome": {
                            "type": "string",
                            "description": "Column name for the outcome variable.",
                        },
                        "adjustment_covariates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of column names for the adjustment covariates.",
                        },
                        "ground_truth_column": {
                            "type": "string",
                            "description": "Optional. Column name containing the ground truth treatment effects. "
                                           "If provided, PEHE is calculated. It is a requirement for calculating PEHE. "
                                           "This columns name must be passed if the column exists.",
                        },
                    },
                    "required": ["df_file_path", "treatment", "outcome", "adjustment_covariates"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "Estimates the average treatment effect (ATE) by fitting a regression model to predict the outcome "
            "given the treatment and adjustment_covariates, computes the mean difference in predicted outcomes "
            "when the treatment is set to 1 versus 0, and calculates PEHE if a ground truth treatment effect column is provided."
        )

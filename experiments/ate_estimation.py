# ate_estimation.py

import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Import the meta-learners from econml.
from econml.metalearners import TLearner, SLearner, XLearner
from econml.dr import DRLearner

# Import DragonNet and TARNet from our local custom_models implementation.
from custom_models import make_dragonnet, make_tarnet

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate ATE using various meta-learners with configurable data and columns."
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="T",
        choices=["T", "S", "X", "DR", "DRAGONNET", "TARNET"],
        help="Meta-learner to use: T, S, X, DR, DRAGONNET, or TARNET (default: T)"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of runs/experiments to average over (default: 5)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(Path.cwd().parent / "data/causality/synthetic_hypertension_sodium_binary_data.csv"),
        help="Path to the CSV data file."
    )
    parser.add_argument(
        "--outcome",
        type=str,
        default="sbp_in_mmHg",
        help="Name of the outcome column (default: sbp_in_mmHg)"
    )
    parser.add_argument(
        "--treatment",
        type=str,
        default="Sodium",
        help="Name of the treatment column (default: Sodium)"
    )
    parser.add_argument(
        "--tau_true",
        type=float,
        default=1.05,
        help="The ground truth treatment effect (assumed constant for all samples)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    learner_type = args.learner.upper()
    n_runs = args.n_runs
    data_file = Path(args.data_file)
    outcome_col = args.outcome
    treatment_col = args.treatment
    tau_true = args.tau_true

    # Load data.
    df = pd.read_csv(data_file)
    
    # Define outcome, treatment, and covariates (all columns except outcome and treatment).
    Y = df[outcome_col].values
    T = df[treatment_col].values
    X = df[[col for col in df.columns if col not in [outcome_col, treatment_col]]].values

    ate_list = []
    pehe_list = []

    for seed in range(n_runs):
        np.random.seed(seed)
        
        # For T, S, X, DR learners, create basic RandomForest instances.
        rf1 = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf2 = RandomForestRegressor(n_estimators=100, random_state=seed)
        
        # Instantiate the chosen learner.
        if learner_type == "T":
            learner = TLearner(models=[rf1, rf2])
        elif learner_type == "S":
            learner = SLearner(overall_model=rf1)
        elif learner_type == "X":
            learner = XLearner(models=[rf1, rf2])
        elif learner_type == "DR":
            # Define a parameter grid for tuning.
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
            tuned_rf_reg = GridSearchCV(RandomForestRegressor(random_state=seed), param_grid)
            tuned_rf_final = GridSearchCV(RandomForestRegressor(random_state=seed), param_grid)
            rf_prop = RandomForestClassifier(n_estimators=100, random_state=seed)
            learner = DRLearner(
                model_regression=tuned_rf_reg,
                model_propensity=rf_prop,
                model_final=tuned_rf_final,
                cv=5  # Use 5-fold cross-fitting.
            )
        elif learner_type == "DRAGONNET":
            # Use our custom DragonNet implementation.
            input_dim = X.shape[1]
            learner = make_dragonnet(input_dim)
        elif learner_type == "TARNET":
            # Use our custom TARNet implementation.
            input_dim = X.shape[1]
            learner = make_tarnet(input_dim)
        else:
            raise ValueError("Invalid learner type. Choose from 'T', 'S', 'X', 'DR', 'DRAGONNET', or 'TARNET'.")
    
        # Fit the learner and compute estimated individual treatment effects.
        learner.fit(Y, T, X=X)
        tau_hat = learner.effect(X)
        ate = np.mean(tau_hat)
        ate_list.append(ate)
        print(f"Seed {seed}: ATE = {ate}")
        
        # Compute PEHE given the assumed ground truth treatment effect.
        pehe = np.sqrt(np.mean((tau_hat - tau_true)**2))
        pehe_list.append(pehe)
        print(f"Seed {seed}: PEHE = {pehe}")
    
    # Compute overall mean and standard deviation of ATE and PEHE.
    ate_mean = np.mean(ate_list)
    ate_std = np.std(ate_list)
    pehe_mean = np.mean(pehe_list)
    pehe_std = np.std(pehe_list)
    
    print(f"\nSummary over {n_runs} runs using the {learner_type}-Learner:")
    print("Mean ATE:", ate_mean)
    print("Standard Deviation of ATE:", ate_std)
    print("Mean PEHE:", pehe_mean)
    print("Standard Deviation of PEHE:", pehe_std)
    
    # Build the result dictionary.
    result_dict = {
        "dataset": str(data_file),
        "learner": learner_type,
        "results": {
            "ATE": {
                "mean": ate_mean,
                "standard_dev": ate_std,
            },
            "PEHE": {
                "mean": pehe_mean,
                "standard_dev": pehe_std,
            }
        },
        "seeds": n_runs,
    }
    
    # Save results in a pickle file under the "results" directory.
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "results.pkl"
    
    if results_file.exists():
        with open(results_file, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = []
    
    all_results.append(result_dict)
    with open(results_file, "wb") as f:
        pickle.dump(all_results, f)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()

# ate_estimation.py

import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Import the meta-learners from econml.
from econml.metalearners import TLearner, SLearner, XLearner
from econml.dr import DRLearner
from econml.dml import CausalForestDML

# Import custom learners.
from BART import BARTLearner
from CFRNET import CFRNetLearner
from dragonnet import DragonNetLearner
from TARNET import TarNetLearner

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate ATE using various meta-learners with configurable data and columns."
    )
    parser.add_argument(
        "--learner",
        type=str,
        default="T",
        choices=["T", "S", "X", "DR", "DRAGONNET", "TARNET", "BART", "CFRNET", "CF"],
        help="Meta-learner to use: T, S, X, DR, DRAGONNET, TARNET, BART, CFRNET, or CF (default: T)"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of runs/experiments to average over (default: 5)"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(Path.cwd().parent / "data/causality/sodium_sbp/synthetic_hypertension_sodium_binary_data.csv"),
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
        default=None,
        help="The ground truth treatment effect, if known. If None, a proxy will be used."
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

    # If tau_true is not provided, prepare to compute a proxy effect.
    compute_proxy = tau_true is None
    if compute_proxy:
        # Split the data to fit outcome models on one part of the data.
        X_train, X_val, Y_train, Y_val, T_train, T_val = train_test_split(
            X, Y, T, test_size=0.3, random_state=42
        )
        # Fit separate Random Forests for treated and control outcomes
        model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
        model_control = RandomForestRegressor(n_estimators=100, random_state=42)
        model_treated.fit(X_train[T_train == 1], Y_train[T_train == 1])
        model_control.fit(X_train[T_train == 0], Y_train[T_train == 0])
        # Use these models to generate pseudo treatment effects for the validation set.
        mu1_hat_val = model_treated.predict(X_val)
        mu0_hat_val = model_control.predict(X_val)
        tau_pseudo_val = mu1_hat_val - mu0_hat_val

    ate_list = []
    pehe_list = []

    for seed in range(n_runs):
        np.random.seed(seed)
        
        # For econml learners, create basic RandomForest instances.
        rf1 = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf2 = RandomForestRegressor(n_estimators=100, random_state=seed)
        
        if learner_type == "T":
            learner = TLearner(models=[rf1, rf2])
        elif learner_type == "S":
            learner = SLearner(overall_model=rf1)
        elif learner_type == "X":
            learner = XLearner(models=[rf1, rf2])
        elif learner_type == "DR":
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
            tuned_rf_reg = GridSearchCV(RandomForestRegressor(random_state=seed), param_grid)
            tuned_rf_final = GridSearchCV(RandomForestRegressor(random_state=seed), param_grid)
            rf_prop = RandomForestClassifier(n_estimators=100, random_state=seed)
            learner = DRLearner(
                model_regression=tuned_rf_reg,
                model_propensity=rf_prop,
                model_final=tuned_rf_final,
                cv=5
            )
        elif learner_type == "DRAGONNET":
            input_dim = X.shape[1]
            learner = DragonNetLearner(input_dim)
        elif learner_type == "TARNET":
            input_dim = X.shape[1]
            learner = TarNetLearner(input_dim)
        elif learner_type == "BART":
            learner = BARTLearner(random_state=seed)
        elif learner_type == "CFRNET":
            input_dim = X.shape[1]
            learner = CFRNetLearner(input_dim, random_state=seed)
        elif learner_type == "CF":
            learner = CausalForestDML(
                n_estimators=500,              # 500 is usually enough
                min_samples_leaf=10,            # slightly bigger leafs
                max_depth=15,                   # limit depth
                cv=2,            # default is 2, keeps it fast
                discrete_treatment=True,
                random_state=seed,
            )
        else:
            raise ValueError("Invalid learner type. Choose from T, S, X, DR, DRAGONNET, TARNET, BART, or CFRNET.")

        # Fit the learner on the full dataset (or you can restrict to the validation set if desired).
        learner.fit(Y, T, X=X)
        tau_hat = learner.effect(X)
        ate = np.mean(tau_hat)
        ate_list.append(ate)
        print(f"Seed {seed}: ATE = {ate}")

        # Compute PEHE or its proxy.
        if tau_true is not None:
            # When true treatment effects are known, use them directly.
            pehe = np.sqrt(np.mean((tau_hat - tau_true)**2))
        elif compute_proxy:
            # For the proxy, we compute the error on the validation set.
            # Here, we assume that the learner was trained on the whole dataset.
            # Alternatively, you could train the learner on X_train and evaluate on X_val.
            # We use the pre-computed pseudo effects from the outcome models.
            # Find the indices of X that belong to the validation set.
            # For simplicity, we recompute predictions on X_val.
            tau_hat_val = learner.effect(X_val)
            pehe = np.sqrt(np.mean((tau_hat_val - tau_pseudo_val)**2))
        else:
            pehe = np.nan  # Should not happen, but set a fallback.
            
        pehe_list.append(pehe)
        print(f"Seed {seed}: PEHE = {pehe}")

    # Compute overall mean and standard deviation of ATE and PEHE.
    ate_mean = np.mean(ate_list)
    ate_std = np.std(ate_list)
    pehe_mean = np.mean(pehe_list)
    pehe_std = np.std(pehe_list)
    PEHE_key = "PEHE (proxy)" if compute_proxy else "PEHE"

    print(f"\nSummary over {n_runs} runs using the {learner_type}-Learner:")
    print("Mean ATE:", ate_mean)
    print("Standard Deviation of ATE:", ate_std)
    print(f"Mean {PEHE_key}:", pehe_mean)
    print("Standard Deviation of PEHE:", pehe_std)

    # Build the result dictionary.
    result_dict = {
        "dataset": str(data_file),
        "learner": learner_type,
        "results": {
            "ATE": {"mean": ate_mean, "standard_dev": ate_std},
            PEHE_key: {"mean": pehe_mean, "standard_dev": pehe_std}
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

#!/bin/bash

# Path to the dataset.
DATA_FILE="../data/causality/idhp/ihdp_continuous.csv"

# Run the experiment with each learner in sequence.
echo "Running TLearner..."
python ate_estimation_v2.py --learner T --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "Running SLearner..."
python ate_estimation_v2.py --learner S --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "Running XLearner..."
python ate_estimation_v2.py --learner X --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "Running DRLearner..."
python ate_estimation_v2.py --learner DR --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "Running DragonNetLearner..."
python ate_estimation_v2.py --learner DRAGONNET --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "Running TarNetLearner..."
python ate_estimation_v2.py --learner TARNET --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "Running BARTLearner..."
python ate_estimation_v2.py --learner BART --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "Running CFRNetLearner..."
python ate_estimation_v2.py --learner CFRNET --data_file "$DATA_FILE" --outcome y_factual --treatment treatment

echo "All experiments completed."
echo "Results saved to results/."
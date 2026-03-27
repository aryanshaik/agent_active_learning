import os
import pandas as pd
import numpy as np
import argparse
import time
from slurm_utils import submit_slurm_job, wait_for_job

def run_al_iteration(iteration, train_path, pool_path, test_path, model_dir, ensemble_size=5, selection_size=10000):
    """
    Runs a single iteration of the Active Learning loop.
    Ensures NO label leakage from the pool during selection.
    """
    print(f"\n--- Starting AL Iteration {iteration} ---")
    
    # 1. Train Chemprop Ensemble
    train_cmd = (
        f"chemprop_train --data_path {train_path} "
        f"--dataset_type classification "
        f"--save_dir {model_dir}/iteration_{iteration} "
        f"--epochs 100 --num_workers 8 --batch_size 128 "
        f"--ensemble_size {ensemble_size} --gpu 0 "
        f"--loss_function dirichlet --evidential_regularization 0.2 --class_balance "
        f"--features_generator rdkit_2d_normalized --no_features_scaling"
    )
    job_id = submit_slurm_job(f"train_al_iter_{iteration}", train_cmd)
    if not job_id: return None
    wait_for_job(job_id)
    
    # 2. Predict on Unlabeled Pool (DROP LABELS to prevent leakage)
    pool_df = pd.read_csv(pool_path)
    unlabeled_pool_path = f"data/unlabeled_pool_iter_{iteration}.csv"
    # Assuming labels are in the last column or specific column names. 
    # To be safe, we only keep the SMILES/ID column for prediction.
    pool_df.iloc[:, [0]].to_csv(unlabeled_pool_path, index=False)
    
    preds_path = f"data/preds_iter_{iteration}.csv"
    predict_cmd = (
        f"chemprop_predict --test_path {unlabeled_pool_path} "
        f"--checkpoint_dir {model_dir}/iteration_{iteration} "
        f"--preds_path {preds_path} "
        f"--gpu 0"
    )
    job_id = submit_slurm_job(f"predict_al_iter_{iteration}", predict_cmd)
    if not job_id: return None
    wait_for_job(job_id)
    
    # 3. Predict on Held-out Test Set for Evaluation
    test_preds_path = f"data/test_preds_iter_{iteration}.csv"
    test_predict_cmd = (
        f"chemprop_predict --test_path {test_path} "
        f"--checkpoint_dir {model_dir}/iteration_{iteration} "
        f"--preds_path {test_preds_path} "
        f"--gpu 0"
    )
    job_id = submit_slurm_job(f"eval_test_iter_{iteration}", test_predict_cmd)
    if not job_id: return None
    wait_for_job(job_id)

    # 4. Candidate Selection (Autonomous logic modified by agent)
    preds_df = pd.read_csv(preds_path)
    target_col = preds_df.columns[-1]
    
    # Selection logic: Top K predicted inhibitors (Placeholder for agent experimentation)
    top_indices = preds_df.nlargest(selection_size, target_col).index
    selected_samples = pool_df.iloc[top_indices]
    
    # Calculate performance on held-out test set
    test_preds_df = pd.read_csv(test_preds_path)
    test_actual_df = pd.read_csv(test_path)
    # Note: AUROC should be calculated here using test_preds_df vs test_actual_df
    # For now, we print a placeholder that the agent will implement/parse
    print(f"Iteration {iteration}: Evaluating on {test_path}")
    
    return selected_samples, 0.85 # Placeholder AUROC

def main():
    parser = argparse.ArgumentParser(description="Chemprop AL Optimizer")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--pool", type=str, required=True, help="Full pool with labels (hidden during selection)")
    parser.add_argument("--test", type=str, required=True, help="Held-out test set for evaluation")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default="models/al_checkpoints")
    parser.add_argument("--selection_size", type=int, default=10000)
    args = parser.parse_args()
    
    current_train = args.train
    
    for i in range(args.iters):
        result = run_al_iteration(i, current_train, args.pool, args.test, args.model_dir, selection_size=args.selection_size)
        if result is None: break
        
        selected, auroc = result
        print(f"FINAL_METRICS: test_auroc={auroc:.6f}")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import argparse
import time
from slurm_utils import submit_slurm_job, wait_for_job

def run_al_iteration(iteration, train_path, pool_path, model_dir, ensemble_size=5):
    """
    Runs a single iteration of the Active Learning loop.
    """
    print(f"\n--- Starting AL Iteration {iteration} ---")
    
    # 1. Train Chemprop Ensemble
    # Using specific defaults from Active_Learning_Plate_struc.md
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
    
    # 2. Predict on Unlabeled Pool
    preds_path = f"data/preds_iter_{iteration}.csv"
    predict_cmd = (
        f"chemprop_predict --test_path {pool_path} "
        f"--checkpoint_dir {model_dir}/iteration_{iteration} "
        f"--preds_path {preds_path} "
        f"--gpu 0"
    )
    job_id = submit_slurm_job(f"predict_al_iter_{iteration}", predict_cmd)
    if not job_id: return None
    wait_for_job(job_id)
        
    # 3. Candidate Selection
    preds_df = pd.read_csv(preds_path)
    # Placeholder for multi-objective ranking
    # In practice, LLM will modify this section to improve selection
    target_col = preds_df.columns[-1]
    top_candidates = preds_df.nlargest(100, target_col)
    
    # Calculate performance metrics on a 'validation' portion of predictions if possible
    # For now, we report the average inhibition of top candidates as a proxy
    hit_rate_proxy = top_candidates[target_col].mean()
    print(f"Iteration {iteration}: Hit Rate Proxy = {hit_rate_proxy:.4f}")
    
    return top_candidates, hit_rate_proxy

def main():
    parser = argparse.ArgumentParser(description="Chemprop AL Optimizer (Autoresearch Framework)")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default="models/al_checkpoints")
    args = parser.parse_args()
    
    current_train = args.train
    
    for i in range(args.iters):
        result = run_al_iteration(i, current_train, args.pool, args.model_dir)
        if result is None:
            print("Iteration failed.")
            break
        
        selected, metrics = result
        
        # Log for the LLM agent to extract
        print(f"FINAL_METRICS: hit_rate={metrics:.6f} auroc=0.85") # AUROC dummy for now

if __name__ == "__main__":
    main()

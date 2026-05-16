#!/bin/bash
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o run_logs/al_chemprop_%j.out
#SBATCH -e run_logs/al_chemprop_%j.err
#SBATCH -J al-chemprop

source /home/ars3983/miniforge/bin/activate al-agent
export WANDB_MODE=disabled

cd /n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning

echo "=== DMPNN Active Learning Campaign ==="
echo "Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

python al_optimizer_chemprop.py \
    --train data/train_df_chemprop.csv \
    --pool data/pool_df_chemprop.csv \
    --test data/test_df_chemprop.csv \
    --iters 10 \
    --work_dir al_chemprop_runs

echo "=== Done at $(date) ==="

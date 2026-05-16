#!/bin/bash
#SBATCH -c 2
#SBATCH -t 1-00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/dmpnn_train_%j.out
#SBATCH -e run_logs/dmpnn_train_%j.err

source /home/ars3983/miniforge/bin/activate al-agent
export WANDB_MODE=disabled

CHEMPROP_DIR=/n/data1/hms/dbmi/farhat/aryan/AL/narrow_lyme_antibiotic/active_learning/run/chemprop
cd "$CHEMPROP_DIR"

echo "=== Training DMPNN Model ==="
echo "Data: $1"
echo "Save dir: $2"
echo "Started at $(date)"

mkdir -p "$2"
mkdir -p run_logs

python train.py \
    --data_path "$1" \
    --smiles_columns 'SMILES' \
    --target_columns 'Y' \
    --dataset_type 'classification' \
    --save_dir "$2" \
    --split_type 'scaffold_balanced' \
    --num_folds '5' \
    --metric 'auc' \
    --extra_metrics 'binary_cross_entropy' 'prc-auc' \
    --loss_function 'dirichlet' \
    --evidential_regularization 0.2 \
    --class_balance \
    --epochs 200 \
    --quiet

echo "=== Done at $(date) ==="

#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-04:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o run_logs/dmpnn_predict_%j.out
#SBATCH -e run_logs/dmpnn_predict_%j.err

source /home/ars3983/miniforge/bin/activate al-agent

CHEMPROP_DIR=/n/data1/hms/dbmi/farhat/aryan/AL/narrow_lyme_antibiotic/active_learning/run/chemprop
cd "$CHEMPROP_DIR"

echo "=== Predicting with DMPNN Model ==="
echo "Test data: $1"
echo "Checkpoint dir: $2"
echo "Output: $3"
echo "Started at $(date)"

mkdir -p "$(dirname "$3")"
mkdir -p run_logs

python predict.py \
    --test_path "$1" \
    --smiles_columns 'SMILES' \
    --checkpoint_dir "$2" \
    --preds_path "$3" \
    --uncertainty_method 'dirichlet'

echo "=== Done at $(date) ==="

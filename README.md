# DMPNN Active Learning Agent

An autonomous active learning pipeline for molecular property prediction, designed for the **HMS O2** cluster. An LLM agent (DeepSeek via [opencode](https://github.com/sstcloud/opencode)) iteratively refines candidate selection strategies, trains a Chemprop DMPNN ensemble, and selects molecules from an unlabeled pool.

## Overview

The Active Learning Agent automates molecular hit discovery through an iterative loop:

1. **Train** a Chemprop DMPNN ensemble on the current labeled dataset with Dirichlet evidential loss.
2. **Predict** activity and uncertainty on a pool of unlabeled candidates.
3. **Select** the most informative candidates using a multi-objective acquisition function (inhibition, uncertainty, novelty, diversity).
4. **Augment** the training set with newly labeled molecules (labels revealed only after selection).
5. **Iterate** until the discovery goals or labeling budget are met.

## Architecture

| Component | Implementation |
|-----------|---------------|
| **Molecular encoder** | DMPNN (Directed Message-Passing Neural Network, 3 layers, hidden=300) |
| **Predictive model** | Chemprop ensemble (5-fold scaffold-balanced CV, Dirichlet head, evidential regularization 0.2) |
| **Agent / orchestrator** | [opencode](https://github.com/sstcloud/opencode) + DeepSeek API |
| **Compute** | HMS O2 GPU nodes (`gpu` or `gpu_quad` partition, 1 GPU, 32-64 GB RAM) |
| **Acquisition** | Weighted sum of predicted inhibition + Dirichlet uncertainty + novelty + diversity |
| **Uncertainty** | Evidential Dirichlet (built-in, no ensemble disagreement needed) |

## Two Model Options

### 1. DMPNN (Chemprop) — Recommended
Higher accuracy, GPU-accelerated, Dirichlet uncertainty built-in.

```bash
python al_optimizer_chemprop.py \
    --train data/train_df_chemprop.csv \
    --pool data/pool_df_chemprop.csv \
    --test data/test_df_chemprop.csv \
    --iters 10
```

### 2. MiniMol + FFN — Fast Baseline
CPU-friendly, Morgan fingerprint-based, ensemble disagreement uncertainty.

```bash
python al_optimizer.py \
    --train data/train_df.csv \
    --pool data/pool_df.csv \
    --test data/test_df.csv \
    --iters 10
```

## Environment Setup

### Prerequisites
- Access to HMS O2 cluster.
- Conda environment `al-agent` with PyTorch, RDKit, scikit-learn, pandas, numpy.
- The bundled Chemprop v1.7.0 code at the path set by `$CHEMPROP_DIR`.
- [opencode](https://github.com/sstcloud/opencode) installed and configured with a DeepSeek API key.

### Activation
On an O2 login node, activate the environment:
```bash
source /home/ars3983/miniforge/bin/activate al-agent
export WANDB_MODE=disabled
```

### GPU Node (for DMPNN training)
```bash
srun --pty -c 4 --mem 64G -t 0-12:00 -p gpu --gres=gpu:1 /bin/bash
source /home/ars3983/miniforge/bin/activate al-agent
```

## Quick Start

### 1. Launch the autonomous research loop (MiniMol)
From the project root on an O2 compute node, start opencode:
```bash
opencode
```
Paste the contents of `program.md` into opencode.

### 2. Launch DMPNN AL campaign
```bash
# Submit as SLURM job
sbatch scripts/run_al_chemprop.sh

# Or run interactively on a GPU node
python al_optimizer_chemprop.py \
    --train data/train_df_chemprop.csv \
    --pool data/pool_df_chemprop.csv \
    --test data/test_df_chemprop.csv \
    --iters 10
```

### 3. Single training run (DMPNN)
```bash
sbatch scripts/train_dmpnn.sh data/train_df_chemprop.csv models/run1/
```

## Project Structure

```
.
├── al_optimizer.py              # MiniMol+FFN active learning loop
├── al_optimizer_chemprop.py     # DMPNN/Chemprop active learning loop (NEW)
├── minimol.py                   # MiniMol encoder (Morgan fingerprint -> 512-dim)
├── program.md                   # Autonomous agent instructions (MiniMol)
├── program_chemprop.md          # Autonomous agent instructions (DMPNN)
├── code/
│   ├── minimol_ffn.py           # FFN model definition + trainer (read-only)
│   └── minimol_ffn.sh           # Standalone MiniMol training script
├── scripts/
│   ├── train_dmpnn.sh           # SLURM GPU training script (DMPNN)
│   ├── predict_dmpnn.sh         # SLURM GPU prediction script (DMPNN)
│   └── run_al_chemprop.sh       # SLURM full AL campaign (DMPNN)
├── data/
│   ├── train_df.csv             # Labeled training set (SMILES, Y) — MiniMol
│   ├── pool_df.csv              # Unlabeled candidate pool (SMILES, Y — hidden)
│   ├── test_df.csv              # Held-out test set (SMILES, Y)
│   ├── train_df_chemprop.csv    # Same, formatted for Chemprop (SMILES,Y)
│   ├── pool_df_chemprop.csv     # Pool for Chemprop
│   └── test_df_chemprop.csv     # Test for Chemprop
├── skills/
│   └── chemprop-al-optimizer.md # Alternative skill-based agent definition
├── results.tsv                  # Experiment log
├── notebooks/
│   └── create_splits.ipynb
└── docs/
```

## Active Learning Strategy

### Problem Setup
- **Pool**: ~96,000 unlabeled molecules.
- **Budget**: 10 iterations, selecting 1,000 molecules per iteration.
- **Evaluation**: AUROC / AUPRC on a fixed held-out test set (~107K molecules).
- **Constraint**: No label leakage — pool labels are only revealed after a molecule is selected.

### Acquisition Function
The agent can tune weights for:
- **Inhibition** — predicted probability of activity (from DMPNN).
- **Uncertainty** — Dirichlet evidential uncertainty (built into Chemprop).
- **Novelty** — dissimilarity to already-labeled training molecules (1 - max Tanimoto).
- **Diversity** — internal chemical diversity of the selected batch (1 - mean pairwise Tanimoto).

Default: `score = 0.5 * inhibition + 1.0 * uncertainty + 0.3 * novelty + 0.2 * diversity`

### Agent Autonomy
The opencode + DeepSeek agent operates on its own initiative:
- Modifies acquisition weights and hyperparameters in `al_optimizer_chemprop.py`.
- Commits each change with a descriptive message before running.
- Reads `al_run.log` to extract metrics and updates `results.tsv`.
- Decides next strategy based on trend analysis across iterations.
- All experimental history is preserved in git — no `git reset`.

### Key Differences: MiniMol vs DMPNN
| | MiniMol+FFN | DMPNN (Chemprop) |
|---|---|---|
| Encoder | Morgan FP (2048-bit) | Learned graph embeddings |
| Model | 3×FFN ensemble | 5-fold DMPNN ensemble |
| Uncertainty | Ensemble std | Evidential Dirichlet |
| Training time | ~2-5 min (CPU) | ~30-60 min (GPU) |
| AUROC (baseline) | ~0.60 | ~0.70+ |
| GPU required | No | Yes |

---
*Developed for the Farhat Lab, HMS DBMI.*

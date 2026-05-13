# Agent-Guided Active Learning for Antibiotic Discovery

An autonomous active learning pipeline where an LLM agent (DeepSeek via [opencode](https://github.com/sstcloud/opencode)) iteratively designs and executes molecular selection strategies to maximize antibiotic discovery from large chemical libraries.

## Overview

1. **Train** a MiniMol + FFN ensemble on the current labeled dataset
2. **Predict** activity and uncertainty on a pool of unlabeled candidates
3. **Select** the most informative candidates using a multi-objective acquisition function (inhibition, uncertainty, diversity)
4. **Augment** the training set with newly labeled molecules
5. **Iterate** — the LLM agent adapts the acquisition strategy based on results

## Architecture

| Component | Implementation |
|-----------|---------------|
| Molecular encoder | MiniMol — Morgan fingerprints (2048-bit, radius 2 → 512-dim) |
| Predictive model | FFN ensemble (3 members, 2×512 hidden, batch norm, dropout 0.1) |
| Agent / orchestrator | [opencode](https://github.com/sstcloud/opencode) + DeepSeek API |
| Acquisition | Weighted sum: inhibition probability + ensemble uncertainty |

## Results

A 10-iteration campaign (May 2026) on M. tuberculosis HTS data (~107K compounds):

| Iteration | test_auroc | Strategy |
|-----------|-----------|----------|
| 0 (baseline) | 0.596 | W_INH=1.0, W_UNC=1.0 |
| 4 | 0.629 | W_INH=0.7 |
| 7 | 0.640 | W_INH=0.5 |
| 10 (final) | **0.666** | W_INH=0.5, W_UNC=1.0 |

The agent autonomously explored exploitation vs. exploration tradeoffs, recovered from regression, and converged on a hybrid strategy (W_INH=0.5, W_UNC=1.0). Full results in [`results.tsv`](results.tsv) and the [`al/may13`](https://github.com/aryanshaik/agent_active_learning/tree/al/may13) branch.

## Setup

### Prerequisites
- HMS O2 cluster access
- Conda environment with PyTorch, RDKit, scikit-learn, pandas, numpy
- [opencode](https://github.com/sstcloud/opencode) with DeepSeek API key
- Git SSH key configured for GitHub

### Environment
```bash
# Create and activate conda environment
conda create -n al-agent python=3.11
conda activate al-agent
pip install -r requirements.txt
```

### Launch
```bash
# On O2: request compute node
srun -p interactive -c 10 --mem 64G --time 12:00:00 --pty bash

# Start persistent session
tmux new -s al-agent

# Setup
source /home/ars3983/miniforge/bin/activate al-agent
cd /n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning

# Launch the agent
opencode
# Paste program.md as the research directive
```

### Manual run (without agent)
```bash
python -u al_optimizer.py --iters 1 --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv 2>&1 | tee al_run.log
```

## Project Structure

```
.
├── al_optimizer.py       # Active learning loop (modifiable by agent)
├── minimol.py            # MiniMol encoder (Morgan fingerprint → 512-dim)
├── program.md            # Agent instructions (fed to opencode)
├── code/
│   └── minimol_ffn.py    # FFN model + trainer (read-only)
├── data/
│   ├── train_df.csv      # Initial labeled set (~10K SMILES)
│   ├── pool_df.csv       # Unlabeled candidate pool (~97K SMILES)
│   └── test_df.csv       # Held-out test set (~108K SMILES)
├── results.tsv           # Experiment log
├── requirements.txt      # Python dependencies
└── notebooks/
    └── create_splits.ipynb
```

## Agent Autonomy

The opencode + DeepSeek agent operates on its own initiative:
- Modifies acquisition weights, ensemble size, and training hyperparameters in `al_optimizer.py`
- Commits each change before running
- Runs the pipeline, parses metrics from `FINAL_METRICS` in `al_run.log`
- Logs results to `results.tsv`
- Adapts strategy based on trend analysis
- All history preserved in git — no `git reset`

## Constraints
- No label leakage — pool labels hidden during selection
- `code/minimol_ffn.py` is read-only; only `al_optimizer.py` is modified
- Every run is a data point — keep all commits

---
*Farhat Lab, HMS DBMI*

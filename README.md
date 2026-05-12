# MiniMol Active Learning Agent

An autonomous active learning pipeline for molecular property prediction, designed for the **HMS O2** cluster. An LLM agent (DeepSeek via [opencode](https://github.com/sstcloud/opencode)) iteratively refines candidate selection strategies, trains an ensemble of MiniMol-based feedforward networks, and selects molecules from an unlabeled pool for labeling.

## Overview

The Active Learning Agent automates molecular hit discovery through an iterative loop:

1. **Train** a MiniMol + FFN ensemble on the current labeled dataset.
2. **Predict** activity and uncertainty on a pool of unlabeled candidates.
3. **Select** the most informative candidates using a multi-objective acquisition function (inhibition, uncertainty, novelty, diversity).
4. **Augment** the training set with newly labeled molecules (labels revealed only after selection).
5. **Iterate** until the discovery goals or labeling budget are met.

## Architecture

| Component | Implementation |
|-----------|---------------|
| **Molecular encoder** | MiniMol — Morgan fingerprints (2048-bit, radius 2) |
| **Predictive model** | FFN ensemble (3 members, 2 hidden layers × 512 dim, batch norm, dropout 0.1) |
| **Agent / orchestrator** | [opencode](https://github.com/sstcloud/opencode) + DeepSeek API |
| **Compute** | HMS O2 interactive CPU nodes (4-8 cores, 64 GB RAM, 12-24 hr) |
| **Acquisition** | Weighted sum of predicted inhibition + epistemic uncertainty |

## Environment Setup

### Prerequisites
- Access to HMS O2 cluster.
- Conda environment with PyTorch, RDKit, scikit-learn, pandas, numpy.
- [opencode](https://github.com/sstcloud/opencode) installed and configured with a DeepSeek API key.

### Activation
On an O2 login node, request an interactive compute node:
```bash
srun --pty -c 8 --mem 64G -t 0-12:00 -p interactive /bin/bash
```

Then activate the environment:
```bash
source /home/ars3983/miniforge/bin/activate al-agent
```

## Quick Start

### 1. Launch the autonomous research loop
From the project root on an O2 compute node, start opencode:
```bash
opencode
```

### 2. Issue the research directive
Paste the contents of `program.md` into opencode. The agent will:
- Read the current codebase (`al_optimizer.py`, `code/minimol_ffn.py`)
- Propose a run tag and modify `al_optimizer.py` with a candidate selection strategy
- Run `python al_optimizer.py --iters 1 --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv`
- Extract metrics (AUROC, AUPRC, hit rate, novelty, diversity) from the output
- Log results to `results.tsv`
- Iterate, adapting the acquisition function based on prior results

### 3. Manual single iteration (without the agent)
```bash
python al_optimizer.py --iters 1 --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv
```

## Project Structure

```
.
├── al_optimizer.py       # Active learning loop: training, prediction, acquisition, metrics
├── minimol.py            # MiniMol encoder (Morgan fingerprint → 512-dim vector)
├── program.md            # Autonomous agent instructions (fed to opencode)
├── code/
│   ├── minimol_ffn.py    # FFN model definition + trainer (read-only, fixed implementation)
│   └── minimol_ffn.sh    # Standalone training script
├── data/
│   ├── train_df.csv      # Labeled training set (SMILES, Y)
│   ├── pool_df.csv       # Unlabeled candidate pool (SMILES, Y — labels hidden from agent)
│   └── test_df.csv       # Held-out test set (SMILES, Y)
├── skills/
│   └── chemprop-al-optimizer.md   # Alternative skill-based agent definition
├── results.tsv           # Experiment log (commit, test_auroc, hit_rate, status, description)
├── notebooks/
│   └── create_splits.ipynb
└── docs/
```

## Active Learning Strategy

### Problem Setup
- **Pool**: ~100,000 unlabeled molecules.
- **Budget**: 10 iterations, selecting 1,000 molecules per iteration.
- **Evaluation**: AUROC / AUPRC on a fixed held-out test set.
- **Constraint**: No label leakage — pool labels are only revealed after a molecule is selected.

### Acquisition Function
The agent can tune weights for:
- **Inhibition** — predicted probability of activity.
- **Uncertainty** — standard deviation across the 3-member ensemble.
- **Diversity** — internal chemical diversity of the selected batch (Tanimoto distance).
- **Novelty** — dissimilarity to already-labeled training molecules.

Default: `score = 1.0 × inhibition + 1.0 × uncertainty`

### Agent Autonomy
The opencode + DeepSeek agent operates on its own initiative:
- Modifies acquisition weights and selection size in `al_optimizer.py`.
- Commits each change with a descriptive message before running.
- Reads `al_run.log` to extract metrics and updates `results.tsv`.
- Decides next strategy based on trend analysis across iterations.
- All experimental history is preserved in git — no `git reset`.

---
*Developed for the Farhat Lab, HMS DBMI.*

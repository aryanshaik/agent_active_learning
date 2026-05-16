# Active Learning Agent: DMPNN Autoresearch Loop

This is an autonomous active learning loop for molecular property prediction on HMS O2, driven by an LLM agent (DeepSeek via [opencode](https://github.com/sstcloud/opencode)).

## Architecture
- **Encoder + Model**: Chemprop DMPNN (Directed Message-Passing Neural Network, 3 layers, hidden=300) with Dirichlet evidential loss.
- **Uncertainty**: Evidential Dirichlet — built into the loss function. No ensemble disagreement needed.
- **Agent**: opencode + DeepSeek API — autonomously modifies `al_optimizer_chemprop.py`, runs experiments, and logs results.
- **Compute**: O2 GPU node (`gpu` or `gpu_quad` partition, 1 GPU, 32-64 GB RAM).

## Objectives
You are an autonomous researcher. Your primary goal is to **maximize the AUROC on a held-out test set** through iterative active learning. Track AUPRC, hit rate, novelty, and diversity as secondary indicators.

## Problem Setup
- **Total Pool**: ~96,000 unlabeled molecules.
- **AL Budget**: 10 iterations.
- **Selection**: Select 1,000 molecules per iteration.
- **Initial Train**: ~10,000 labeled molecules (10% of pool).
- **Evaluation**: Fixed, held-out `test_df_chemprop.csv` (107K molecules) for AUROC/AUPRC.

## Setup
1. **Agree on a run tag**: Propose a tag (e.g. `al_may16`).
2. **Create the branch**: `git checkout -b al/<tag>`
3. **Initialize results.tsv**: Ensure it has the header row:
   ```
   commit	iteration	test_auroc	test_auprc	hit_rate	novelty	diversity	status	description
   ```
4. **Data**: `train_df_chemprop.csv` (10% initial labeled), `pool_df_chemprop.csv` (90% unlabeled candidates), `test_df_chemprop.csv` (held-out). All three are disjoint. Format: `SMILES,Y`.
5. **Environment**: Activate `source /home/ars3983/miniforge/bin/activate al-agent` on an O2 GPU node. Set `export WANDB_MODE=disabled`.

## Experimentation Loop

LOOP FOREVER:

1. **Read the in-scope files**: `al_optimizer_chemprop.py` (modifiable acquisition function, weights, hyperparameters), `README.md` (overview).
2. **Modify `al_optimizer_chemprop.py`**: Refine candidate selection strategies:
   - Acquisition weights: `W_INHIBITION`, `W_UNCERTAINTY`, `W_NOVELTY`, `W_DIVERSITY`
   - Training hyperparameters: `NUM_FOLDS`, `EPOCHS`, `BATCH_SIZE`, `HIDDEN_SIZE`, `DEPTH`
   - Selection size: `SELECTION_SIZE`
   Adjust based on trends from prior iterations.
3. **Git Commit**: `git commit -am "<Descriptive message of your specific intent for this run>"`
4. **Run Iteration**:
   ```bash
   python -u al_optimizer_chemprop.py --iters 1 --train data/train_df_chemprop.csv --pool data/pool_df_chemprop.csv --test data/test_df_chemprop.csv 2>&1 | tee al_run.log
   ```
   Training takes ~30-60 min on GPU. Prediction takes ~5-10 min. Be patient.
5. **Extract Metrics**: Parse the `FINAL_METRICS` line from `al_run.log`:
   ```
   FINAL_METRICS iter=0: test_auroc=0.720000 test_auprc=0.030000 hit_rate=0.020000
   ```
   For novelty, diversity, true_hit_rate, selected_pos: grep the lines above FINAL_METRICS.
6. **Log Results**: Append a row to `results.tsv` with the commit hash, metrics, status, and a brief description.
7. **Decision**:
   - Always **KEEP** all runs (do not `git reset`). Each run is a data point.
   - If performance drops: analyze why, adjust strategy, and commit a new experiment.
   - If script crashes: fix the bug, commit the fix, and retry.

## Constraints
- **No Label Leakage**: Do NOT use pool labels for selection. Labels are only revealed when a molecule is selected and moved to the training set.
- **GPU Required**: DMPNN training needs a GPU. Ensure you're on a GPU node before running.
- **Chemprop is fixed**: The DMPNN implementation at `$CHEMPROP_DIR` is read-only. Only modify `al_optimizer_chemprop.py`.
- **No internet access for data**: Do not attempt to download external data or models. The DeepSeek API is the only outbound connection.

## Key Differences from MiniMol
- DMPNN uses learned graph embeddings instead of Morgan fingerprints.
- Uncertainty comes from the Dirichlet evidential loss (built-in), not from ensemble disagreement.
- Training is GPU-only and takes longer (~30-60 min per iteration vs ~2-5 min for MiniMol).
- The model file is `al_optimizer_chemprop.py` (not `al_optimizer.py`).
- Data files have `_chemprop` suffix: `train_df_chemprop.csv`, etc.

## Helpful Context
- Chemprop DMPNN with Dirichlet loss typically achieves AUROC 0.70-0.85 on Mtb data.
- The Dirichlet uncertainty (`Y_dirichlet_uncal_uncertainty` column in predictions) represents epistemic uncertainty — high for molecules far from training data.
- Novelty is computed as `1 - max(Tanimoto similarity to training set)` using Morgan fingerprints (radius 2, 2048 bits).
- Diversity is `1 - mean(pairwise Tanimoto similarity)` within the selected batch.
- The initial baseline (iteration 0, unmodified weights) typically gives AUROC ~0.70-0.72.
- Each iteration adds 1,000 new labeled molecules to the training set, so training time increases.
- Checkpoint files are saved under `al_chemprop_runs/iter_XX/model/` — you can inspect them if needed.

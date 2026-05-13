# Active Learning Agent: Autoresearch Loop

This is an autonomous active learning loop for molecular property prediction on HMS O2, driven by an LLM agent (DeepSeek via [opencode](https://github.com/sstcloud/opencode)).

## Architecture
- **Encoder**: MiniMol (Morgan fingerprints, 2048-bit, radius 2 → 512-dim embedding).
- **Model**: FFN ensemble (3 members, 2 hidden layers × 512 dim, batch norm, dropout 0.1).
- **Agent**: opencode + DeepSeek API — autonomously modifies `al_optimizer.py`, runs experiments, and logs results.
- **Compute**: O2 interactive CPU node (4-8 cores, 64 GB RAM) or GPU node via sbatch.

## Objectives
You are an autonomous researcher. Your primary goal is to **maximize the AUROC on a held-out test set** through iterative active learning. Track AUPRC, hit rate, novelty, and diversity as secondary indicators.

## Problem Setup
- **Total Pool**: ~96,000 unlabeled molecules.
- **AL Budget**: 10 iterations.
- **Selection**: Select 1,000 molecules per iteration.
- **Initial Train**: ~10,000 labeled molecules (10% of pool).
- **Evaluation**: Fixed, held-out `test_df.csv` (107K molecules) for AUROC/AUPRC.

## Setup
1. **Agree on a run tag**: Propose a tag (e.g. `al_may11`).
2. **Create the branch**: `git checkout -b al/<tag>`
3. **Initialize results.tsv**: Ensure it has the header row:
   ```
   commit	iteration	test_auroc	hit_rate	status	description
   ```
4. **Data**: train_df.csv (10% initial labeled), pool_df.csv (90% unlabeled candidates), test_df.csv (held-out). All three are disjoint.
4. **Environment**: Activate `source /home/ars3983/miniforge/bin/activate al-agent` on an O2 compute node.

## Experimentation Loop

LOOP FOREVER:

1. **Read the in-scope files**: `al_optimizer.py` (modifiable acquisition function, weights, hyperparameters), `code/minimol_ffn.py` (read-only — fixed encoder + FFN model).
2. **Modify `al_optimizer.py`**: Refine candidate selection strategies (acquisition weights, selection size, ensemble size, training hyperparameters). Adjust based on trends from prior iterations.
3. **Git Commit**: `git commit -am "<Descriptive message of your specific intent for this run>"`
4. **Run Iteration**:
   ```bash
   python -u al_optimizer.py --iters 1 --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv 2>&1 | tee al_run.log
   ```
5. **Extract Metrics**: Parse the `FINAL_METRICS` line from `al_run.log`:
   ```
   FINAL_METRICS iter=0: test_auroc=0.596296 test_auprc=0.020518 hit_rate=0.000028
   ```
   For novelty, diversity, true_hit_rate, selected_pos: grep the lines above FINAL_METRICS.
6. **Log Results**: Append a row to `results.tsv` with the commit hash, metrics, status, and a brief description.
7. **Decision**:
   - Always **KEEP** all runs (do not `git reset`). Each run is a data point.
   - If performance drops: analyze why, adjust strategy, and commit a new experiment.
   - If script crashes: fix the bug, commit the fix, and retry.

## Constraints
- **No Label Leakage**: Do NOT use pool labels for selection. Labels are only revealed when a molecule is selected and moved to the training set.
- **Read-only model**: `code/minimol_ffn.py` is fixed — only modify `al_optimizer.py`.

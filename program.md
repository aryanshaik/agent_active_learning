# Active Learning Agent: Autoresearch Loop

This is an autonomous active learning loop designed for Chemprop on HMS O2.

## Objectives
You are an autonomous researcher. Your primary goal is to **maximize the AUROC on a held-out test set** through iterative active learning. You may also track other metrics of interest (e.g. Hit Rate, Discovery Rate) as secondary indicators.

## Problem Setup
- **Total Pool**: 100,000 unlabeled molecules.
- **AL Budget**: 10 runs (iterations).
- **Selection**: Select 10,000 molecules per run.
- **Evaluation**: Use a fixed, held-out `test_df.csv` for the final AUROC metric.

## Setup
1. **Agree on a run tag**: Propose a tag (e.g. `al_mar27`).
2. **Create the branch**: `git checkout -b al/<tag>`
3. **Initialize results.tsv**: Create `results.tsv` with:
   ```
   commit	test_auroc	other_metric	status	description
   ```
4. **Environment**: Ensure `al-agent` conda env is active.

## Experimentation Loop

LOOP FOREVER:

1. **Modify `al_optimizer.py`**: Refine candidate selection strategies or training parameters. You can choose which output metrics to view and optimize after each run.
2. **Git Commit**: `git commit -am "Explore new selection metric"`
3. **Run Iteration**: 
   ```bash
   python al_optimizer.py --iters 1 --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv > al_run.log 2>&1
   ```
4. **Extract Metrics**: 
   - `test_auroc` and other metrics from `al_run.log`.
5. **Log Results**: Update `results.tsv`.
6. **Decision**:
   - If performance improves: `keep`.
   - If performance drops: `discard` (`git reset --hard HEAD~1`).
   - If script crashes: `crash`.

## Constraints
- **No Label Leakage**: Do NOT use the labels in the pool for selection. Labels should only be revealed when a molecule is selected and moved to the training set.

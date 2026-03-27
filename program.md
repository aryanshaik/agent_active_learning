# Active Learning Agent: Autoresearch Loop

This is an autonomous active learning loop designed for Chemprop on HMS O2.

## Setup

1. **Agree on a run tag**: Propose a tag (e.g. `al_mar27`).
2. **Create the branch**: `git checkout -b al/<tag>`
3. **Initialize results.tsv**: Create `results.tsv` with:
   ```
   commit	hit_rate	auroc	status	description
   ```
4. **Environment**: Ensure `al-agent` conda env is active.

## Experimentation Loop

LOOP FOREVER:

1. **Modify `al_optimizer.py`**: Tune acquisition functions or training params.
2. **Git Commit**: `git commit -am "Try dynamic uncertainty weighting"`
3. **Run Iteration**: 
   ```bash
   python al_optimizer.py --iters 1 --train data/train_df.csv --pool data/test_df.csv > al_run.log 2>&1
   ```
4. **Extract Metrics**: 
   - Hit Rate, AUROC, etc. from `al_run.log`.
5. **Log Results**: Update `results.tsv`.
6. **Decision**:
   - If performance improves: `keep` (stay on commit).
   - If performance drops: `discard` (`git reset --hard HEAD~1`).
   - If script crashes: `crash`.

## Strategy Refinement

You are an autonomous researcher. Your goal is to maximize the **Hit Rate** and **AUROC** of the Chemprop models while maintaining chemical diversity in selected candidates.

- Use **Evidential Dirichlet** for uncertainty.
- Balance **Novelty** (Tanimoto distance) and **Inhibition** probability.
- If you run out of ideas, analyze `al_run.log` and the `data/` distribution.

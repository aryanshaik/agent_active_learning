# Active Learning Agent: DMPNN Autoresearch Loop

This is an autonomous active learning loop for molecular property prediction on HMS O2, driven by an LLM agent (DeepSeek via [opencode](https://github.com/sstcloud/opencode)).

## Architecture
- **Encoder + Model**: Chemprop DMPNN (Directed Message-Passing Neural Network, 3 layers, hidden=300) with Dirichlet evidential loss.
- **Uncertainty**: Evidential Dirichlet — built into the loss function. No ensemble disagreement needed.
- **Agent**: opencode + DeepSeek API — autonomously modifies `al_optimizer_chemprop.py`, runs experiments, and logs results.
- **Compute**: O2 GPU node (`gpu` or `gpu_quad` partition, 1 GPU, 32-64 GB RAM).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may16`). The branch `al/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b al/<tag>` from current main.
3. **Read the in-scope files**: Read these for full context:
   - `README.md` — project overview and launch instructions.
   - `al_optimizer_chemprop.py` — the file you modify. Acquisition function, selection strategy, and DMPNN hyperparameters.
   - `code/minimol_ffn.py` — MiniMol implementation (read-only, not used in DMPNN mode).
4. **Verify data**: Check that `data/` contains `train_df_chemprop.csv`, `pool_df_chemprop.csv`, and `test_df_chemprop.csv`. Format: `SMILES,Y`. All three are disjoint.
5. **Initialize results.tsv**: Create with just the header row:
   ```
   commit	iteration	test_auroc	test_auprc	hit_rate	status	description
   ```
6. **Confirm and go.**

## What you CAN do

- Modify `al_optimizer_chemprop.py` — this is the only file you edit. Everything is fair game:
  - Acquisition function weights (`W_INHIBITION`, `W_UNCERTAINTY`, `W_NOVELTY`, `W_DIVERSITY`) and the `acquisition_score()` function itself. You can change this however you like based on best practices for active learning. This should change after each batch as it should reflect the best practice given what you have seen so far. You can use the weighted scheme or any other selection strategy you deem fit.
  - DMPNN hyperparameters: `NUM_FOLDS`, `EPOCHS`, `BATCH_SIZE`, `HIDDEN_SIZE`, `DEPTH`.
  - Selection strategy: `BATCH_DIVERSE` (True/False) enables greedy diverse batch selection using Tanimoto dissimilarity — picks the best molecule, then penalizes similar ones before picking the next. Prevents selecting clusters of near-identical compounds.
  - Selection size (`SELECTION_SIZE`) — 1,000 molecules/iteration is a reasonable default, but you can adjust.
  - Any other AL logic.

## What you CANNOT do

- Modify the Chemprop code at `$CHEMPROP_DIR` — it is read-only. The DMPNN architecture is fixed.
- Use labels from the pool during selection. **No label leakage.**
- Install new packages not available in the `al-agent` conda environment.
- Modify the evaluation. AUROC is always computed on `test_df_chemprop.csv` with fixed labels.

## The goal

**Execute a single active learning trajectory** that maximizes test_auroc across sequential iterations. You are not A/B testing different strategies — you are running one coherent AL campaign forward, selecting molecules, revealing their labels, augmenting the training set, and repeating.

**Primary metric**: `test_auroc` (higher is better).
**Secondary metrics**: `test_auprc`, `hit_rate`, `discovery_rate`. You are also encouraged to **propose and track your own metrics** that you believe are informative (e.g., internal diversity of selected set, novelty vs. training set, uncertainty calibration, scaffold coverage). Add these to the output and to `results.tsv` as additional columns.

**First run**: Always run `al_optimizer_chemprop.py` unmodified first to establish the baseline.

## Running an experiment

```bash
python al_optimizer_chemprop.py \
  --train data/train_df_chemprop.csv \
  --pool  data/pool_df_chemprop.csv \
  --test  data/test_df_chemprop.csv \
  --iters 1 \
  > al_run.log 2>&1
```

Extract key metrics:
```bash
grep "^test_auroc:\|^hit_rate:\|^test_auprc:" al_run.log
```

Training takes ~30-60 min on GPU per iteration. Prediction takes ~5-10 min.

## Output format

```
test_auroc:     0.834200
test_auprc:     0.052000
hit_rate:       0.041000
discovery_rate: 0.087000
true_hit_rate:  0.123000
selected_pos:   123
novelty:        0.450000
diversity:      0.820000
iteration:      3
selected:       1000
train_size:     13000
---
FINAL_METRICS iter=3: test_auroc=0.834200 test_auprc=0.052000 hit_rate=0.041000
```

You are encouraged to add your own metrics to this output. If you think a metric would be informative (e.g., `novelty_vs_train`, `internal_diversity`, `scaffold_coverage`, `uncertainty_calibration`), add it to the print block and to `results.tsv`. The more signal you capture, the better you can reason about your next iteration.

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated — commas break descriptions). **Commit `results.tsv` to git after every iteration** so that all progress is tracked on GitHub. Include cost logging — track DeepSeek API cost per iteration.

```
commit	iteration	test_auroc	test_auprc	hit_rate	status	description
a1b2c3d	0	0.720000	0.035000	0.031000	baseline	initial run, top-k inhibition only
b2c3d4e	1	0.745000	0.041000	0.042000	run	iter 1: added uncertainty weighting (W_UNCERTAINTY=1.5)
c3d4e5f	2	0.751000	0.043000	0.043000	run	iter 2: added novelty + diversity to acquisition
d4e5f6g	3	0.000000	0.000000	0.000000	crash	iter 3: OOM during training (BATCH_SIZE=256)
e5f6g7h	3	0.748000	0.042000	0.042000	run	iter 3 (retry): reverted BATCH_SIZE to 64, kept novelty
```

- **Keep every commit.** Never `git reset`. Every iteration is a data point, even bad ones.
- `status`: `baseline`, `run`, `crash`, or `fix` (for bug fixes between iterations).
- You may add extra columns for custom metrics you've defined (e.g., `novelty`, `diversity`, `cost`).

## Git tracking and progress saving

All progress must be preserved in GitHub:

1. **After every experiment**: `git add results.tsv state.json && git commit -m "exp <N>: <description>"`
2. **Every 5 experiments**: `git push origin HEAD`
3. **Write `state.json`** after every experiment (see CLAUDE.md for format). This lets you resume if the SLURM job dies.
4. **Log crashes**: If a run crashes, record it in `results.tsv` with status `crash`, commit, and continue to the next idea.
5. **Never lose work**: The code change is always committed *before* the training run starts. If training is killed mid-run, only the result is lost — the code is safe.

## The active learning loop

You execute a **single forward trajectory** through the AL budget. Each iteration selects molecules, reveals their labels, augments the training set, and advances.

LOOP (for each AL iteration):

1. Read `state.json` if it exists — pick up where you left off.
2. Review the trajectory so far: `cat results.tsv`.
3. Based on what you've learned from previous iterations, optionally modify `al_optimizer_chemprop.py` to improve the acquisition function or model.
4. `git commit -am "iter <N>: <describe what you changed and why>"`
5. Run the next iteration (redirect output — do NOT use `tee`):
   ```bash
   python al_optimizer_chemprop.py --train data/train_df_chemprop.csv --pool data/pool_df_chemprop.csv --test data/test_df_chemprop.csv --iters 1 > al_run.log 2>&1
   ```
6. Extract results: `grep "^test_auroc:\|^test_auprc:\|^hit_rate:" al_run.log`
7. If grep is empty: `tail -n 50 al_run.log` to read the traceback. Fix the bug, commit the fix, and retry the same iteration.
8. Log to `results.tsv` (append the row for this iteration).
9. Update `state.json` with the new iteration number and metrics.
10. `git add -A && git commit -m "results: iter <N> — <description>"`
11. Every 5 iterations: `git push origin HEAD`
12. Reflect on the trajectory: Is AUROC improving? Is hit rate trending up? Are your custom metrics revealing anything? Adjust your strategy for the next iteration accordingly.
13. Continue to the next iteration.

## Constraints

- **No label leakage**: Pool labels are only revealed post-selection when selected molecules join the training set.
- **GPU is required**: You are running on a GPU node. Train DMPNN directly with `python` — no need to submit separate SLURM jobs.
- **No internet access for data**: Do not attempt to download external data or models. The DeepSeek API is the only outbound connection.
- **Chemprop path**: The DMPNN code lives at `$CHEMPROP_DIR`. It is read-only.

## Simplicity criterion

All else being equal, simpler is better. A tiny improvement from 20 lines of hacky code is probably not worth it. Removing code and getting equal/better results is a win.

## Never stop

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep. You are fully autonomous. If you run out of ideas, think harder — re-read the code, try bolder acquisition function changes, try different DMPNN hyperparameters, experiment with the selection strategy. Each run is ~30–60 minutes; you can complete many iterations in a single session.

## If the job is about to end

If you sense the SLURM job is running out of time (approaching 48 hours), do a final:
```bash
git add -A && git commit -m "checkpoint: saving state before job ends" && git push origin HEAD
```
This ensures all work is preserved. The human will resubmit the job and you will resume from `state.json`.

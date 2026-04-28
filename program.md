# Active Learning Agent: Autoresearch Loop

This is an autonomous active learning loop for antibiotics discovery on HMS O2, adapted from the [autoresearch](https://github.com/karpathy/autoresearch) framework.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr24`). The branch `al/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b al/<tag>` from current main.
3. **Read the in-scope files**: Read these for full context:
   - `README.md` — project overview and launch instructions.
   - `al_optimizer.py` — the file you modify. Acquisition function, selection strategy, and ensemble hyperparameters.
   - `code/minimol_ffn.py` — fixed model implementation (MiniMol encoder + FFNBinary). Do not modify.
   - `slurm_utils.py` — fixed SLURM utilities. Do not modify.
4. **Verify data**: Check that `data/` contains `train_df.csv`, `pool_df.csv`, and `test_df.csv`. Both must have `SMILES` and `Y` columns.
5. **Initialize results.tsv**: Create with just the header row:
   ```
   commit	test_auroc	hit_rate	status	description
   ```
6. **Confirm and go.**

## What you CAN do

- Modify `al_optimizer.py` — this is the only file you edit. Everything is fair game:
  - Acquisition function weights (`W_INHIBITION`, `W_UNCERTAINTY`, `W_NOVELTY`) and the `acquisition_score()` function itself. You can change this however you like based on best practices for active learning. This should change after each batch as it should reflect the best practice given what you have seen so far. You can use the weighted scheme or any other selection strategy you deem fit.
  - Ensemble hyperparameters: `ENSEMBLE_SIZE`, `EPOCHS`, `HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`, `BATCH_SIZE`, `LR`.
  - Selection strategy (greedy, diverse, or anything else). Greedy/diversity-only are very simple. You can do combinations of these or completely different strategies.
  - Validation split size
  - Any other AL logic.

## What you CANNOT do

- Modify `code/minimol_ffn.py` — it is read-only. It contains the MiniMol encoder and FFN model.
- Modify `slurm_utils.py`. Read-only.
- Use labels from the pool during selection. **No label leakage.**
- Install new packages not available in the `al-agent` conda environment.
- Modify the evaluation. AUROC is always computed on `test_df.csv` with fixed labels.

## The goal

**Execute a single active learning trajectory** that maximizes test_auroc across sequential iterations. You are not A/B testing different strategies — you are running one coherent AL campaign forward, selecting molecules, revealing their labels, augmenting the training set, and repeating.

**Primary metric**: `test_auroc` (higher is better).
**Secondary metrics**: `hit_rate`, `discovery_rate`. You are also encouraged to **propose and track your own metrics** that you believe are informative (e.g., internal diversity of selected set, novelty vs. training set, uncertainty calibration, scaffold coverage). Add these to the output and to `results.tsv` as additional columns.

**First run**: Always run `al_optimizer.py` unmodified first to establish the baseline.

## Running an experiment

```bash
python al_optimizer.py \
  --train data/train_df.csv \
  --pool  data/pool_df.csv \
  --test  data/test_df.csv \
  --iters 1 \
  > al_run.log 2>&1
```

Extract key metrics:
```bash
grep "^test_auroc:\|^hit_rate:" al_run.log
```

## Output format

```
test_auroc:     0.834200
hit_rate:       0.041000
discovery_rate: 0.087000
iteration:      3
selected:       1000
train_size:     8000
---
FINAL_METRICS iter=3: test_auroc=0.834200 hit_rate=0.041000
```

You are encouraged to add your own metrics to this output. If you think a metric would be informative (e.g., `novelty_vs_train`, `internal_diversity`, `scaffold_coverage`, `uncertainty_calibration`), add it to the print block and to `results.tsv`. The more signal you capture, the better you can reason about your next iteration.

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated — commas break descriptions). **Commit `results.tsv` to git after every iteration** so that all progress is tracked on GitHub.

```
commit	iteration	test_auroc	hit_rate	status	description
a1b2c3d	0	0.820000	0.031000	baseline	initial run, top-k inhibition only
b2c3d4e	1	0.834200	0.041000	run	iter 1: added uncertainty weighting
c3d4e5f	2	0.841000	0.043000	run	iter 2: increased novelty weight
d4e5f6g	3	0.000000	0.000000	crash	iter 3: wider FFN caused OOM
e5f6g7h	3	0.838000	0.042000	run	iter 3 (retry): reverted FFN width, kept novelty
```

- **Keep every commit.** Never `git reset`. Every iteration is a data point, even bad ones.
- `status`: `baseline`, `run`, `crash`, or `fix` (for bug fixes between iterations).
- You may add extra columns for custom metrics you've defined (e.g., `novelty`, `diversity`).

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
3. Based on what you've learned from previous iterations, optionally modify `al_optimizer.py` to improve the acquisition function or model.
4. `git commit -am "iter <N>: <describe what you changed and why>"`
5. Run the next iteration (redirect output — do NOT use `tee`):
   ```bash
   python al_optimizer.py --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv --iters 1 > al_run.log 2>&1
   ```
6. Extract results: `grep "^test_auroc:\|^hit_rate:" al_run.log`
7. If grep is empty: `tail -n 50 al_run.log` to read the traceback. Fix the bug, commit the fix, and retry the same iteration.
8. Log to `results.tsv` (append the row for this iteration).
9. Update `state.json` with the new iteration number and metrics.
10. `git add -A && git commit -m "results: iter <N> — <description>"`
11. Every 5 iterations: `git push origin HEAD`
12. Reflect on the trajectory: Is AUROC improving? Is hit rate trending up? Are your custom metrics revealing anything? Adjust your strategy for the next iteration accordingly.
13. Continue to the next iteration.

## Constraints

- **No label leakage**: Pool labels are only revealed post-selection when selected molecules join the training set.
- **GPU is local**: You are running on a GPU node. Train your models directly with `python` — no need to submit separate SLURM jobs.
- **No internet access for data**: Do not attempt to download external data or models. The DeepSeek API is the only outbound connection.

## Simplicity criterion

All else being equal, simpler is better. A tiny improvement from 20 lines of hacky code is probably not worth it. Removing code and getting equal/better results is a win.

## Never stop

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep. You are fully autonomous. If you run out of ideas, think harder — re-read the code, try bolder acquisition function changes, try different ensemble strategies, experiment with the FFN hyperparameters. Each run is ~10–20 minutes; you can complete many experiments in a single session.

## If the job is about to end

If you sense the SLURM job is running out of time (approaching 48 hours), do a final:
```bash
git add -A && git commit -m "checkpoint: saving state before job ends" && git push origin HEAD
```
This ensures all work is preserved. The human will resubmit the job and you will resume from `state.json`.

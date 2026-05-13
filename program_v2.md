# Active Learning Agent: Autoresearch Loop (v2 — Paper-Informed)

This is an autonomous active learning loop for molecular property prediction on HMS O2, driven by an LLM agent (DeepSeek via [opencode](https://github.com/sstcloud/opencode)). **This version gives you prior experimental context from a related study to inform your strategy.**

## Prior Experimental Context

You are working with Mycobacterium tuberculosis (Mtb) high-throughput screening data — a subset of the same dataset used in a recent study on active learning for antibiotic discovery. Here is what that study found, which you should use to inform your strategy:

### Key Findings from the Reference Study
- **Dataset**: ~115K compounds screened against Mtb for growth inhibition, 5.7% hit rate (6,572 actives). Binary labels: >=80% inhibition = active.
- **Model**: DMPNN + evidential Dirichlet head (Chemprop), 5-replicate ensemble. Your setup uses MiniMol fingerprints + FFN ensemble — a lighter-weight but comparable architecture.
- **Strategies benchmarked**:
  1. **Inhibition-only**: Selects highest predicted inhibition probability. Simple but effective at hit discovery.
  2. **Diversity-Inhibition-Novelty (DNI)**: Weights diversity, novelty from training set, and inhibition. **Best overall performer** — matched Inhibition-only on hit discovery but produced more generalizable models with better precision-recall on out-of-distribution molecules.
  3. **DNI + Uncertainty**: Added aleatoric + epistemic uncertainty terms. Similar hit discovery to DNI, but uncertainty prioritized novelty at early iterations. Did NOT improve AUPRC over DNI alone when labeling budget was large (~50%).
- **Key insight 1**: Novelty contribution gets rapidly exhausted as the library shrinks. After a point, remaining molecules have similar novelty scores, so novelty-based rankings become ineffective. Watch for this in later iterations.
- **Key insight 2**: Diversity (intra-batch chemical dissimilarity) was more durable than novelty and maintained selection quality throughout the campaign.
- **Key insight 3**: Real-time application showed 5-fold increase in hit rate (1.0% vs 0.1%) compared to non-AL scientist-guided selection.
- **Key insight 4**: Class imbalance is extreme (~0.7% hit rate in your data). The paper used class balancing during training. Consider this.

### Your Task
Use these insights to design a more informed strategy than the default (equal-weight inhibition + uncertainty). You have:
- **Initial train**: ~10,000 labeled molecules (~65 positives, 0.65% hit rate)
- **Pool**: ~97,000 unlabeled molecules (~750 positives hidden)
- **Test**: ~108,000 held-out molecules
- **Budget**: 10 iterations, 1,000 molecules per iteration

## Architecture
- **Encoder**: MiniMol (Morgan fingerprints, 2048-bit, radius 2 → 512-dim embedding).
- **Model**: FFN ensemble (3 members, 2 hidden layers × 512 dim, batch norm, dropout 0.1).
- **Agent**: opencode + DeepSeek API — autonomously modifies `al_optimizer.py`, runs experiments, and logs results.
- **Compute**: O2 interactive CPU node (8-10 cores, 64 GB RAM).

## Objectives
Your primary goal is to **maximize the AUROC on a held-out test set** through iterative active learning. Track AUPRC, hit rate, novelty, and diversity as secondary indicators. The paper found DNI was best for generalizability — aim for strategies that balance exploitation (inhibition) with diversity and novelty.

## Problem Setup
- **Total Pool**: ~96,000 unlabeled molecules.
- **AL Budget**: 10 iterations.
- **Selection**: Select 1,000 molecules per iteration.
- **Initial Train**: ~10,000 labeled molecules (10% of pool).
- **Evaluation**: Fixed, held-out `test_df.csv` (107K molecules) for AUROC/AUPRC.

## Setup
1. **Branch**: Already on `al/may13_v2` — start here.
2. **Initialize results.tsv**: Already initialized with baseline row.
4. **Data**: train_df.csv (10% initial labeled), pool_df.csv (90% unlabeled candidates), test_df.csv (held-out). All three are disjoint.
4. **Environment**: Already activated (`al-agent` conda env).

## Experimentation Loop

LOOP FOREVER:

1. **Read the in-scope files**: `al_optimizer.py` (modifiable acquisition function, weights, hyperparameters), `code/minimol_ffn.py` (read-only — fixed encoder + FFN model).
2. **Modify `al_optimizer.py`**: Refine candidate selection strategies (acquisition weights, selection size, ensemble size, training hyperparameters). **Use the paper's findings** — consider diversity, novelty exhaustion, class balancing, and ensemble calibration.
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
   - **Reference the paper insights** when making decisions — explain in commit messages which finding informed your choice.

## Constraints
- **No Label Leakage**: Do NOT use pool labels for selection. Labels are only revealed when a molecule is selected and moved to the training set.
- **Read-only model**: `code/minimol_ffn.py` is fixed — only modify `al_optimizer.py`.

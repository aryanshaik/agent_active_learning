# Active Learning Agent: Autoresearch Loop

This is an autonomous active learning loop for molecular property prediction on HMS O2. You are an LLM agent (DeepSeek via opencode). **Start by reading the full reference paper at the bottom of this file — it describes prior experimental results on this exact dataset and task.**

## Architecture
- **Encoder**: MiniMol (Morgan fingerprints, 2048-bit, radius 2 → 512-dim embedding).
- **Model**: FFN ensemble (3 members, 2 hidden layers × 512 dim, batch norm, dropout 0.1).
- **Agent**: opencode + DeepSeek API.
- **Compute**: O2 interactive CPU node (8-10 cores, 64 GB RAM).

## Objectives
Maximize the **AUROC on a held-out test set** through iterative active learning. Track AUPRC, hit rate, novelty, and diversity as secondary indicators.

## Problem Setup
- **Total Pool**: ~96,000 unlabeled molecules.
- **AL Budget**: 10 iterations.
- **Selection**: 1,000 molecules per iteration.
- **Initial Train**: ~10,000 labeled molecules (10% of pool).
- **Evaluation**: Fixed, held-out test_df.csv (107K molecules).

## Setup (already done)
- Branch: al/may13_v2
- results.tsv: initialized with baseline row
- Data: train_df.csv (10%), pool_df.csv (90%), test_df.csv — all disjoint
- Environment: al-agent conda env activated

## Cost Tracking
After each iteration, add token usage and estimated cost to the results.tsv description field. Track:
- Approximate tokens used (prompt + response) for that iteration'\''s decisions
- Estimated DeepSeek API cost (DeepSeek Flash: ~$0.07/M input tokens, ~$0.28/M output tokens)
- Cumulative cost across the campaign

## Experimentation Loop

LOOP:

1. **Read files**: al_optimizer.py (modifiable), code/minimol_ffn.py (read-only), results.tsv (history).
2. **Modify al_optimizer.py**: Refine acquisition weights, selection size, ensemble size, training hyperparameters. Use the paper'\''s findings.
3. **Git Commit**: git commit -am "<strategy description, include cost estimate>"
4. **Run**:
   python -u al_optimizer.py --iters 1 --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv 2>&1 | tee al_run.log
5. **Extract Metrics**: Parse FINAL_METRICS line from al_run.log:
   FINAL_METRICS iter=N: test_auroc=X test_auprc=Y hit_rate=Z
   Also grep novelty, diversity, true_hit_rate, selected_pos from above.
6. **Log Results**: Append row to results.tsv: commit hash, iteration, metrics, status, description (including token usage and cost).
7. **Decision**: Keep all runs (no git reset). If performance drops, analyze and adjust.

## Constraints
- **No Label Leakage**: Pool labels hidden until selection.
- **Read-only model**: code/minimol_ffn.py is fixed. Only modify al_optimizer.py.
- **Never ask to stop**: Run fully autonomously.

---

# Reference Paper

The following is a draft manuscript describing active learning experiments on M. tuberculosis HTS data — the same dataset and task. Read it carefully.

## Active Learning-guided narrow-spectrum antibiotics discovery for infectious disease

Lia, Andrew, Alex, Kee-Lee, Yasha, Jim, Peter, Eric, Linden, Maha

### ABSTRACT

Antibiotic discovery is increasingly reliant on high throughput screening (HTS) for identifying whole bacterial cell growth inhibitors—especially for bacteria with poorly understood drug targets or difficult to penetrate cell walls. HTS can be costly, and the choice of compounds to screen is critical to success. The use of machine learning (ML) to guide HTS compound selection offers promise not only for increasing screening hit rate but also for building more generalizable models able to predict inhibitory effects for out-of-distribution molecules. Such models can greatly accelerate antibiotic discovery when resources are limited, effectively driving down the cost needed to generate new leads. The iterative ML and experimental HTS strategy, also known as Active Learning (AL) has been successfully applied for targeted bioactivity screening, but not yet for whole-cell bacterial bioactivity. Further, it is not known if, within AL, prioritizing compound novelty, inhibition or a combination is needed across metrics of success of ML and whole-cell bacterial bioactive HTS. In this work, we design an AL strategy to prioritize a balance of novelty, growth inhibition and other chemical characteristics for a plate-based selection process compatible with HTS. We explore how our strategy benefits the discovery of bioactives using simulations of retrospective Mycobacterium tuberculosis HTS data (n=20,000 of 114,000). We demonstrate the success of the chosen strategy in a real-time HTS for a Borrelia burgdorferi antibiotic discovery campaign resulting in a 10-fold increase in discovery rate (1.0%) relative to non-AL-assisted plate selection (0.1%). Further, we show that AL can be adapted to guide inhibitor discovery that demonstrate narrow-spectrum activity against the target bacteria while sparing commensals. AL demonstrates promise in navigating large chemical spaces for identifying compounds with multiple desirable properties.

### MAIN TEXT

Antibiotic discovery increasingly relies on high throughput in vitro screening (HTS) for identifying novel compounds that inhibit the growth of whole bacterial cells. However, this approach can be costly, making the prioritization of compounds for screening critical. Active Learning (AL) is a strategy applied to myriad ML-powered tasks wherein samples are iteratively and selectively labeled for supervised model training depending upon their potential value to the model. AL is particularly apt for HTS due to its iterative format compatible with batched HTS timelines and because experimenters are typically choosing from vast chemical libraries consisting of hundreds of thousands of compounds. As such, AL has demonstrated to be effective in steering such selection processes, balancing sample labeling priority between model improvement and discovery of bioactives. No such methods, however, have applied AL-curated training data on whole-cell activity prediction for causative agents of infectious disease. While many assays screen for compound activity on specific intracellular targets, the success rate does not always translate to whole-cell growth inhibition, often due to the inability to permeabilize the cell wall or account for compensatory cellular effects, rendering whole cell activity hit discovery elusive. Further, we demonstrate our AL-guided models can discover potent B. burgdorferi inhibitors with underrepresented chemical scaffolds.

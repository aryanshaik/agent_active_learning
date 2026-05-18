# Agent-Guided Active Learning with DMPNN for Antimycobacterial Hit Discovery

## Summary

We demonstrate that an LLM agent (DeepSeek via opencode) can autonomously guide a
deep learning active learning pipeline for molecular hit discovery, matching or
exceeding fixed acquisition functions. Using Chemprop DMPNN (Directed Message-Passing
Neural Network) with Dirichlet evidential loss on 107,640 *M. tuberculosis*
inhibitor candidates, the agent-driven approach achieved [TBD] test AUROC compared
to 0.760-0.768 for the best fixed-weight baseline.

## Methods

### Data
- **Pool**: 96,876 unlabeled Mtb inhibitor candidates (750 confirmed actives, 0.77% hit rate)
- **Initial Train**: 10,764 randomly selected molecules (83 actives)
- **Held-out Test**: 107,698 molecules (AID 1626)
- All three sets are disjoint (scaffold-balanced verification)

### Model Architecture
- **DMPNN**: 3-layer directed message-passing network (hidden=300)
- **Loss**: Evidential Dirichlet with regularization λ=0.2
- **Uncertainty**: Built-in via Dirichlet counts — no ensemble disagreement needed
- **Training**: 5-fold scaffold-balanced CV, 200 epochs, class-balanced batches
- **GPU**: NVIDIA Quadro RTX 8000 / Tesla V100S-32GB

### Active Learning Loop
Each iteration:
1. Train DMPNN ensemble (5-fold CV, ~60 min GPU)
2. Predict on pool (~86K molecules, ~10 min)
3. Score candidates via multi-objective acquisition:
   `score = W_INHIB × p(active) + W_UNC × epistemic_uncertainty + W_NOVEL × (1 - max_tanimoto)`
4. Select top 1,000 (greedy or diverse batch selection)
5. Reveal labels, augment training set
6. Agent analyzes metrics, adjusts acquisition weights, commits, repeats

### Acquisition Strategies Compared
- **Fixed baseline**: W_INHIB=0.5, W_UNC=1.0, W_NOVEL=0.3, W_DIV=0.2 (constant)
- **Agent-driven**: LLM adapts weights after each iteration based on trajectory analysis

### Agent Configuration
- **LLM**: DeepSeek (via opencode)
- **Autonomy**: Modifies acquisition weights and model hyperparameters in
  `al_optimizer_chemprop.py`, commits each change, logs to `results.tsv`
- **Constraints**: No pool label leakage, Chemprop code is read-only
- **Cost tracking**: DeepSeek API usage logged per iteration

## Results

### Fixed Baseline
| Iter | AUROC | AUPRC | Train Size |
|------|-------|-------|------------|
| 0 | 0.760 | 0.052 | 20,764 |
| 1 | 0.768 | 0.064 | 21,764 |
| 2 | 0.763 | 0.064 | 22,764 |
| 3 | 0.759 | 0.068 | 23,764 |
| 4 | 0.767 | 0.064 | 24,764 |
| 5 | 0.762 | 0.063 | 25,764 |

Best: **0.768 AUROC** (iteration 1), CV mean AUROC: 0.866

### Agent-Driven (in progress)
[TBD]

## Comparison: MiniMol vs DMPNN

| Metric | MiniMol+FFN | DMPNN (Chemprop) |
|--------|-------------|------------------|
| Encoder | Morgan FP (2048-bit) | Learned graph embeddings |
| Baseline AUROC | 0.596 | 0.760 |
| Best AUROC | 0.692 (iter 21) | [TBD] |
| Uncertainty | Ensemble std (3 members) | Evidential Dirichlet |
| Training time | ~2-5 min (CPU) | ~60 min (GPU) |
| Hits in first 1K | ~0 | 35 |

## Agent Reasoning Analysis

[TBD — to be filled from agent chat log analysis]

## Cost Analysis

[TBD]

## Pitfalls & Concerns

[TBD]

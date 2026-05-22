# Run 3: DMPNN Agent-Driven Active Learning — `al/may18`

**Date**: 2026-05-18  
**Model**: Chemprop DMPNN (3-layer, hidden=300, Dirichlet evidential loss)  
**Agent**: opencode + DeepSeek Flash  
**Data**: 10,764 initial train, 96,876 pool, 107,698 held-out test (AID 1626)  
**Compute**: O2 gpu_quad (Quadro RTX 8000 / Tesla V100S)

## Agent Reasoning & Acquisition Function Evolution

The agent ("aggressive exploration"): W_INHIB=0.5→0.2, W_UNC=1.0→1.5, W_NOV=0.3→1.0, W_DIV=0.2→0.5, BATCH_DIVERSE=True  
*"AUPRC=0.025 is critically low; need to explore diverse OOD chemical space to find more actives and improve precision-recall."*

### Iter 2: Balanced exploration-exploitation
**AUROC: 0.710** (+0.001)  
W_INHIB=0.2→0.4, W_UNC=1.5, W_NOV=1.0→0.6, W_DIV=0.5→0.4  
*"Increased inhibition to leverage improving AUROC, reduced novelty to avoid overly OOD negatives. AUPRC remains critically low at 0.029."*

### Iter 3: Break local bias via re-exploration
**AUROC: 0.713** (+0.003)  
W_INHIB=0.4→0.2, W_UNC=1.5, W_NOV=0.6→1.2, W_DIV=0.4→0.6  
*"Hit rate tripled (9→29%) but AUROC flat + AUPRC down signals overconfident model stuck in narrow positive region. Forcing exploration of unfamiliar chemical space."*

### Iter 4: Confident exploration pivot → **BREAKTHROUGH**
**AUROC: 0.730** (+0.017) — LARGEST JUMP  
W_INHIB=0.2→0.6, W_UNC=1.5→1.8, W_NOV=1.2→0.5, W_DIV=0.6→0.4  
*"AUROC trending up (0.702→0.713) so model signal is real; pivot to inhibition+uncertainty dominance. Trust improving predictions while exploring uncertain regions."*

### Iter 5: Capitalize on momentum
**AUROC: 0.738** (+0.008)  
W_INHIB=0.6→0.7, W_UNC=1.8→1.5, W_NOV=0.5→0.8, W_DIV=0.4→0.5  
*"AUROC jumped +0.017 with confident exploration. Increasing inhibition to exploit improving signal. Adding novelty to find diverse positives."*

### Iter 6: Late-stage exploitation ramp
**AUROC: 0.740** (+0.002)  
W_INHIB=0.7→0.9, W_UNC=1.5→1.2, W_NOV=0.8→0.7, W_DIV=0.5→0.4  
*"AUROC/AUPRC both at all-time highs for 3 straight iterations. Increasing inhibition to maximize exploitation of improving model as we converge."*

### Iter 7: Exploration reset after plateau
**AUROC: 0.737** (-0.003)  
W_INHIB=0.9→0.6, W_UNC=1.2→1.6, W_NOV=0.7→1.0, W_DIV=0.4→0.5  
*"AUROC/AUPRC stalled (+0.002/+0.0001) under pure exploitation. Drastic cut to inhibition, spike uncertainty and novelty to discover remaining undiscovered positives."*

### Iter 8: Revert to exploitation peak → **BEST**
**AUROC: 0.757** (+0.020)  
W_INHIB=0.6→0.9, W_UNC=1.6→1.0, W_NOV=1.0→0.5, W_DIV=0.5→0.3  
*"Exploration pulse regressed both AUROC and AUPRC, proving iter 6 plateau was genuine convergence. Returning to iter 6 peak configuration."*

### Iter 9: Maximal exploitation (KILLED BY TIMEOUT)
W_INHIB=0.9→1.0, W_UNC=1.0→0.8, W_NOV=0.5→0.3, W_DIV=0.3→0.2  
*"Doubling down on winning formula. Iter 8 was the best by far."*

## Key Insights

1. **Explore-then-exploit pattern**: Agent independently discovered that early exploration builds a diverse training set, then exploitation maximizes AUROC — same pattern seen in MiniMol Run 2 but executed faster (8 vs 21 iterations).

2. **Self-correction**: Agent detected a false plateau at iter 7 and deliberately tried an exploration reset. When that failed, it correctly reverted to the exploitation strategy.

3. **Strategic pivots aligned with improvements**: The two largest AUROC jumps (+0.017 at iter 4, +0.020 at iter 8) both followed deliberate strategic pivots.

4. **Cost**: DeepSeek API ~$0.01-0.02 total for 9 agent turns. Negligible.

5. **vs Fixed Baseline**: The fixed baseline achieved 0.768 with 20,764 initial training molecules. The agent achieved 0.757 starting from only 10,764 (half the data). At iter 8, training set = 18,764 molecules — approaching the fixed baseline's size with comparable performance.

## Comparison Across All Runs

| Run | Model | Initial Train | Best AUROC | Iterations | Agent? |
|-----|-------|---------------|------------|------------|--------|
| Run 1 | MiniMol+FFN | 10,764 | 0.666 | 10 | Yes (simple) |
| Run 2 | MiniMol+FFN | 10,764 | 0.692 | 21 | Yes (sophisticated) |
| Run 3 | DMPNN | 20,764 | 0.768 | 6 | No (fixed weights) |
| Run 4 | DMPNN | 10,764 | **0.757** | 9 | Yes (openCode) |

DMPNN agent (0.757 from 10.7K) is comparable to DMPNN fixed (0.768 from 20.8K) despite half the initial training data.

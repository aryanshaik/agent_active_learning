#!/bin/bash
# Hybrid agent loop: opencode proposes weights, bash runs training
set -e
source /home/ars3983/miniforge/bin/activate al-agent
export WANDB_MODE=disabled
cd /n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning
cp data/train_df_chemprop_orig.csv data/train_df_chemprop.csv
cp data/pool_df_chemprop_orig.csv data/pool_df_chemprop.csv

OPCODE="/home/ars3983/.opencode/bin/opencode"
PORT=4096

echo "=== Hybrid Agent AL Loop ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Start server once
$OPCODE serve --port $PORT &
SERVER_PID=$!
for i in $(seq 1 30); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready (${i}s)"; break
  fi
  sleep 1
done

ATTACH="--attach http://localhost:$PORT --dangerously-skip-permissions"

# ── SETUP ──
echo "=== SETUP ==="
$OPCODE run $ATTACH \
"Read skills/chemprop-al-optimizer.md.
Then:
1. git checkout -b al/may18 (mv results.tsv to results.tsv.bak if it blocks)
2. printf 'commit\titeration\ttest_auroc\ttest_auprc\thit_rate\tstatus\tdescription\n' > results.tsv
3. git add results.tsv && git commit -m 'init results.tsv'
Report when done."

echo "Setup complete: $(date)"

# ── ITER 0: BASELINE (no agent, default weights) ──
echo "=== Iter 0: Baseline ==="
python -u al_optimizer_chemprop.py \
  --train data/train_df_chemprop.csv \
  --pool data/pool_df_chemprop.csv \
  --test data/test_df_chemprop.csv \
  --iters 1 \
  --work_dir al_chemprop_runs \
  > al_run.log 2>&1

# Parse metrics
AUROC=$(grep "FINAL_METRICS" al_run.log | grep -oP 'test_auroc=\K[\d.]+')
AUPRC=$(grep "FINAL_METRICS" al_run.log | grep -oP 'test_auprc=\K[\d.]+')
HIT=$(grep "FINAL_METRICS" al_run.log | grep -oP 'hit_rate=\K[\d.]+')
COMMIT=$(git rev-parse --short HEAD)
echo -e "$COMMIT\t0\t$AUROC\t$AUPRC\t$HIT\tbaseline\tDefault weights (W_INHIB=0.5,W_UNC=1.0,W_NOV=0.3,W_DIV=0.2)" >> results.tsv
git add -A && git commit -m "iter 0: baseline (default weights) — AUROC=$AUROC"
echo "Iter 0: AUROC=$AUROC AUPRC=$AUPRC"

# ── ITER 1-9: Agent proposes, bash runs ──
for iter in $(seq 1 9); do
  echo "=== Iter $iter: Agent thinking... ($(date)) ==="

  # Let agent analyze trajectory and propose changes
  $OPCODE run $ATTACH --continue \
"AL iteration $iter. Read results.tsv for the full trajectory.

Based on the AUROC/AUPRC trends, propose NEW acquisition weights for al_optimizer_chemprop.py.
Respond with the EXACT sed commands or python code to update the weights.
For example: 'Change W_INHIBITION=0.7 W_UNCERTAINTY=1.5 W_NOVELTY=0.5'

Then modify the file and commit:
1. Update W_INHIBITION, W_UNCERTAINTY, W_NOVELTY, W_DIVERSITY in al_optimizer_chemprop.py
2. git commit -am 'iter $iter: <your strategy>'
3. Report the new weights and your reasoning."

  echo "Iter $iter: agent done, running training..."

  # Run training with new weights
  python -u al_optimizer_chemprop.py \
    --train data/train_df_chemprop.csv \
    --pool data/pool_df_chemprop.csv \
    --test data/test_df_chemprop.csv \
    --iters 1 \
    --work_dir al_chemprop_runs \
    > al_run.log 2>&1

  AUROC=$(grep "FINAL_METRICS" al_run.log | grep -oP 'test_auroc=\K[\d.]+')
  AUPRC=$(grep "FINAL_METRICS" al_run.log | grep -oP 'test_auprc=\K[\d.]+')
  HIT=$(grep "FINAL_METRICS" al_run.log | grep -oP 'hit_rate=\K[\d.]+')
  COMMIT=$(git rev-parse --short HEAD)

  if [ -n "$AUROC" ]; then
    echo -e "$COMMIT\t$iter\t$AUROC\t$AUPRC\t$HIT\trun\tAgent-modified weights" >> results.tsv
    git add -A && git commit -m "results: iter $iter — AUROC=$AUROC"
    echo "Iter $iter: AUROC=$AUROC AUPRC=$AUPRC"
  else
    echo -e "$COMMIT\t$iter\t0.0\t0.0\t0.0\tcrash\tTraining failed" >> results.tsv
    git add -A && git commit -m "crash: iter $iter"
    echo "Iter $iter: CRASHED"
  fi

  # Push every 5
  if [ $((iter % 5)) -eq 0 ]; then
    git push origin HEAD 2>/dev/null || true
  fi
done

echo "=== ALL 10 ITERS DONE: $(date) ==="
cat results.tsv
git push origin HEAD 2>/dev/null || true
kill $SERVER_PID 2>/dev/null

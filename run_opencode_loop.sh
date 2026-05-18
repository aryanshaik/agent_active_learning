#!/bin/bash
# OpenCode autonomous AL loop using chemprop-al-optimizer skill
# Requires: opencode installed, GPU node, conda env al-agent
set -e
source /home/ars3983/miniforge/bin/activate al-agent
export WANDB_MODE=disabled
cd /n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning

# Restore fresh data
cp data/train_df_chemprop_orig.csv data/train_df_chemprop.csv
cp data/pool_df_chemprop_orig.csv data/pool_df_chemprop.csv

echo "=== OpenCode DMPNN AL Loop ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

OPCODE="/home/ars3983/.opencode/bin/opencode"
PORT=4096

# Start headless server
$OPCODE serve --port $PORT &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 30); do
  if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server ready (${i}s)"
    break
  fi
  sleep 1
done

ATTACH="--attach http://localhost:$PORT --dangerously-skip-permissions"

# ── SETUP: create branch, init, run baseline ──
echo "=== SETUP ==="
$OPCODE run $ATTACH \
"You are the chemprop-al-optimizer agent. Read skills/chemprop-al-optimizer.md for your full instructions.

START NOW:
1. git checkout -b al/may18 (if blocked by results.tsv, mv it to results.tsv.bak first)
2. Initialize results.tsv: printf 'commit\titeration\ttest_auroc\ttest_auprc\thit_rate\tstatus\tdescription\n' > results.tsv
3. Write state.json: {\"iteration\": 0, \"branch\": \"al/may18\"}
4. Run baseline (iteration 0): python -u al_optimizer_chemprop.py --train data/train_df_chemprop.csv --pool data/pool_df_chemprop.csv --test data/test_df_chemprop.csv --iters 1 --work_dir al_chemprop_runs > al_run.log 2>&1
5. Parse metrics from al_run.log
6. Append to results.tsv with status=baseline
7. git add -A && git commit -m 'iter 0: baseline (default weights)'
8. Report test_auroc

Do ALL 8 steps. Do not stop."

echo "Setup done: $(date)"

# ── ITERATION LOOP ──
for iter in $(seq 1 9); do
  echo "=== Iter $iter — $(date) ==="

  $OPCODE run $ATTACH --continue \
"AL iteration $iter of 10. Read results.tsv and state.json for trajectory.

Based on AUROC/AUPRC trends, modify acquisition weights in al_optimizer_chemprop.py
(W_INHIBITION, W_UNCERTAINTY, W_NOVELTY, W_DIVERSITY, BATCH_DIVERSE, SELECTION_SIZE).

Then:
1. git commit -am 'iter $iter: <describe strategy>'
2. python -u al_optimizer_chemprop.py --train data/train_df_chemprop.csv --pool data/pool_df_chemprop.csv --test data/test_df_chemprop.csv --iters 1 --work_dir al_chemprop_runs > al_run.log 2>&1
3. grep FINAL_METRICS al_run.log
4. Append to results.tsv with status=run
5. Update state.json
6. git add -A && git commit -m 'results: iter $iter'
7. Report test_auroc and cost

Never git reset. Keep all commits."

done

echo "=== ALL 10 ITERS DONE: $(date) ==="
cat results.tsv
kill $SERVER_PID 2>/dev/null

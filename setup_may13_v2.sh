#!/bin/bash
set -e
source /home/ars3983/miniforge/bin/activate al-agent
cd /n/data1/hms/dbmi/farhat/aryan/AL/agent_active_learning

echo "=== Creating al/may13_v2 branch ==="
git checkout main
git checkout -b al/may13_v2

echo "=== Resetting data to initial split ==="
python3 -c "
import pandas as pd, numpy as np
train = pd.read_csv('data/train_df.csv')
pool = pd.read_csv('data/pool_df.csv')
combined = pd.concat([train, pool]).drop_duplicates(subset='SMILES', keep='first')
np.random.seed(42)
idx = np.random.permutation(len(combined))
split = int(len(combined) * 0.10)
combined.iloc[idx[:split]].to_csv('data/train_df.csv', index=False)
combined.iloc[idx[split:]].to_csv('data/pool_df.csv', index=False)
print(f'Reset: Train={split}, Pool={len(combined)-split}')
"

echo "=== Resetting results.tsv ==="
head -1 results.tsv > results.tsv.tmp
echo "1a8d464	0	0.596296	0.000028	baseline	Initial run: default weights W_INHIBITION=1.0 W_UNCERTAINTY=1.0, 3-member FFN ensemble, 10% initial labeled data (65 pos), 1000 selected, 74 true hits (7.4%), novelty=0.42, diversity=0.85" >> results.tsv.tmp
mv results.tsv.tmp results.tsv

echo "=== Cleaning cache ==="
rm -f cache/iter0*

echo "=== Done ==="
git branch --show-current
echo "---"
head -3 results.tsv
echo "---"
wc -l data/train_df.csv data/pool_df.csv

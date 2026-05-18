"""Recover metrics from saved predictions — no GPU needed."""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score

iter_dir = "al_chemprop_runs/iter_00"
test_df = pd.read_csv("data/test_df_chemprop.csv")
pool_df = pd.read_csv("data/pool_df_chemprop.csv")
train_df = pd.read_csv("data/train_df_chemprop.csv")

# Load test predictions
test_preds = pd.read_csv(f"{iter_dir}/test_preds.csv")
merged = test_preds.merge(test_df[["SMILES", "Y"]].rename(columns={"Y": "_true_Y"}), on="SMILES", how="inner")
y_true = merged["_true_Y"].values.astype(float)
y_prob = merged["Y"].values.astype(float)
auroc = roc_auc_score(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
hit_rate = float((y_prob >= 0.5).mean())

# Load pool predictions
pool_preds = pd.read_csv(f"{iter_dir}/pool_preds.csv")
mean_probs = pool_preds["Y"].values.astype(float)
unc_col = [c for c in pool_preds.columns if "uncertainty" in c.lower()][0]
uncertainty = pool_preds[unc_col].values.astype(float)

# Compute novelty
def compute_novelty(pool_smiles, train_smiles):
    fps_train = []
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps_train.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    novelties = []
    for s in pool_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol and fps_train:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
            novelties.append(1.0 - max(sims))
        else:
            novelties.append(0.0)
    return np.array(novelties)

def compute_diversity(smiles_list):
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    return 1.0 - (float(np.mean(sims)) if sims else 0.0)

# Selection (greedy top-k)
W_INHIBITION = 0.5
W_UNCERTAINTY = 1.0
W_NOVELTY = 0.3
SELECTION_SIZE = 1000

novelty_scores = compute_novelty(pool_df["SMILES"].tolist(), train_df["SMILES"].tolist())
scores = W_INHIBITION * mean_probs + W_UNCERTAINTY * uncertainty + W_NOVELTY * novelty_scores

df = pool_df.copy()
df["_score"] = scores
selected = df.sort_values("_score", ascending=False).head(SELECTION_SIZE)

selected_smiles = selected["SMILES"].tolist()
novelty_mean = float(np.mean(compute_novelty(selected_smiles, train_df["SMILES"].tolist())))
diversity = compute_diversity(selected_smiles)

pool_df_orig = pd.read_csv("data/pool_df_chemprop.csv")
selected_with_labels = pool_df_orig[pool_df_orig["SMILES"].isin(selected_smiles)].drop_duplicates("SMILES")
true_hit_rate = float(selected_with_labels["Y"].mean())
selected_pos = int(selected_with_labels["Y"].sum())

print(f"=== BASELINE (Iteration 0) ===")
print(f"CV mean AUROC: 0.8660 (from training log)")
print(f"test_auroc:     {auroc:.6f}")
print(f"test_auprc:     {auprc:.6f}")
print(f"hit_rate:       {hit_rate:.6f}")
print(f"true_hit_rate:  {true_hit_rate:.6f}")
print(f"selected_pos:   {selected_pos}")
print(f"novelty:        {novelty_mean:.6f}")
print(f"diversity:      {diversity:.6f}")
print(f"train_size:     {len(train_df)}")
print(f"pool_size:      {len(pool_df) - SELECTION_SIZE}")
print(f"selected:       {SELECTION_SIZE}")
print(f"iteration:      0")
print(f"\nFINAL_METRICS iter=0: test_auroc={auroc:.6f} test_auprc={auprc:.6f} hit_rate={hit_rate:.6f}")

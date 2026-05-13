import os
import sys
import json
import argparse
import time
import importlib.util

import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location("minimol_ffn", "code/minimol_ffn.py")
mffn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mffn)
MinimolFFNTrainer = mffn.MinimolFFNTrainer
DEVICE = mffn.DEVICE
MinimolFFNBinary = mffn.MinimolFFNBinary
CachedEncoder = mffn.CachedEncoder

from minimol import Minimol

W_INHIBITION = 0.2
W_UNCERTAINTY = 1.0
SELECTION_SIZE = 1000
VALIDATION_FRAC = 0.1

ENSEMBLE_SIZE = 5
EPOCHS = 25
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 256
LR = 1e-3


def train_ensemble(train_df, val_df, test_df, cache_file, n_members=ENSEMBLE_SIZE):
    models = []
    for i in range(n_members):
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)

        trainer = MinimolFFNTrainer(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cache_file=cache_file,
            batch_size=BATCH_SIZE,
            lr=LR,
            epochs=EPOCHS,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        trainer.run()
        models.append(trainer.model)
    return models


def make_encoder(train_df, cache_file):
    encoder_model = Minimol()
    if torch.cuda.is_available():
        try:
            encoder_model = encoder_model.to("cuda")
        except Exception:
            pass
    return CachedEncoder(encoder_model, cache_file)


def predict_ensemble(models, smiles_list, encoder):
    all_probs = []
    with torch.no_grad():
        x = encoder.encode(smiles_list).to(DEVICE)
        for model in models:
            model.eval()
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.stack(all_probs, axis=-1)
    mean_probs = all_probs.mean(axis=-1)
    std_probs = all_probs.std(axis=-1)
    return mean_probs, std_probs, all_probs


def acquisition_score(mean_probs, std_probs):
    return W_INHIBITION * mean_probs + W_UNCERTAINTY * std_probs


def select_molecules(pool_df, mean_probs, std_probs, selection_size=SELECTION_SIZE):
    scores = acquisition_score(mean_probs, std_probs)
    df = pool_df.copy()
    df["_score"] = scores
    df = df.sort_values("_score", ascending=False)
    return df.head(selection_size).drop(columns=["_score"])


def compute_test_metrics(models, test_df, cache_file):
    encoder = make_encoder(test_df, cache_file)
    smiles_list = test_df["SMILES"].tolist()
    y_true = test_df["Y"].values

    x = encoder.encode(smiles_list).to(DEVICE)
    member_probs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            logits = m(x)
            member_probs.append(torch.sigmoid(logits).cpu().numpy())
    member_probs = np.stack(member_probs, axis=-1)
    y_prob = member_probs.mean(axis=-1)

    from sklearn.metrics import roc_auc_score, average_precision_score

    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = 0.5
        auprc = 0.0
    y_pred = (y_prob >= 0.5).astype(float)
    hit_rate = y_pred.mean()
    return {"auroc": auroc, "auprc": auprc, "hit_rate": hit_rate, "y_prob": y_prob}


def compute_novelty(selected_smiles, train_smiles):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps_train = []
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps_train.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    novelties = []
    for s in selected_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol and fps_train:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
            novelties.append(1.0 - max(sims))
        else:
            novelties.append(0.0)
    return float(np.mean(novelties)) if novelties else 0.0


def compute_diversity(smiles_list):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
    return 1.0 - (float(np.mean(sims)) if sims else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="cache")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    train_df = pd.read_csv(args.train)
    pool_df = pd.read_csv(args.pool)
    test_df = pd.read_csv(args.test)
    pool_df_orig = pool_df.copy()

    print(f"Train: {len(train_df)} | Pool: {len(pool_df)} | Test: {len(test_df)}")

    for iteration in range(args.iters):
        print(f"\n{'=' * 60}")
        print(f"  Iteration {iteration}")
        print(f"{'=' * 60}")
        print(f"  Train size: {len(train_df)} (pos={train_df.Y.sum()})")
        print(f"  Pool size:  {len(pool_df)}")

        train_inner = train_df.sample(frac=1 - VALIDATION_FRAC, random_state=42)
        train_smiles_set = set(train_inner["SMILES"])
        val_df = train_df[~train_df["SMILES"].isin(train_smiles_set)].reset_index(
            drop=True
        )
        train_inner = train_inner.reset_index(drop=True)

        cache_file = os.path.join(args.cache_dir, f"iter{iteration}")

        t0 = time.time()
        models = train_ensemble(train_inner, val_df, test_df, cache_file)
        train_time = time.time() - t0
        print(f"  Train: {train_time:.1f}s")

        encoder = make_encoder(train_inner, cache_file)
        pool_smiles = pool_df["SMILES"].tolist()
        t1 = time.time()
        mean_probs, std_probs, _ = predict_ensemble(models, pool_smiles, encoder)
        pred_time = time.time() - t1
        print(f"  Predict: {pred_time:.1f}s")

        selected = select_molecules(pool_df, mean_probs, std_probs, SELECTION_SIZE)
        selected_smiles = selected["SMILES"].tolist()

        novelty = compute_novelty(selected_smiles, train_df["SMILES"].tolist())
        diversity = compute_diversity(selected_smiles)

        selected_with_labels = pool_df_orig[
            pool_df_orig["SMILES"].isin(selected_smiles)
        ].drop_duplicates("SMILES")

        test_metrics = compute_test_metrics(models, test_df, cache_file)
        auroc = test_metrics["auroc"]
        hit_rate = test_metrics["hit_rate"]
        auprc = test_metrics["auprc"]
        true_hit_rate = float(selected_with_labels["Y"].mean())
        selected_pos = int(selected_with_labels["Y"].sum())

        train_df = pd.concat([train_df, selected_with_labels], ignore_index=True)
        pool_df = pool_df_orig[
            ~pool_df_orig["SMILES"].isin(train_df["SMILES"])
        ].reset_index(drop=True)

        print(f"\n  test_auroc:     {auroc:.6f}")
        print(f"  test_auprc:     {auprc:.6f}")
        print(f"  hit_rate:       {hit_rate:.6f}")
        print(f"  true_hit_rate:  {true_hit_rate:.6f}")
        print(f"  selected_pos:   {selected_pos}")
        print(f"  novelty:        {novelty:.6f}")
        print(f"  diversity:      {diversity:.6f}")
        print(f"  train_size:     {len(train_df)}")
        print(f"  pool_size:      {len(pool_df)}")
        print(f"  selected:       {len(selected)}")
        print(f"  iteration:      {iteration}")
        print(
            f"\n  FINAL_METRICS iter={iteration}: test_auroc={auroc:.6f} test_auprc={auprc:.6f} hit_rate={hit_rate:.6f}"
        )


if __name__ == "__main__":
    main()

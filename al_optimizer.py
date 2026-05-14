import os
import sys
import json
import argparse
import time
import hashlib
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

W_INHIBITION = 5.0
W_UNCERTAINTY = 0.0
W_NOVELTY = 1.0
W_DIVERSITY = 0.3
SELECTION_SIZE = 1000
VALIDATION_FRAC = 0.1
BATCH_DIVERSE = True

ENSEMBLE_SIZE = 3
EPOCHS = 15
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.4
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 3.0


def oversample_positives(df, target_ratio=0.1):
    pos = df[df["Y"] == 1]
    neg = df[df["Y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return df
    n_pos_target = int(len(neg) * target_ratio / (1 - target_ratio))
    n_pos_target = max(n_pos_target, len(pos))
    n_repeat = (n_pos_target // len(pos)) + 1
    pos_oversampled = pd.concat([pos] * n_repeat, ignore_index=True).iloc[:n_pos_target]
    return (
        pd.concat([neg, pos_oversampled], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def train_ensemble(train_df, val_df, test_df, cache_file, n_members=ENSEMBLE_SIZE):
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from minimol import Minimol

    models = []
    for i in range(n_members):
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)

        encoder_model = Minimol()
        if torch.cuda.is_available():
            try:
                encoder_model = encoder_model.to("cuda")
            except Exception:
                pass
        encoder = CachedEncoder(encoder_model, cache_file)

        def make_loader(df, shuffle):
            loader = DataLoader(
                mffn.MinimolDataset(df),
                batch_size=BATCH_SIZE,
                shuffle=shuffle,
                collate_fn=mffn.make_collate_fn(encoder),
            )
            return loader

        train_loader = make_loader(train_df, shuffle=True)
        val_loader = make_loader(val_df, shuffle=False)

        model = MinimolFFNBinary(
            input_dim=512,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)

        best_val_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= max(len(train_loader.dataset), 1)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model(x)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, y, pos_weight=pos_weight
                    )
                    val_loss += loss.item() * x.size(0)
            val_loss /= max(len(val_loader.dataset), 1)

            print(
                f"  Ens[{i}] Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")

        model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
        encoder.save()
        models.append(model)
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


def compute_novelty_scores(smiles_list, train_smiles):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps_train = []
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps_train.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if not fps_train:
        return np.zeros(len(smiles_list))
    scores = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps_train)
            scores.append(1.0 - max(sims))
        else:
            scores.append(0.0)
    return np.array(scores)


def acquisition_score(mean_probs, std_probs, novelty_scores=None):
    score = W_INHIBITION * mean_probs + W_UNCERTAINTY * std_probs
    if novelty_scores is not None:
        score += W_NOVELTY * novelty_scores
    return score


def select_molecules(
    pool_df, mean_probs, std_probs, train_smiles, selection_size=SELECTION_SIZE
):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    novelty_scores = compute_novelty_scores(pool_df["SMILES"].tolist(), train_smiles)
    scores = acquisition_score(mean_probs, std_probs, novelty_scores)

    df = pool_df.copy()
    df["_score"] = scores
    df = df.sort_values("_score", ascending=False)

    if not BATCH_DIVERSE or selection_size >= len(df):
        return df.head(selection_size).drop(columns=["_score"])

    pool_smiles = df["SMILES"].tolist()
    fps = []
    for s in pool_smiles:
        mol = Chem.MolFromSmiles(s)
        fps.append(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
        )

    scores_arr = df["_score"].values
    top_n = min(5000, len(df))
    candidate_indices = list(range(top_n))
    selected = []
    used_mask = [False] * top_n

    for _ in range(min(selection_size, top_n)):
        best_adj = -float("inf")
        best_idx = -1
        for ci in candidate_indices:
            if used_mask[ci]:
                continue
            if fps[ci] is None:
                continue
            penalty = 0.0
            if selected:
                sims = [
                    DataStructs.TanimotoSimilarity(fps[ci], fps[s])
                    for s in selected
                    if fps[s] is not None
                ]
                penalty = W_DIVERSITY * (max(sims) if sims else 0.0)
            adj = scores_arr[ci] - penalty
            if adj > best_adj:
                best_adj = adj
                best_idx = ci
        if best_idx < 0:
            break
        selected.append(best_idx)
        used_mask[best_idx] = True

    return df.iloc[selected].drop(columns=["_score"])


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
        train_inner_aug = oversample_positives(train_inner, target_ratio=0.2)
        print(
            f"  Augmented train: {len(train_inner_aug)} (pos={train_inner_aug.Y.sum()})"
        )

        cache_file = os.path.join(args.cache_dir, f"iter{iteration}")

        t0 = time.time()
        models = train_ensemble(train_inner_aug, val_df, test_df, cache_file)
        train_time = time.time() - t0
        print(f"  Train: {train_time:.1f}s")

        encoder = make_encoder(train_inner, cache_file)
        pool_smiles = pool_df["SMILES"].tolist()
        t1 = time.time()
        mean_probs, std_probs, _ = predict_ensemble(models, pool_smiles, encoder)
        pred_time = time.time() - t1
        print(f"  Predict: {pred_time:.1f}s")

        selected = select_molecules(
            pool_df,
            mean_probs,
            std_probs,
            train_inner["SMILES"].tolist(),
            SELECTION_SIZE,
        )
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

    train_df.to_csv(args.train, index=False)
    pool_df.to_csv(args.pool, index=False)
    print(f"Saved updated train ({len(train_df)}) and pool ({len(pool_df)}) CSVs.")


if __name__ == "__main__":
    main()

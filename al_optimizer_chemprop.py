#!/usr/bin/env python
"""
al_optimizer_chemprop.py — DMPNN-based active learning optimizer.

Replaces MiniMol+FFN with Chemprop DMPNN (message-passing neural network)
for molecular property prediction. Uses Dirichlet evidential loss for
built-in uncertainty estimation.

The agent modifies acquisition weights and model hyperparameters in this file.

Usage:
    python al_optimizer_chemprop.py \\
        --train data/train_df_chemprop.csv \\
        --pool data/pool_df_chemprop.csv \\
        --test data/test_df_chemprop.csv \\
        --iters 10
"""

import os
import sys
import argparse
import time
import csv
import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Chemprop in-process setup ────────────────────────────────────────────────
CHEMPROP_DIR = os.environ.get(
    "CHEMPROP_DIR",
    "/n/data1/hms/dbmi/farhat/aryan/AL/narrow_lyme_antibiotic/active_learning/run/chemprop",
)
sys.path.insert(0, CHEMPROP_DIR)

# Patch wandb to prevent hangs in subprocess/script context
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
try:
    import wandb
    wandb.init = lambda *a, **kw: type("DummyWandb", (), {"log": lambda *a, **kw: None})()
    wandb.log = lambda *a, **kw: None
except ImportError:
    pass

# Now import Chemprop modules
from chemprop.args import TrainArgs, PredictArgs
from chemprop.train.cross_validate import cross_validate
from chemprop.train.run_training import run_training
from chemprop.train.make_predictions import make_predictions as chemprop_make_predictions

# ── Acquisition weights (agent modifies these) ─────────────────────────────
W_INHIBITION = 0.5
W_UNCERTAINTY = 1.0
W_NOVELTY = 0.3
W_DIVERSITY = 0.2
BATCH_DIVERSE = False
SELECTION_SIZE = 1000
VALIDATION_FRAC = 0.1

# ── Model hyperparameters (agent modifies these) ───────────────────────────
NUM_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 64
HIDDEN_SIZE = 300
DEPTH = 3


def train_chemprop(train_csv: str, save_dir: str, epochs: int = EPOCHS) -> str:
    """Train Chemprop DMPNN in-process via cross_validate(). Returns model dir."""
    train_csv = os.path.abspath(train_csv)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Parse args the same way train.py does
    raw_args = [
        "--data_path", train_csv,
        "--smiles_columns", "SMILES",
        "--target_columns", "Y",
        "--dataset_type", "classification",
        "--save_dir", save_dir,
        "--split_type", "scaffold_balanced",
        "--num_folds", str(NUM_FOLDS),
        "--metric", "auc",
        "--extra_metrics", "binary_cross_entropy", "prc-auc",
        "--loss_function", "dirichlet",
        "--evidential_regularization", "0.2",
        "--class_balance",
        "--epochs", str(epochs),
        "--quiet",
    ]
    args = TrainArgs().parse_args(raw_args)

    print(f"  Training DMPNN ({NUM_FOLDS}-fold CV, {epochs} epochs, Dirichlet loss)...")
    print(f"  Data: {train_csv} ({args.num_tasks} task(s))")
    t0 = time.time()

    mean_score, std_score = cross_validate(args=args, train_func=run_training)

    elapsed = time.time() - t0
    print(f"  CV complete: mean {args.metric}={mean_score:.4f} +/- {std_score:.4f} ({elapsed:.1f}s)")

    return save_dir


def predict_chemprop(model_dir: str, smiles_csv: str, output_csv: str) -> str:
    """Run Chemprop prediction in-process. Returns path to predictions CSV."""
    model_dir = os.path.abspath(model_dir)
    smiles_csv = os.path.abspath(smiles_csv)
    output_csv = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Find checkpoints
    checkpoints = sorted(glob.glob(os.path.join(model_dir, "**", "model.pt"), recursive=True))
    if not checkpoints:
        raise FileNotFoundError(f"No model.pt found under {model_dir}")

    raw_args = [
        "--test_path", smiles_csv,
        "--smiles_columns", "SMILES",
        "--checkpoint_paths"] + checkpoints + [
        "--preds_path", output_csv,
        "--dataset_type", "classification",
        "--uncertainty_method", "dirichlet",
        "--loss_function", "dirichlet",
    ]
    args = PredictArgs().parse_args(raw_args)

    print(f"  Predicting on {smiles_csv} ({len(checkpoints)} checkpoints)...")
    t0 = time.time()

    chemprop_make_predictions(args=args)

    elapsed = time.time() - t0
    print(f"  Prediction saved to {output_csv} ({elapsed:.1f}s)")
    return output_csv


def load_predictions(pred_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions CSV, return (mean_probs, uncertainty) arrays."""
    df = pd.read_csv(pred_csv)
    prob_col = "Y"
    unc_col = None
    for c in df.columns:
        if "uncertainty" in c.lower() or "dirichlet" in c.lower() or "unc" in c.lower():
            unc_col = c
            break
    if prob_col not in df.columns:
        for c in df.columns:
            if c != "SMILES" and "uncertainty" not in c.lower():
                prob_col = c
                break
    probs = df[prob_col].values.astype(np.float64)
    unc = df[unc_col].values.astype(np.float64) if unc_col else np.zeros_like(probs)
    return probs, unc


def compute_test_metrics(pred_csv: str, test_df: pd.DataFrame) -> dict:
    """Compute AUROC/AUPRC/hit_rate from prediction CSV."""
    pred_df = pd.read_csv(pred_csv)
    merged = pred_df.merge(test_df[["SMILES", "Y"]], on="SMILES", how="inner")
    if len(merged) == 0:
        return {"auroc": 0.5, "auprc": 0.0, "hit_rate": 0.0}
    y_true = merged["Y"].values.astype(float)
    prob_col = "Y"
    if prob_col not in pred_df.columns:
        for c in pred_df.columns:
            if c != "SMILES" and "uncertainty" not in c.lower():
                prob_col = c
                break
    y_prob = merged[prob_col].values.astype(float)
    if len(np.unique(y_true)) < 2:
        auroc = 0.5
        auprc = 0.0
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    y_pred = (y_prob >= 0.5).astype(float)
    hit_rate = float(y_pred.mean())
    return {"auroc": auroc, "auprc": auprc, "hit_rate": hit_rate}


def acquisition_score(
    mean_probs: np.ndarray,
    uncertainty: np.ndarray,
    novelty_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score each molecule for acquisition. Higher -> selected first."""
    score = W_INHIBITION * mean_probs + W_UNCERTAINTY * uncertainty
    if novelty_scores is not None:
        score += W_NOVELTY * novelty_scores
    return score


def compute_novelty(pool_smiles: List[str], train_smiles: List[str]) -> np.ndarray:
    """Compute novelty as 1 - max Tanimoto similarity to training set."""
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


def compute_diversity(smiles_list: List[str]) -> float:
    """Compute internal diversity (1 - mean pairwise Tanimoto similarity)."""
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


def select_molecules(
    pool_df: pd.DataFrame,
    mean_probs: np.ndarray,
    uncertainty: np.ndarray,
    train_smiles: List[str],
    selection_size: int = SELECTION_SIZE,
    diverse: bool = False,
) -> pd.DataFrame:
    """Select molecules by acquisition score.

    When diverse=True, uses greedy diverse selection: iteratively picks the
    highest-scoring molecule, then downweights remaining candidates by their
    Tanimoto similarity to the selected molecule.
    """
    novelty_scores = compute_novelty(pool_df["SMILES"].tolist(), train_smiles)
    scores = acquisition_score(mean_probs, uncertainty, novelty_scores)

    if not diverse:
        df = pool_df.copy()
        df["_score"] = scores
        df = df.sort_values("_score", ascending=False)
        return df.head(selection_size).drop(columns=["_score"])

    pool_smiles = pool_df["SMILES"].tolist()
    pool_fps = []
    for s in pool_smiles:
        mol = Chem.MolFromSmiles(s)
        pool_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None)

    remaining = list(range(len(pool_smiles)))
    selected_indices = []
    current_scores = scores.copy()

    for _ in range(min(selection_size, len(remaining))):
        best_idx = max(remaining, key=lambda i: current_scores[i])
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        if remaining and pool_fps[best_idx] is not None:
            for i in remaining:
                if pool_fps[i] is not None:
                    sim = DataStructs.TanimotoSimilarity(pool_fps[best_idx], pool_fps[i])
                    current_scores[i] *= (1.0 - W_DIVERSITY * sim)

    return pool_df.iloc[selected_indices].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="AL Optimizer — Chemprop DMPNN")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--pool", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--work_dir", type=str, default="al_chemprop_runs")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    train_df = pd.read_csv(args.train)
    pool_df = pd.read_csv(args.pool)
    test_df = pd.read_csv(args.test)
    pool_df_orig = pool_df.copy()

    print(f"Train: {len(train_df)} (pos={train_df.Y.sum()}) | "
          f"Pool: {len(pool_df)} | Test: {len(test_df)}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    for iteration in range(args.iters):
        print(f"\n{'=' * 60}")
        print(f"  Iteration {iteration}")
        print(f"{'=' * 60}")
        print(f"  Train size: {len(train_df)} (pos={train_df.Y.sum()})")
        print(f"  Pool size:  {len(pool_df)}")

        iter_dir = os.path.join(args.work_dir, f"iter_{iteration:02d}")

        # Train DMPNN in-process
        model_dir = train_chemprop(args.train, os.path.join(iter_dir, "model"))

        # Evaluate on test set
        test_pred_csv = os.path.join(iter_dir, "test_preds.csv")
        predict_chemprop(model_dir, args.test, test_pred_csv)
        test_metrics = compute_test_metrics(test_pred_csv, test_df)
        auroc = test_metrics["auroc"]
        auprc = test_metrics["auprc"]
        hit_rate = test_metrics["hit_rate"]

        # Predict on pool
        pool_pred_csv = os.path.join(iter_dir, "pool_preds.csv")
        predict_chemprop(model_dir, args.pool, pool_pred_csv)
        mean_probs, uncertainty = load_predictions(pool_pred_csv)

        # Select molecules
        selected = select_molecules(
            pool_df, mean_probs, uncertainty,
            train_df["SMILES"].tolist(), SELECTION_SIZE,
            diverse=BATCH_DIVERSE,
        )
        selected_smiles = selected["SMILES"].tolist()

        novelty = compute_novelty(selected_smiles, train_df["SMILES"].tolist())
        diversity = compute_diversity(selected_smiles)

        # Reveal labels ONLY after selection
        selected_with_labels = pool_df_orig[
            pool_df_orig["SMILES"].isin(selected_smiles)
        ].drop_duplicates("SMILES")

        true_hit_rate = float(selected_with_labels["Y"].mean())
        selected_pos = int(selected_with_labels["Y"].sum())

        # Update datasets
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
            f"\n  FINAL_METRICS iter={iteration}: "
            f"test_auroc={auroc:.6f} test_auprc={auprc:.6f} "
            f"hit_rate={hit_rate:.6f}"
        )

        train_df.to_csv(args.train, index=False)
        pool_df.to_csv(args.pool, index=False)
        print(f"  Saved updated train ({len(train_df)}) and pool ({len(pool_df)})")


if __name__ == "__main__":
    main()

import os
import pickle
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from minimol import Minimol


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CachedEncoder:
    def __init__(self, encoder, cache_file: str):
        self.encoder = encoder
        self.cache_file = cache_file
        self.cache: Dict[str, torch.Tensor] = {}

        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    self.cache = obj
                    print(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                print(f"Warning: failed to load cache: {e}")
                self.cache = {}

    def encode(self, smiles_list: List[str], chunk_size: int = 50000) -> torch.Tensor:
        missing = [s for s in smiles_list if s not in self.cache]

        if missing:
            n = len(missing)
            for i in range(0, n, chunk_size):
                chunk = missing[i:i + chunk_size]
                print(f"  Encoding {i + len(chunk):,} / {n:,} fingerprints...", flush=True)
                embs = self.encoder(chunk)
                for s, e in zip(chunk, embs):
                    self.cache[s] = e.detach().cpu()

        # Avoid giant torch.stack — batch them too
        if len(smiles_list) <= chunk_size:
            return torch.stack([torch.as_tensor(self.cache[s]) for s in smiles_list], dim=0)
        result = []
        for i in range(0, len(smiles_list), chunk_size):
            chunk = smiles_list[i:i + chunk_size]
            result.append(torch.stack([torch.as_tensor(self.cache[s]) for s in chunk], dim=0))
        return torch.cat(result, dim=0)

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: failed to save cache: {e}")


class MinimolDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        if "SMILES" not in df.columns or "Y" not in df.columns:
            raise ValueError("Each dataframe must contain 'SMILES' and 'Y' columns.")
        self.smiles = df["SMILES"].astype(str).tolist()
        self.y = df["Y"].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        return self.smiles[idx], self.y[idx]


def make_collate_fn(encoder: CachedEncoder):
    def collate_fn(batch):
        smiles = [x[0] for x in batch]
        y = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        x = encoder.encode(smiles)
        return x, y
    return collate_fn


class MinimolFFNBinary(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        logits = self.head(h).squeeze(-1)
        return logits


class MinimolFFNTrainer:
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cache_file: str = "cache.pkl",
        batch_size: int = 128,
        lr: float = 1e-3,
        epochs: int = 20,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        encoder_model = Minimol()
        if torch.cuda.is_available():
            try:
                encoder_model = encoder_model.to("cuda")
            except Exception:
                pass

        self.encoder = CachedEncoder(encoder_model, cache_file)
        collate_fn = make_collate_fn(self.encoder)

        self.train_loader = DataLoader(
            MinimolDataset(train_df),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            MinimolDataset(val_df),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            MinimolDataset(test_df),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        self.model = MinimolFFNBinary(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        total_n = 0

        for x, y in self.train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        return total_loss / max(total_n, 1)

    def _eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_n = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                logits = self.model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_n += bs

        return total_loss / max(total_n, 1)

    def train(self):
        best_val_loss = float("inf")
        best_model_path = "best_model.pt"

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._eval_loss(self.val_loader)

            print(f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)

        self.model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_y = []
        all_probs = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs.tolist())
                all_y.extend(y.numpy().tolist())

        y_true = np.array(all_y)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= 0.5).astype(np.float32)

        acc = float((y_pred == y_true).mean())

        auroc = None
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) == 2:
                auroc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

        return {
            "acc": acc,
            "auroc": auroc,
        }

    def run(self):
        self.train()
        self.encoder.save()
        val_metrics = self.evaluate(self.val_loader)
        test_metrics = self.evaluate(self.test_loader)

        print("Validation metrics:", val_metrics)
        print("Test metrics:", test_metrics)

        return val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--cache_file", type=str, default="cache.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--input_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_batch_norm", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    trainer = MinimolFFNTrainer(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cache_file=args.cache_file,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=not args.no_batch_norm,
    )

    trainer.run()


if __name__ == "__main__":
    main()
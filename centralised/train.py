#!/usr/bin/env python

## Imports
import warnings, os, json, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from codecarbon import EmissionsTracker
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import *
from .utils import set_seed, free_memory, load_data
from torch.nn import functional as F


class MicrobiomeTabularConvNet(nn.Module):
    def __init__(self, n_classes=1):
        super(MicrobiomeTabularConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(19360, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, n_classes)
        #self.dropout = nn.Dropout(0.8)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `fold` column (StratifiedKFold)."""
    df = df.copy()
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(skf.split(df, df[TARGET_COL])):
        df.loc[val_idx, "fold"] = fold
    return df


def main(args):
    set_seed(SEED, use_cuda=torch.cuda.is_available())

    train_df, test_df, train_feat, test_feat = load_data()

    train_df["filename"] = train_df["filename"].str.replace(".mgb", ".fastq")
    test_df["filename"]  = test_df["filename"].str.replace(".mgb", ".fastq")

    train_feat[TARGET_COL] = train_feat["file"].map(
        dict(zip(train_df["filename"], train_df["SampleType"]))
    )
    train_feat[TARGET_COL] = train_feat[TARGET_COL].map(TARGET_MAP)


    meta_cols = {"ID", "file", "fold", TARGET_COL}
    cont_features = [c for c in train_feat.columns if c not in meta_cols]

   
    scaler = StandardScaler()
    train_feat[cont_features] = scaler.fit_transform(train_feat[cont_features])
    test_feat[cont_features] = scaler.transform(test_feat[cont_features])

   
    train_feat = make_folds(train_feat)

 
    oof_cols = [f"oof_{lbl}" for lbl in TARGET_LABELS]
    train_feat[oof_cols] = 0.0

    # ------------------------------------------------------------------
    # Training loop over folds
    # ------------------------------------------------------------------
    for fold in range(NUM_FOLDS):
        free_memory()
        print("\n" + "#" * 80)
        print(f"🚂 Training fold {fold}/{NUM_FOLDS - 1}")

        tr_idx = train_feat["fold"] != fold
        val_idx = train_feat["fold"] == fold

        X_tr = train_feat.loc[tr_idx, cont_features].values.astype(np.float32)
        y_tr = train_feat.loc[tr_idx, TARGET_COL].values
        X_val = train_feat.loc[val_idx, cont_features].values.astype(np.float32)
        y_val = train_feat.loc[val_idx, TARGET_COL].values

        train_ds = TensorDataset(torch.from_numpy(X_tr),
                                 torch.from_numpy(y_tr))
        val_ds   = TensorDataset(torch.from_numpy(X_val),
                                 torch.from_numpy(y_val))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, pin_memory=True)

        model = MicrobiomeTabularConvNet(n_classes=len(TARGET_LABELS)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_loss = np.inf
        best_state = None

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            tr_loss = 0.0
            with EmissionsTracker(project_name="centralised_training") as trk:
                trk.start()
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                trk.stop()
            tr_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            correct = 0
            all_probs = []
            with torch.no_grad():
                with EmissionsTracker(project_name="centralised_training") as trk:
                    trk.start()
                    for xb, yb in val_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item()
                        preds = logits.argmax(dim=1)
                        correct += (preds == yb).sum().item()
                        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                    trk.stop()
            val_loss /= len(val_loader)
            val_acc = correct / len(val_idx)

            print(
                f"[fold {fold}] Epoch {epoch:02d} | "
                f"train loss {tr_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"val acc {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "model": model.state_dict(),
                    "probs": np.concatenate(all_probs, axis=0),
                }
                torch.save(best_state,
                           f"{args.ckpt_dir}/fold{fold}_best.pth")

        train_feat.loc[val_idx, oof_cols] = best_state["probs"]

        # clean up GPU memory before next fold
        del model, optimizer, train_loader, val_loader
        free_memory()

  
    oof_probs = train_feat[oof_cols].values
    overall_logloss = log_loss(train_feat[TARGET_COL], oof_probs)
    print("\n⭐ Overall OOF LogLoss:", overall_logloss)

   

    meta = {
        "cont_features": cont_features,
        "target_map": TARGET_MAP,
        "target_labels": TARGET_LABELS,
    }
    with open(f"{args.out_dir}/meta.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    print("✅ Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Model on MPEG‑Kmer data."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="Folder where per‑fold .pth files will be stored.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Folder for train‑with‑OOF CSV and meta‑json.",
    )
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    main(args)

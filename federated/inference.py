#!/usr/bin/env python
"""Generate predictions for the test set using saved global checkpoints."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import DEVICE, CHECKPOINT_DIR
from model import MicrobiomeTabularConvNet
from data import test_features, cont_features
from scipy.stats.mstats import gmean



X_test = test_features[cont_features].values.astype(np.float32)
test_dataset = TensorDataset(torch.from_numpy(X_test))
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


def inference(model: torch.nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main() -> None:

    ckpt_paths = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "global_round_*.pth")))
    
    model = MicrobiomeTabularConvNet(n_classes=4).to(DEVICE)

    round_probs = []
    for path in ckpt_paths:
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        probs = inference(model, test_loader, DEVICE)   # (N_test, 4)
        round_probs.append(probs)

 
    stacked = np.stack(round_probs, axis=0)          # (n_rounds, N, 4)
    final_probs = np.mean(stacked, axis=0)            # (N, 4)

    tgt_cols = ["Mouth", "Nasal", "Skin", "Stool"]
    sub_df = test_features[["file"]].copy()
    sub_df[tgt_cols] = final_probs
    sub_df["filename"] = sub_df["file"].str.replace(".fastq", "")
    sub_df = sub_df.drop(columns=["file"])

    out_path = "kmers_fl_submission_nn.csv"
    sub_df[["filename"] + tgt_cols].to_csv(out_path, index=False)
    print(f"Submission written to {out_path}")


if __name__ == "__main__":
    main()

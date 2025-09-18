#!/usr/bin/env bash
set -euo pipefail



python -m src.train \
    --ckpt_dir checkpoints \
    --out_dir artifacts


python -m src.inference \
    --ckpt_dir checkpoints \
    --meta_path artifacts/meta.json \
    --out_dir submission

echo "✅ All done! Find the submission at $(realpath submission/kmers_centralised_submission_nn.csv)"

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Data locations
# ----------------------------------------------------------------------
DATA_ROOT = Path("./MPEG")
TRAIN_CSV   = DATA_ROOT / "Train.csv"
TEST_CSV    = DATA_ROOT / "Test.csv"
TRAIN_FEAT  = DATA_ROOT / "train_features_with_kmers_new.csv"
TEST_FEAT   = DATA_ROOT / "test_features_with_kmers_new.csv"

# ----------------------------------------------------------------------
# Global training options
# ----------------------------------------------------------------------
NUM_CLIENTS = 4
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

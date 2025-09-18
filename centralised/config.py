# --------------------------------------------------------------
# Central place for paths, constants and hyper‑parameters
# --------------------------------------------------------------
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]          # project root
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "Train.csv"
TEST_CSV  = DATA_DIR / "Test.csv"
TRAIN_SUBJ_CSV = DATA_DIR / "Train_Subjects.csv"
TRAIN_FEAT_CSV = DATA_DIR / "train_features_with_kmers_new.csv"
TEST_FEAT_CSV  = DATA_DIR / "test_features_with_kmers_new.csv"

#  Model & training
NUM_FOLDS = 10
BATCH_SIZE = 256
LR = 4e-4
MAX_EPOCHS = 100
SEED = 42
DEVICE = "cuda" if (torch.cuda.is_available()) else "cpu"

# Target columns
TARGET_COL = "SampleType"
TARGET_MAP = {"Mouth":0, "Nasal":1, "Skin":2, "Stool":3}
TARGET_LABELS = ["Mouth", "Nasal", "Skin", "Stool"]

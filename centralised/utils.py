# --------------------------------------------------------------
# Reusable helpers
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import gc, time

def set_seed(seed: int, use_cuda: bool = True):
    """Make experiments reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if use_cuda else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def free_memory(sleep: float = 0.1):
    """Best‑effort clean‑up of Python & CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(sleep)

def load_data():
    """Read all CSVs used by the pipeline."""
    from . import config as cfg
    train_df   = pd.read_csv(cfg.TRAIN_CSV)
    test_df    = pd.read_csv(cfg.TEST_CSV)
    train_feat = pd.read_csv(cfg.TRAIN_FEAT_CSV)
    test_feat  = pd.read_csv(cfg.TEST_FEAT_CSV)
    return train_df, test_df, train_feat, test_feat

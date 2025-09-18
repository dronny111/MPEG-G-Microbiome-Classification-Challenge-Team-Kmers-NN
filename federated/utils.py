import gc
import time
import numpy as np
import torch
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)
from torch import nn
from typing import List, Tuple
from torch.utils.data import TensorDataset

import pandas as pd

def get_parameters(net: nn.Module) -> Parameters:
    """Serialize all tensors in `net.state_dict()` to a Flower Parameters object."""
    ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
    return ndarrays_to_parameters(ndarrays)


def set_parameters(net: nn.Module, parameters) -> None:
    """Load a Flower Parameters object (or a list of ndarrays) into `net`."""
    if isinstance(parameters, list):
        parameters = ndarrays_to_parameters(parameters)

    ndarrays = parameters_to_ndarrays(parameters)
    new_state = {}
    for (k, old_tensor), arr in zip(net.state_dict().items(), ndarrays):
        new_state[k] = torch.tensor(
            arr, dtype=old_tensor.dtype, device=old_tensor.device
        )
    net.load_state_dict(new_state, strict=False)


def free_memory(sleep_time: float = 0.1) -> None:
    gc.collect()
    time.sleep(sleep_time)


def df_to_tensors(
    df: pd.DataFrame,
    cont_feats: List[str],
    tgt_col: str,
) -> Tuple[TensorDataset, int]:
    """Convert a dataframe into a `TensorDataset`."""

    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[cont_feats] = df[cont_feats].fillna(0.0)

    X = torch.from_numpy(df[cont_feats].values.astype("float32"))
    y = torch.from_numpy(df[tgt_col].values.astype("int64"))
    ds = TensorDataset(X, y)
    return ds, len(ds)

import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import flwr as fl
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    Scalar,
)

from config import DEVICE
from utils import get_parameters, set_parameters, df_to_tensors
from model import MicrobiomeTabularConvNet
from data import train_features, cont_features


class MicrobeClient(fl.client.NumPyClient):
    """Virtual client used in the simulation."""

    def __init__(
        self,
        global_model: nn.Module,
        train_dataset,
        val_dataset,
        batch_size: int = 256,
        local_epochs: int = 5,
    ):
        self.model = copy.deepcopy(global_model).to(DEVICE)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=4e-4)

        self.local_epochs = local_epochs


    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters, config):
        set_parameters(self.model, parameters)


    def fit(self, parameters, config):
        # Load the latest global model
        self.set_parameters(parameters, config)

        self.model.train()
        epoch_losses = []

        for _ in range(self.local_epochs):
            epoch_loss = 0.0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_losses.append(epoch_loss / len(self.train_loader))

        updated = get_parameters(self.model)

        # Return list of ndarrays (Flower expects this)
        ndarrays = updated if isinstance(updated, list) else parameters_to_ndarrays(updated)
        return ndarrays, len(self.train_loader), {"loss": float(np.mean(epoch_losses))}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = self.model(xb)
                loss_sum += self.criterion(logits, yb).item()
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return loss_sum / len(self.val_loader), total, {
            "accuracy": correct / total if total else 0.0
        }


def client_fn(context: fl.common.Context) -> fl.client.Client:
    """Factory that creates a new client for each simulated participant."""

    train_split, val_split = train_test_split(
        train_features,
        test_size=0.2,
        stratify=train_features["SampleType"],
        random_state=42,
        shuffle=True,
    )

    train_ds, _ = df_to_tensors(train_split, cont_features, "SampleType")
    val_ds,   _ = df_to_tensors(val_split,   cont_features, "SampleType")

    global_model = MicrobiomeTabularConvNet( n_classes=4).to(DEVICE)

    batch_size   = context.run_config.get("batch-size", 256)
    local_epochs = context.run_config.get("local_epochs", 100)

    return MicrobeClient(
        global_model=global_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
        local_epochs=local_epochs,
    ).to_client()

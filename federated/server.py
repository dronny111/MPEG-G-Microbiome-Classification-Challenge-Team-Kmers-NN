import os
import torch
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from typing import Dict, Tuple, Optional
from config import DEVICE, CHECKPOINT_DIR, NUM_CLIENTS
from utils import get_parameters, set_parameters
from model import MicrobiomeTabularConvNet
from data import train_features, cont_features
from sklearn.model_selection import train_test_split
from typing import List

def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    """Global validation performed after each round."""
    # Re‑create the model & load the received parameters
    net = MicrobiomeTabularConvNet(n_classes=4).to(DEVICE)

    set_parameters(net, ndarrays_to_parameters(parameters))

    # Build the same validation split used by the clients
    _, val_split = train_test_split(
        train_features,
        test_size=0.2,
        stratify=train_features["SampleType"],
        random_state=42,
        shuffle=True,
    )
    from utils import df_to_tensors
    val_ds, _ = df_to_tensors(val_split, cont_features, "SampleType")
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = net(xb)
            loss_sum += criterion(logits, yb).item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    loss = loss_sum / len(val_loader)
    acc = correct / total if total else 0.0
    print(f"[SERVER] round {server_round} – loss {loss:.4f} – acc {acc:.4f}")
    return loss, {"accuracy": acc}


def get_initial_parameters() -> fl.common.Parameters:
    """Create a fresh model and return its parameters (used for the first round)."""
    model = MicrobiomeTabularConvNet(
        input_dim=len(cont_features), n_classes=4
    )
    return get_parameters(model)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg that also saves a PyTorch checkpoint after every aggregation."""
    def __init__(self, checkpoint_dir: str = "./checkpoints", **kwargs):
        super().__init__(**kwargs)
        self.ckpt_dir = checkpoint_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def aggregate_fit(self, rnd, results, failures):
        agg_params, metrics = super().aggregate_fit(rnd, results, failures)

        if agg_params is None:
            return agg_params, metrics

        # Save a checkpoint for the aggregated global model
        tmp = MicrobiomeTabularConvNet(
            input_dim=len(cont_features), n_classes=4
        )
        set_parameters(tmp, agg_params)
        ckpt_path = os.path.join(self.ckpt_dir, f"global_round_{rnd}.pth")
        torch.save(tmp.state_dict(), ckpt_path)
        print(f"[SERVER] Saved checkpoint {ckpt_path}")

        return agg_params, metrics


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """
    Example of a custom metric aggregation (you can keep it or drop it).
    The Whisper example uses a similar helper.
    """
    accuracies = [
        num_examples * m["accuracy"]
        for num_examples, m in metrics
        if "accuracy" in m
    ]
    losses = [
        num_examples * m["loss"]
        for num_examples, m in metrics
        if "loss" in m
    ]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "train_accuracy": sum(accuracies) / sum(examples) if examples else 0.0,
        "train_loss": sum(losses) / sum(examples) if examples else 0.0,
    }

def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    """Factory that creates the Flower ServerApp."""
    num_rounds = int(context.run_config.get("num_rounds", 5))

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.1,
        min_fit_clients=context.run_config.get("num_clients", NUM_CLIENTS),
        min_evaluate_clients=max(
            1, context.run_config.get("num_clients", NUM_CLIENTS) // 2
        ),
        min_available_clients=context.run_config.get("num_clients", NUM_CLIENTS),
        on_fit_config_fn=lambda rnd: {"local_epochs": 100},
        evaluate_fn=evaluate,
        initial_parameters=get_initial_parameters(),
        # evaluate_metrics_aggregation_fn=weighted_average,
        # fit_metrics_aggregation_fn=weighted_average,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    config = fl.server.ServerConfig(num_rounds=num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)



server_app = fl.server.ServerApp(server_fn=server_fn)

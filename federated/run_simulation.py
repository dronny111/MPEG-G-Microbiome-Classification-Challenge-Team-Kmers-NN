#!/usr/bin/env python
"""Run the Flower simulation using the modules defined in this repository."""

import flwr as fl
from client import client_fn
from server import server_app
from config import NUM_CLIENTS, DEVICE
import os

def main() -> None:
    # Backend configuration – 1 CPU per client, GPU if available
    backend_cfg = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    if DEVICE == "cuda":
        backend_cfg["client_resources"]["num_gpus"] = 1.0

    print("[INFO] Starting simulation …")
    history = fl.simulation.run_simulation(
        server_app=server_app,
        client_app=fl.client.ClientApp(client_fn=client_fn),
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_cfg,
    )
    print("[INFO] Simulation finished.")
    print("Checkpoints saved in:", os.listdir("./checkpoints"))


if __name__ == "__main__":
    main()

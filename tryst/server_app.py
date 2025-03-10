"""Federated learning server application for Heart Disease Model using Flower/PyTorch."""

from typing import List, Tuple, Optional, Union
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import Metrics, parameters_to_ndarrays, ndarrays_to_parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, ServerApp, ServerAppComponents

# Import your model and utility functions from your task file.
from tryst.task import Net, get_weights, set_weights

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using a weighted average (weighted by number of examples)."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0}

class SaveModelStrategy(FedAvg):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint."""
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            # Convert `list[np.ndarray]` to PyTorch `state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Ensure that the 'tryst' directory exists
            os.makedirs("tryst", exist_ok=True)
            # Save the model to disk
            torch.save(self.model.state_dict(), f"tryst/model_round_{server_round}.pth")
            print(f"Model weights saved to 'tryst/model_round_{server_round}.pth'")
        
        return aggregated_parameters, aggregated_metrics

def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """Configure server behavior."""
    print("Server starting...")

    # Read configuration values (with defaults)
    num_rounds = context.run_config.get("num-server-rounds", 3)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    # Determine model input size from heart.csv (assumes last column is 'target')
    df = pd.read_csv("tryst/heart.csv")
    num_features = df.shape[1] - 1  # Exclude target column

    # Initialize the heart disease model using the determined input size
    model = Net(input_dim=num_features)
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the custom strategy with the model instance
    strategy = SaveModelStrategy(
        model=model,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )

    # Create server configuration
    config = ServerConfig(num_rounds=num_rounds)

    print("Server configuration:")
    print(f"- Number of rounds: {num_rounds}")
    print(f"- Fraction fit: {fraction_fit}")
    print(f"- Fraction evaluate: {fraction_evaluate}")
    print(f"- Model input size (number of features): {num_features}")
    print("Server ready, waiting for clients...")

    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp instance
app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("Run this application using the command 'flwr run .'")

"""Federated learning server application for Heart Disease Model using Flower/PyTorch."""

from typing import List, Tuple
import pandas as pd

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
# Import your model and utility functions from your task file.
# Ensure that your task file (e.g., heart_task.py) defines:
# - Net(input_dim) -> your model
# - get_weights(net) -> returns a list of numpy arrays
from tryst.task import Net, get_weights


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using a weighted average (weighted by number of examples)."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0}


def server_fn(context: Context):
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

    # Define the federated learning strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
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

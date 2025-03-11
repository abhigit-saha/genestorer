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
import glob
import matplotlib.pyplot as plt
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
    print("Server starting...")

    # Read configuration values (with defaults)
    num_rounds = context.run_config.get("num-server-rounds", 3)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    # Determine model input size from heart.csv (assumes last column is 'target')
    df = pd.read_csv("tryst/heart.csv")
    num_features = df.shape[1] - 1  # Assuming last column is target

    # --- Initialize model ---
    model = Net(input_dim=num_features)

    # --- Load the latest checkpoint if available ---
    list_of_files = glob.glob("tryst/model_round_*.pth")
    if list_of_files:
        latest_round_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading pre-trained model from: {latest_round_file}")
        try:
            state_dict = torch.load(latest_round_file)
            model.load_state_dict(state_dict)
            print("Model loaded successfully")
            
            # Verify model parameters to ensure they were loaded properly
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.data.mean().item():.4f} (mean), {param.data.std().item():.4f} (std)")
            
            # Set model to evaluation mode
            model.eval()

            # --- Prepare input data ---
            # Normalize the data similar to how it's done in HeartDataset
            X_test = df.iloc[:, :-1].values.astype(np.float32)
            y_true = df.iloc[:, -1].values.astype(np.float32)
            
            # Normalize features using same approach as in HeartDataset
            X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
            
            # Convert to tensor
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

            # --- Make predictions with fixed logic ---
            with torch.no_grad():
                # Get raw logits from the model
                logits = model(X_test_tensor)
                
                # Print raw output statistics for debugging
                print(f"Raw output stats - Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(logits).squeeze()
                predicted_classes = (probabilities > 0.5).int().numpy()
            
            # Calculate accuracy and other metrics
            accuracy = np.mean(predicted_classes == y_true)
            tp = np.sum((predicted_classes == 1) & (y_true == 1))
            tn = np.sum((predicted_classes == 0) & (y_true == 0))
            fp = np.sum((predicted_classes == 1) & (y_true == 0))
            fn = np.sum((predicted_classes == 0) & (y_true == 1))
            
            # Calculate precision, recall, and F1 if possible
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print("\n--- Model Evaluation on Test Data ---")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Show distribution of predictions
            unique_preds, counts = np.unique(predicted_classes, return_counts=True)
            print("\nPrediction distribution:")
            for pred, count in zip(unique_preds, counts):
                print(f"Class {pred}: {count} samples ({count/len(predicted_classes)*100:.1f}%)")
            
            # Show distribution of actual labels
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            print("\nTrue label distribution:")
            for label, count in zip(unique_true, counts_true):
                print(f"Class {int(label)}: {count} samples ({count/len(y_true)*100:.1f}%)")
            
            # Show detailed sample predictions with both logits and probabilities
            print("\nDetailed Sample Predictions (first 10):")
            for i in range(min(10, len(y_true))):
                print(f"Sample {i}: True={int(y_true[i])}, Predicted={predicted_classes[i]}, "
                      f"Logit={logits.squeeze()[i].item():.4f}, Probability={probabilities[i].item():.4f}")

# Assuming 'region' is a categorical column in the dataset
            if "region" in df.columns:
                regions = df["region"].values  # Extract region data
                region_probs = {}  # Dictionary to store mean probability per region
    
                for region in np.unique(regions):
                    region_indices = regions == region
                    mean_prob = probabilities[region_indices].mean().item()
                    region_probs[region] = mean_prob
    
    # Plot the data
                plt.figure(figsize=(10, 6))
                plt.bar(region_probs.keys(), region_probs.values(), color='skyblue')
                plt.xlabel("Region")
                plt.ylabel("Average Probability of Disease")
                plt.title("Probability of Disease vs Region")
                plt.xticks(rotation=45)  # Rotate region names for better visibility
                plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
                plt.savefig("tryst/probability_vs_region.png")
                print("Plot saved as 'tryst/probability_vs_region.png'")
                plt.show()
            else:
                print("No 'region' column found in dataset. Skipping plot.")

            
        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
            print("Initializing with fresh weights.")
    else:
        print("No pre-trained model found. Will initialize with fresh weights.")

    # Get initial parameters for strategy
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

app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    print("Run this application using the command 'flwr run .'")
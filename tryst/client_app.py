import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tryst.task import Net, load_data, get_weights, set_weights, test, train


class HeartClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.net.to(self.device)

    def get_parameters(self, config):
        return get_weights(self.net)

    def fit(self, parameters, config):
        # Update model parameters with the federated parameters.
        set_weights(self.net, parameters)
        
        print("\nTraining locally with federated parameters...")
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        
        # Return updated parameters and metrics.
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": float(train_loss)},
        )

    def evaluate(self, parameters, config):
        # Set the model parameters received from the server.
        set_weights(self.net, parameters)
        
        print("\nEvaluating globally aggregated model...")
        loss, accuracy = test(self.net, self.testloader, self.device)
        
        # Return loss, the number of evaluation examples, and a dictionary with evaluation metrics.
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    # Retrieve client configuration.
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 2)
    local_epochs = context.run_config.get("local-epochs", 1)
    
    print(f"\nInitializing client {partition_id+1} of {num_partitions}")
    
    # Load data partition: this function returns (trainloader, testloader, num_features)
    trainloader, testloader, num_features = load_data(partition_id, num_partitions)
    
    # Create model using the number of features determined from the dataset.
    net = Net(input_dim=num_features)
    print(f"Created heart model with input size: {num_features}")
    
    # Return a Flower client instance.
    return HeartClient(net, trainloader, testloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)

if __name__ == "__main__":
    print("Run this application using the 'flwr run .' command")

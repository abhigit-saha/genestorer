from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import pandas as pd
import numpy as np

# -------------------------
# 1. Custom Dataset Class for Heart Disease Data
# -------------------------
class HeartDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # All columns except 'target' are features
        self.X = self.data.drop('target', axis=1).values.astype(np.float32)
        self.y = self.data['target'].values.astype(np.float32).reshape(-1, 1)
        # Normalize features: (x - mean) / std
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# 2. Model Definition (MLP with Dropout)
# -------------------------
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.1)  # Dropout after first hidden layer
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)  # Dropout after second hidden layer
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)  # Dropout after third hidden layer
        self.fc4 = nn.Linear(32, 1)
        self.activation = nn.ReLU()       # Using ReLU activation
        self.sigmoid = nn.Sigmoid()         # Sigmoid for binary classification

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        x = self.activation(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x

# -------------------------
# 3. Federated Data Loader for Heart CSV
# -------------------------
def load_data(partition_id: int, num_partitions: int):
    """
    Loads the heart.csv dataset, partitions it IID among num_partitions,
    and then splits each partition into train (80%) and test (20%).
    """
    dataset = HeartDataset("tryst/heart.csv")
    total = len(dataset)
    indices = list(range(total))
    split_size = total // num_partitions
    start = partition_id * split_size
    # Make sure the last partition gets any remaining samples
    end = total if partition_id == num_partitions - 1 else (partition_id + 1) * split_size
    partition_indices = indices[start:end]
    partition_subset = Subset(dataset, partition_indices)
    
    train_size = int(0.8 * len(partition_subset))
    test_size = len(partition_subset) - train_size
    train_dataset, test_dataset = random_split(partition_subset, [train_size, test_size])
    
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    input_size = next(iter(trainloader))[0].shape[1] 
    return trainloader, testloader, input_size

# -------------------------
# 4. Federated Training Function
# -------------------------
def train(net, trainloader, epochs, device):
    """Train the model on the training set using BCELoss and Adagrad."""
    net.to(device)
    criterion = nn.BCELoss().to(device)
    # Use Adagrad as optimizer
    optimizer = torch.optim.Adagrad(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for features, labels in trainloader:
            features = torch.tensor(features).to(device)
            labels = torch.tensor(labels).to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

# -------------------------
# 5. Federated Test Function
# -------------------------
def test(net, testloader, device):
    """Validate the model on the test set using BCELoss."""
    net.to(device)
    criterion = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            features = torch.tensor(features).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item() * features.size(0)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = loss / len(testloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# -------------------------
# 6. Utility Functions for Federated Learning
# -------------------------
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

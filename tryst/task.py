from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import pandas as pd
import numpy as np


class HeartDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop('target', axis=1).values.astype(np.float32)
        self.y = self.data['target'].values.astype(np.float32).reshape(-1, 1)

        # Normalize features safely
        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        self.X = (self.X - mean) / std
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


import torch.nn.init as init

import torch.nn.init as init

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, 1)  # Last layer

        self.activation = nn.ReLU()

        # Explicitly initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)  # Explicitly set biases to zero

            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        x = self.dropout1(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout2(self.activation(self.bn2(self.fc2(x))))
        x = self.dropout3(self.activation(self.bn3(self.fc3(x))))
        x = self.fc4(x)  # No activation, as BCEWithLogitsLoss expects raw logits
        return x




def load_data(partition_id: int, num_partitions: int):
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


def train(net, trainloader, epochs, device):
    net.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item() * features.size(0)

        avg_epoch_loss = epoch_loss / len(trainloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        scheduler.step(avg_epoch_loss)

    return running_loss / (len(trainloader.dataset) * epochs)



def test(net, testloader, device):
    net.to(device)
    # Changed to BCEWithLogitsLoss to match training
    criterion = nn.BCEWithLogitsLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item() * features.size(0)
            # Apply sigmoid here since we're using logits
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = loss / len(testloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def predict(net, features, device='cpu'):
    net.to(device)
    net.eval()
    
    # Convert features to correct format if needed
    if isinstance(features, pd.DataFrame):
        # Drop target column if it exists
        if 'target' in features.columns:
            features = features.drop('target', axis=1)
        features = features.values.astype(np.float32)
    
    # Normalize features using the same method as in HeartDataset
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    
    # Convert to tensor
    features_tensor = torch.tensor(features).to(device)
    
    with torch.no_grad():
        # Get logits
        logits = net(features_tensor)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy()
        # Get binary predictions
        binary_predictions = (probabilities >= 0.5).astype(int)
    
    return probabilities, binary_predictions


def load_and_predict(model_input_dim, features, weights=None, device='cpu'):
    # Initialize model
    net = Net(input_dim=model_input_dim)
    
    # Load weights if provided
    if weights is not None:
        set_weights(net, weights)
    
    return predict(net, features, device)
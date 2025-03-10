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
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.activation(self.bn1(self.fc1(x))))
        x = self.dropout2(self.activation(self.bn2(self.fc2(x))))
        x = self.dropout3(self.activation(self.bn3(self.fc3(x))))
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
    """Train the model on the training set using BCELoss and Adam with learning rate scheduling."""
    net.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, labels in trainloader:
            features = torch.tensor(features).to(device)
            labels = torch.tensor(labels).to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)
        
        avg_epoch_loss = epoch_loss / len(trainloader.dataset)
        scheduler.step(avg_epoch_loss)
        running_loss += epoch_loss
    
    avg_trainloss = running_loss / (len(trainloader.dataset) * epochs)
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

# -------------------------
# 7. Prediction Function
# -------------------------
def predict(net, features, device='cpu'):
    """
    Make predictions using the trained model.
    
    Args:
        net: Trained PyTorch model
        features: numpy array or pandas DataFrame of features
        device: 'cpu' or 'cuda' device to use
        
    Returns:
        predictions: numpy array of predicted probabilities
        binary_predictions: numpy array of binary predictions (0 or 1)
    """
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
        # Get predictions
        outputs = net(features_tensor)
        predictions = outputs.cpu().numpy()
        binary_predictions = (predictions >= 0.5).astype(int)
    
    return predictions, binary_predictions

def load_and_predict(model_input_dim, features, weights=None, device='cpu'):
    """
    Load a model with weights and make predictions.
    
    Args:
        model_input_dim: Input dimension for the model
        features: Features to predict on
        weights: Optional model weights to load
        device: Device to use for computation
        
    Returns:
        predictions: Predicted probabilities and binary predictions
    """
    # Initialize model
    net = Net(input_dim=model_input_dim)
    
    # Load weights if provided
    if weights is not None:
        set_weights(net, weights)
    
    # Make predictions
    return predict(net, features, device)

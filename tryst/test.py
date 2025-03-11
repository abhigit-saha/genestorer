import torch
import numpy as np
from tryst.task import Net  # Import your model

# Load the trained model
model = Net(input_dim=14)  # Ensure this matches your FastAPI model
model.load_state_dict(torch.load("tryst/tryst/model_round_10.pth"))  # Use the latest checkpoint
model.eval()

# Test input (convert to tensor)
features = np.array([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 4], dtype=np.float32)
features = (features - features.mean()) / features.std()  # Ensure correct normalization
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

# Get prediction
with torch.no_grad():
    logits = model(features_tensor)
    probability = torch.sigmoid(logits).item()
    prediction = 1 if probability > 0.5 else 0

print(f"Logits: {logits.item()}, Probability: {probability}, Prediction: {prediction}")

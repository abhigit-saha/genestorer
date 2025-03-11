from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from tryst.task import Net
import os
from fastapi.responses import FileResponse
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load model
model = Net(input_dim=14)
model.load_state_dict(torch.load("tryst/model_round_10.pth"))
model.eval()

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    print(f"Received input: {data.features}")  # Debug input

    # Convert input to numpy array
    features = np.array(data.features, dtype=np.float32)
    
    print(f"Before normalization: {features}")  # Check values before normalization

    # Ensure correct normalization
    if features.std() != 0:  # Avoid division by zero
        features = (features - features.mean()) / features.std()

    print(f"After normalization: {features}")  # Check values after normalization

    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        logits = model(features_tensor)
        probability = torch.sigmoid(logits).item()
        prediction = 1 if probability > 0.5 else 0

    print(f"Logits: {logits.item()}, Probability: {probability}, Prediction: {prediction}")  # Debug outputs

    return {"prediction": prediction, "probability": probability}

@app.get("/regional-insights")
def get_regional_insights():
    image_path = "tryst/probability_vs_region.png"
    print("sent")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    return {"error": "Graph not found, train the model first."}


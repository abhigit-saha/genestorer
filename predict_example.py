import pandas as pd
import torch
from tryst.task import Net, predict, load_and_predict

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data = pd.read_csv("tryst/heart.csv")
    features = data.drop('target', axis=1)
    true_labels = data['target']

    # Initialize model with the correct input dimension
    input_dim = features.shape[1]
    print(f"Input dimension: {input_dim}")

    # Try to load saved model weights
    model_path = "tryst/model_weights.pth"
    net = Net(input_dim=input_dim).to(device)
    
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
        print(f"\nLoaded trained model weights from {model_path}")
    except FileNotFoundError:
        print("\nNo saved weights found. Using an untrained model.")

    # Predict using the loaded model
    predictions, binary_preds = predict(net, features, device)
    print_predictions(predictions, binary_preds, true_labels)

    # Predict for a single sample
    print("\nPredicting for a single sample:")
    single_sample = features.iloc[0:1]  # Get first sample
    pred_prob, pred_binary = predict(net, single_sample, device)
    print(f"Sample features:\n{single_sample}")
    print(f"Predicted probability: {pred_prob[0][0]:.4f}")
    print(f"Binary prediction: {pred_binary[0][0]}")

def print_predictions(predictions, binary_preds, true_labels=None):
    """Helper function to print prediction results."""
    print(f"\nFirst 5 predictions:")
    for i in range(min(5, len(predictions))):
        print(f"Sample {i+1}:")
        print(f"  Probability: {predictions[i][0]:.4f}")
        print(f"  Binary prediction: {binary_preds[i][0]}")
        if true_labels is not None:
            print(f"  True label: {true_labels.iloc[i]}")
    
    if true_labels is not None:
        accuracy = (binary_preds.flatten() == true_labels.values).mean()
        print(f"\nOverall accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()

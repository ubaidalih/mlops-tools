import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

# --- 1. Model Definition (LeNet-5) ---
class LeNet5(nn.Module):
    """A classic LeNet-5 architecture for MNIST."""
    def __init__(self):
        super(LeNet5, self).__init__()
        # Input is 1x28x28, we need to pad it to 1x32x32 for this architecture
        self.pad = nn.ConstantPad2d(2, 0) 
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5) # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. Helper Functions ---
def get_data_loaders(batch_size):
    """Loads MNIST and provides PyTorch DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device):
    """Evaluates the model on the test set."""
    model.eval() # Set model to evaluation mode
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def main():
    # --- Load Parameters ---
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- MLflow Tracking Setup ---
    mlflow.set_experiment("PyTorch LeNet5 MNIST")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log parameters
        mlflow.log_params(params['train'])
        mlflow.log_param("device", device)

        # --- TensorBoard Setup ---
        writer = SummaryWriter(log_dir=f"reports/logs/{run_id}")
        
        # --- Data, Model, Optimizer, and Loss ---
        train_loader, test_loader = get_data_loaders(params['train']['batch_size'])
        model = LeNet5().to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['train']['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # --- Training Loop ---
        print("Starting training...")
        for epoch in range(params['train']['epochs']):
            model.train() # Set model to training mode
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # --- Log metrics for this epoch to TensorBoard ---
            train_loss, train_acc = evaluate_model(model, train_loader, device)
            test_loss, test_acc = evaluate_model(model, test_loader, device)
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)
            
            print(f"Epoch {epoch+1}/{params['train']['epochs']} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        writer.close()
        print("Training finished.")

        # --- Final Evaluation and Logging ---
        final_loss, final_accuracy = evaluate_model(model, test_loader, device)
        print(f"\nFinal Test Loss: {final_loss}, Final Test Accuracy: {final_accuracy}")

        # Log final metrics to MLflow
        mlflow.log_metric("final_test_loss", final_loss)
        mlflow.log_metric("final_test_accuracy", final_accuracy)

        # Log metrics for DVC
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump({"accuracy": final_accuracy, "loss": final_loss}, f)

        # --- Log Artifacts ---
        # Create model output directory
        os.makedirs("models", exist_ok=True)
        model_path = "models/lenet5.pth"
        torch.save(model.state_dict(), model_path)
        
        # Log the model using the PyTorch flavor
        mlflow.pytorch.log_model(model, artifact_path="model")
        
        # Log the TensorBoard logs directory
        mlflow.log_artifacts(f"reports/logs/{run_id}", artifact_path="tensorboard_logs")

        print("\nExperiment run logged successfully.")

if __name__ == "__main__":
    main()
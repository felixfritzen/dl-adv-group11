import torch
from tqdm import tqdm
from torchvision import models
import datasets.datasets
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Get dataloaders for Waterbirds dataset
dataloaders = datasets.datasets.get_dataloaders('waterbirds')
print(dataloaders.keys())
class ResNet50XDNN(nn.Module):
    def __init__(self, pretrained_path):
        super(ResNet50XDNN, self).__init__()
        self.resnet50 = models.resnet50()  # Initialize ResNet50
        checkpoint = torch.load(pretrained_path,map_location="cpu")  # Load weights
        # checkpoint = torch.load(pretrained_path)
        model_state_dict = checkpoint['state_dict']

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
            new_state_dict[name] = v
        self.resnet50.load_state_dict(new_state_dict,strict=True)
        
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        return self.resnet50(x)

# Load the pre-trained model
model = ResNet50XDNN('xdnn/xfixup_resnet50_model_best.pth.tar').to(device)

# Set model to evaluation mode (important for testing)
model.eval()

# Testing Loop
correct = 0
total = 0

with torch.no_grad():  # No need to calculate gradients during testing
    for inputs, labels in tqdm(dataloaders['val_ood'], desc="Testing Model on Test Data"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass through the model
        outputs = model(inputs)

        # Get predictions
        _, preds = torch.max(outputs, 1)

        # Update correct predictions
        correct += (preds == labels).sum().item()
        total += labels.size(0)

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

import torch
import torch.nn as nn
from tqdm import tqdm
from fixup_resnet import xfixup_resnet50  # Assuming xfixup_resnet50 is defined in fixup_resnet.py
import datasets.datasets

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load test dataloader
dataloaders = datasets.datasets.get_dataloaders("waterbirds")
test_loader = dataloaders['test_id']  # Testing on out-of-distribution test set

# Recreate the model with 2 output classes
model = xfixup_resnet50(num_classes=2)  # Ensure this matches the fine-tuned model structure
model.fc = nn.Linear(model.fc.in_features, 2)  # Ensure the final layer is binary classification
model = model.to(device)

# Load fine-tuned weights
fine_tuned_weights_path = "/home/shared_project/dl-adv-group11/models/weights/waterbirds/old/resnet50_finetuned_final_unfrozen_augmented_200epoker.pth"  # Path to the saved weights
model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=device))

print("Fine-tuned weights loaded successfully.")

# Evaluate the model
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No gradients needed during evaluation
    for inputs, labels, _ in tqdm(test_loader, desc="Testing Fine-Tuned Model"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Get predictions
        _, preds = torch.max(outputs, 1)

        # Update correct predictions count
        correct += (preds == labels).sum().item()
        total += labels.size(0)

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Test Accuracy (Fine-Tuned): {accuracy:.2f}%")

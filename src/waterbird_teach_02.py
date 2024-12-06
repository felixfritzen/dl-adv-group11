import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
import datasets.datasets
import utils
from fixup_resnet import xfixup_resnet50
# Import localization loss functions
from localization_losses import get_localization_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize dataset and dataloaders
dataloaders = datasets.datasets.get_dataloaders('waterbirds')

# Initialize model
model = xfixup_resnet50(num_classes=2)  # Final layer outputs 2 classes

# Load pre-trained weights
checkpoint = torch.load('/home/shared_project/dl-adv-group11/models/pretrained/xdnn/xfixup_resnet50_model_best.pth.tar', map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Clean up the state_dict by removing 'module.' prefix if present
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')  # Remove 'module.' prefix
    new_state_dict[new_key] = value

if 'fc.weight' in new_state_dict:
    del new_state_dict['fc.weight']
if 'fc.bias' in new_state_dict:
    del new_state_dict['fc.bias']

# Load the cleaned model weights
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust final layer for binary classification
model.load_state_dict(new_state_dict, strict=False)
model = model.to(device)

# Freeze all layers except the final one
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

for param in model.fc.parameters():
    param.requires_grad = True  # Unfreeze the final layer

# Choose localization loss function based on command line argument
localization_loss_fn = get_localization_loss("Energy")  # Adjust this depending on the command-line arg
def total_loss(classification_loss, localization_loss, localization_lambda):
    return classification_loss + localization_lambda * localization_loss
# Set optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Number of epochs
epochs = 50

# Training Loop with Localization Loss (No CrossEntropyLoss)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        #bb_coordinates = bb_coordinates.to(device)  # Assuming bb_coordinates are available in the data

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Localization loss (no classification loss)
        #loc_loss = localization_loss_fn(outputs, bb_coordinates)  # You might pass attributions or other intermediate outputs
        
        # Total loss (you can include other attribution losses here if needed)
        #total_loss = loc_loss
        total_loss = criterion(outputs, labels)
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        # If you have a way to measure accuracy through localization, you can do it here
        # correct += (preds == labels).sum().item()
        # total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloaders['train'])
    # If using a classification accuracy metric, you can add that here
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'resnet50_finetuned_waterbirds.pth')
print("Model saved successfully!")

import torch
import torch.nn as nn
from tqdm import tqdm
from fixup_resnet import xfixup_resnet50  # Ensure this is correctly defined or imported
import datasets.datasets  # Custom datasets module
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
from localization_losses import get_localization_loss
import os
# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:1") 
print(f"Using device: {device}")

#Save paths
MODEL_PATH = "/home/shared_project/dl-adv-group11/src/experiments/waterbirds/5/"
FIGURE_PATH = MODEL_PATH
os.makedirs(MODEL_PATH, exist_ok=False)

# Load dataloaders
dataloaders = datasets.datasets.get_dataloaders("waterbirds")
train_loader = dataloaders['train']
id_val_loader = dataloaders['val_id']
ood_val_loader = dataloaders['val_ood']

# Recreate the model with 2 output classes
model = xfixup_resnet50(num_classes=2)  # Assuming your custom Fixup ResNet50 supports num_classes
model.fc = nn.Linear(model.fc.in_features, 2)  # Update final layer for binary classification
model = model.to(device)

# Load fine-tuned weights
fine_tuned_weights_path ="dl-adv-group11/models/pretrained/xdnn/xfixup_resnet50_model_best.pth.tar"  # Path to your pretrained model
model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=device))

# Freeze all layers except the final one
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True  # Only fine-tune the final fully connected layer

def total_loss(classification_loss, localization_loss, localization_lambda):
    return classification_loss #+ localization_lambda * localization_loss

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
localization_loss_fn = get_localization_loss("Energy")
optimizer = optim.Adam(model.parameters(), lr=1e-4)

lambda_loc =1e-3

# Training settings
epochs = 151
save_interval = 50
train_loss_list, train_acc_list = [], []
id_val_loss_list, id_val_acc_list = [], []
ood_val_loss_list, ood_val_acc_list = [], []

for epoch in range(1,epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels, bb_coordinates  in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        #bb_coordinates = bb_coordinates.to(device)
        #print(bb_coordinates)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss =total_loss(criterion(outputs,labels), localization_loss_fn(outputs,bb_coordinates), lambda_loc)


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    # In-Distribution Validation
    model.eval()
    id_val_loss = 0.0
    id_val_correct = 0
    id_val_total = 0

    with torch.no_grad():
        for inputs, labels, _ in tqdm(id_val_loader, desc=f"Epoch {epoch+1}/{epochs} - ID Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            id_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            id_val_correct += (preds == labels).sum().item()
            id_val_total += labels.size(0)

    id_val_loss /= len(id_val_loader)
    id_val_acc = id_val_correct / id_val_total
    id_val_loss_list.append(id_val_loss)
    id_val_acc_list.append(id_val_acc)

    # Out-of-Distribution Validation
    ood_val_loss = 0.0
    ood_val_correct = 0
    ood_val_total = 0

    with torch.no_grad():
        for inputs, labels, _ in tqdm(ood_val_loader, desc=f"Epoch {epoch+1}/{epochs} - OOD Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            ood_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            ood_val_correct += (preds == labels).sum().item()
            ood_val_total += labels.size(0)

    ood_val_loss /= len(ood_val_loader)
    ood_val_acc = ood_val_correct / ood_val_total
    ood_val_loss_list.append(ood_val_loss)
    ood_val_acc_list.append(ood_val_acc)

    # Log Epoch Stats
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"ID Val Loss: {id_val_loss:.4f}, ID Val Acc: {id_val_acc:.4f}")
    print(f"OOD Val Loss: {ood_val_loss:.4f}, OOD Val Acc: {ood_val_acc:.4f}")

    # Save model every `save_interval` epochs
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), f"{MODEL_PATH}resnet50_finetuned_augdata_fc50+{epoch+1}.pth")

# Save the final model
torch.save(model.state_dict(), MODEL_PATH +'resnet50_augmented_fc_200epokerPP.pth')
print("Final model saved successfully!")

# Plotting Training and Validation Metrics
epochs_range = range(1, epochs + 1)

# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_acc_list, label='Train Accuracy')
plt.plot(epochs_range, id_val_acc_list, label='ID Val Accuracy')
plt.plot(epochs_range, ood_val_acc_list, label='OOD Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(FIGURE_PATH +"new_accuracy_plot_augdata_nolocloss.png", dpi=200, bbox_inches='tight')  # Save accuracy plot

# Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_loss_list, label='Train Loss')
plt.plot(epochs_range, id_val_loss_list, label='ID Val Loss')
plt.plot(epochs_range, ood_val_loss_list, label='OOD Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(FIGURE_PATH +"new_loss_plot_augdata_nolocloss.png", dpi=200, bbox_inches='tight')  # Save loss plot

print("Training complete and plots saved!")

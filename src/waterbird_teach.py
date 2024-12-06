import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
import datasets.datasets
import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


dataloaders = datasets.datasets.get_dataloaders('waterbirds')

class ResNet50XDNN(nn.Module):
    def __init__(self, pretrained_path):
        super(ResNet50XDNN, self).__init__()
        self.resnet50 = models.resnet50(weights=None)  # Initialize ResNet50
        checkpoint = torch.load(pretrained_path,weights_only=True)  # Load weights
        self.resnet50.load_state_dict(checkpoint, strict=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        return self.resnet50(x)

model = ResNet50XDNN('/home/shared_project/dl-adv-group11/models/pretrained/xdnn/xfixup_resnet50_model_best.pth.tar').to(device)

for param in model.resnet50.parameters():
    param.requires_grad = True  # Freeze all layers

for param in model.resnet50.fc.parameters():
    param.requires_grad = True  # Unfreeze the final layer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 200

#Training Loop for fine tuning 
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, batch in enumerate(tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}")):
        inputs, labels =batch[0], batch[1]
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if i%10==0:
            torch.save(model.state_dict(), 'new_resnet50_finetuned_waterbirds.pth')
    
    epoch_loss = running_loss / len(dataloaders['train'])
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

#Save the Fine-tuned Model
torch.save(model.state_dict(), 'new_resnet50_finetuned_waterbirds.pth')
print("Model saved successfully!")

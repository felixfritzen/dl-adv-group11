import torch

def inspect_weights(weight_path):
    # Load weights
    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # Check if it's a full model or just state_dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        print("Weights contain a `state_dict`. Extracting it...")
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict):
        print("Weights loaded as a `state_dict`.")

    print("\nKeys in the state_dict:")
    for key in state_dict.keys():
        print(f"- {key}")

    # Check if the final layer corresponds to Waterbirds (2 output classes)
    if "fc.weight" in state_dict:
        print("\nFinal layer (fc.weight):")
        print(state_dict["fc.weight"])

        # Check if the final layer has 2 output units (binary classification for Waterbirds)
        if state_dict["fc.weight"].shape[0] == 2:  # fc.weight has shape [out_features, in_features]
            print("\nThe model appears to be fine-tuned for a binary classification task (likely Waterbirds).")
        else:
            print("\nThe final layer has more than 2 output features. It may not be fine-tuned for Waterbirds.")
    else:
        print("\nNo final layer explicitly found. This may not be a classification model.")

def main():
    # Path to weights
    weight_paths = [
        "xdnn/xfixup_resnet50_model_best.pth.tar",  # Adjust if needed
        "bcos/resnet_50-f46c1a4159.pth"             # Adjust if needed
    ]

    for path in weight_paths:
        print(f"\nInspecting weights at: {path}")
        inspect_weights(path)

if __name__ == "__main__":
    main()

from wilds import get_dataset
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
#from src.datasets import datasets
# Load the Waterbirds dataset
# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the image to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

# Load the Waterbirds dataset
dataset = get_dataset(dataset="waterbirds", download=True)

# Create DataLoader for the test subset (apply the transform)
test_dataset = dataset.get_subset("test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#dataset = get_dataset(dataset="waterbirds", download=True)
#dataset = datasets.datasets_gte.WaterBirdsDataset(root= './data',image_set=att, 
#                           transform=transform, target_transform=None,
#                           target_name='waterbird_complete95',
#                           confounder_names=None,
#                           reverse_problem=False)
# You can access the dataset directly
#test_dataset = dataset.get_subset("test")  # 'test' is the name of the split

# Create a DataLoader for the test dataset
#test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the model loading function
#def load_model_with_weights(weight_path):
#    model = models.resnet50(weights=None)  # Use ResNet50 without default pretrained weights
#    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Modify final layer for 2 classes (Waterbirds)

    # Load the checkpoint (weights, optimizer, etc.)
#    checkpoint = torch.load(weight_path, map_location="cpu")
    # Extract the state_dict from the checkpoint
#    state_dict = checkpoint['state_dict']  # Extract model weights
     # Load the state_dict into the model, allowing for missing keys
#    model.load_state_dict(state_dict, strict=False)
   #    model.eval()  # Set the model to evaluation mode
#    return model
   # model = models.resnet50(pretrained=False)  # Do not load the default pretrained weights
   # model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust output layer for 2 classes (Waterbirds)

    # Load custom weights
   # state_dict = torch.load(weight_path, map_location="cpu")
   # model.load_state_dict(state_dict)
   # model.load_state_dict(state_dict['state_dict'], strict=False)
   # model.eval()  # Set to evaluation mode
   # return model
def load_model_with_weights(weight_path):
    # Initialize the ResNet50 model
    model = models.resnet50(weights=None)  # Do not load the pretrained weights
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Modify final layer for 2 classes (Waterbirds)

    # Load the state_dict using safetensors (this is an alternative to torch.load)
    state_dict = torch.load(weight_path,map_location="cpu",weights_only=True)
#state_dict = load_file(weight_path)

    # Load the model weights
    model.load_state_dict(state_dict,strict=False) # to ignore missing keys

    model.eval()  # Set model to evaluation mode
    return model
# Load the model
model = load_model_with_weights("xdnn/xfixup_resnet50_model_best.pth.tar")

print(model)
state_dict = torch.load("xdnn/xfixup_resnet50_model_best.pth.tar", map_location="cpu",weights_only=True)
print(state_dict['state_dict'].keys())
#print(state_dict.keys())  # Look for the architecture
#print(state_dict['fc.weight'].shape)  # Check the final layer shape
print(state_dict['state_dict']['model.fc.weight'].shape)
"""
# Initialize variables to calculate accuracy
correct = 0
total = 0

# Evaluate on the Waterbirds test dataset
with torch.no_grad():
    for images, labels,_ in test_loader:
        outputs = model(images)  # Get model outputs
        predictions = outputs.argmax(dim=1)  # Get the class with the highest score
        correct += (predictions == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Count total samples

# Print the accuracy of the model
print(f"Accuracy on the Waterbirds test set: {100 * correct / total:.2f}%")
"""

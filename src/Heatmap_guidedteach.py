import torch
import torch.nn as nn
from fixup_resnet import xfixup_resnet50  # Assuming xfixup_resnet50 is defined in fixup_resnet.py
import datasets.datasets
from model_functions import GradCAM
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
import utils
from utils import show_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('Agg')
# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


#Save paths
MODEL_PATH = "/home/shared_project/dl-adv-group11/models/weights/"
FIGURE_PATH = "/home/shared_project/dl-adv-group11/figs/"

MODEL_PATH +="waterbirds/"
FIGURE_PATH +="waterbirds/"

# Load test dataloader
do_aug =True
dataloaders = datasets.datasets.get_dataloaders("waterbirds", do_aug)
test_loader = torch.utils.data.DataLoader(
    dataloaders['test_ood'].dataset,  # Use the same dataset
    batch_size=1,  # Set the batch size to 1
    shuffle=False,  # Typically, test sets are not shuffled
    num_workers=dataloaders['test_ood'].num_workers  # Keep the same number of workers
)

# Recreate the model with 2 output classes
model = xfixup_resnet50(num_classes=2)  # Ensure this matches the fine-tuned model structure
model.fc = nn.Linear(model.fc.in_features, 2)  # Ensure the final layer is binary classification
model = model.to(device)

# Load fine-tuned weights
fine_tuned_weights_path =MODEL_PATH + "old/resnet50_finetuned_final_unfrozen_augmented_200epoker.pth"  # Path to the saved weights
fine_tuned_weights_path =MODEL_PATH + "resnet50_finetuned_augdata_fc50+50.pth"  # Path to the saved weights
fine_tuned_weights_path =MODEL_PATH +"new_resnet50_finetuned_epoch_augdata_unfrozen150+25.pth"
fine_tuned_weights_path =MODEL_PATH +"new_resnet50_finetuned_final_unfrozen_augmented_200epoker.pth" #best
fine_tuned_weights_path =MODEL_PATH +"resnet50_augmented_fc_200epoker.pth"
#fine_tuned_weights_path =MODEL_PATH +"new_resnet50_finetuned_final_unfrozen_augmented_200epoker.pth"
model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=device))

print("Fine-tuned weights loaded successfully.")

# Set the model to evaluation mode
model.eval()  # Make sure we are in evaluation mode for inference

# Initialize GradCAM
gradcam = GradCAM(model=model, target_layer="layer4")

# Get a single batch of images and labels
itt = iter(test_loader)
input_image, label, bb_coordinate = next(itt)
input_image, label, bb_coordinate = next(itt)
input_image, label, bb_coordinate = next(itt)
input_image, label, bb_coordinate = next(itt) # good 
input_image, label, bb_coordinate = next(itt)
input_image, label, bb_coordinate = next(itt)
#input_image, label, bb_coordinate = next(itt)
#input_image, label = input_image.to(device), label.to(device)
#print(bb_coordintate)
"""
x_min, y_min, x_max, y_max = bb_coordinate[0].tolist()

# Convert to (x_min, y_min, width, height)
width = x_max - x_min
height = y_max - y_min
rect = patches.Rectangle(
    (x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none'
)
"""
# Generate heatmap
heatmap = gradcam.generate_heatmap(input_image, class_idx=label[0].item(), device=device)

# Detach the heatmap, move it to CPU and convert to numpy
heatmap = heatmap.detach().cpu().numpy()
heatmap = np.uint8(255 * heatmap.squeeze())

# Resize the heatmap to match the original image size
heatmap_resized = Image.fromarray(heatmap).resize((input_image.shape[2], input_image.shape[3]), Image.LANCZOS)

# Extract the original image for overlay (permute to HWC format)
#original_image = input_image[0].permute(1, 2, 0).cpu().numpy()
#print(original_image.shape)
#print(x_min, y_min, x_max, y_max)
#original_image = (original_image * 255).astype(np.uint8)  # Denormalize if needed
original_image = utils.show_image(input_image[0],FIGURE_PATH+'dataset_image.png',plot= False)
heatmap_resized = np.array(heatmap_resized)
# Apply a colormap to the heatmap and superimpose it on the original image
heatmap_color = plt.cm.jet(heatmap_resized / 255.0)[:, :, :3] * 255  # Apply 'jet' colormap
#superimposed_image = np.uint8(0.6 * heatmap_color.transpose(1, 0, 2)  + 0.4 * original_image)
heatmap_color = heatmap_color.astype(np.uint8)
cont = 0.7
if do_aug:
    superimposed_image = np.uint8(cont * heatmap_color+ (1-cont)* original_image)
else:
    superimposed_image = np.uint8(cont * heatmap_color.transpose([1,0,2]) + (1-cont)* original_image)
# Plot the results
plt.figure(figsize=(12, 12))

# Display the original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")
#ax1 = plt.gca()

#ax1.add_patch(rect)

# Display the superimposed image with GradCAM heatmap
plt.subplot(1, 2, 2)
plt.title("GradCAM Heatmap")
plt.imshow(superimposed_image)
plt.axis("off")
#ax2 = plt.gca()  # Get the current Axes instance for the second subplot
# Optionally, add the same bounding box to the GradCAM visualization
#ax2.add_patch(rect) 
plt.savefig(FIGURE_PATH+"heat_map_guidedteach_oodtest_alt1_fc_augdata_notranspose2.png")
# Show the plot
#plt.show()

"""
# Get a single batch from the test_loader
input_image, label = next(iter(test_loader))  # Get a single batch
print("Input Image Shape:", input_image.shape)  # Check the shape (Should be [1, 3, 224, 224])

# Remove batch dimension to get single image (C, H, W) -> (H, W, C)
input_image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
print("Image Shape after squeeze and permute:", input_image.shape)  # Check shape again

# If normalized (between 0 and 1), multiply by 255 to bring it to 0-255
# Check the min and max values before scaling
print("Min and Max of Image Tensor before scaling:", input_image.min(), input_image.max())

# Assuming the image was normalized, multiply by 255 and cast to uint8 for proper visualization
input_image = (input_image * 255).astype(np.uint8)  # Ensure it's in uint8 format
print("Min and Max of Image after scaling:", input_image.min(), input_image.max())  # Check after scaling

# Plot the image
plt.figure(figsize=(8, 8))
plt.imshow(input_image)
plt.axis("off")
plt.title(f"Test Image (Label: {label.item()})")
plt.show()
"""

"""
# Get one batch of data from the test_loader
input_image, label = next(iter(test_loader))  # Get a single batch of data
input_image = input_image.squeeze(0)  # Remove the batch dimension (to get a single image)

# Convert the image from Tensor (C, H, W) to NumPy array (H, W, C)
original_image = input_image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
original_image = (original_image * 255).astype(np.uint8)  # De-normalize the image if necessary

print("Input Image Shape:", input_image.shape)  # Should be [1, C, H, W] for batch_size=1
print("Label:", label)  # Check if the label is correct

# Plot the image
plt.figure(figsize=(8, 8))
plt.imshow(original_image)
plt.axis("off")
plt.title("Test Image")
plt.show()
"""

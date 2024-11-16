import matplotlib.pyplot as plt
import requests
import torch
import torch.nn.functional as F

def key2key(dict1, dict2):
    """If targets are the same, maps key to key"""
    feature_to_key_dict2 = {}
    for key, value in dict2.items():
        feature_to_key_dict2[tuple(value)] = key
    mapping = {}
    for key1, value1 in dict1.items():
        feature_tuple = tuple(value1)
        if feature_tuple in feature_to_key_dict2:
            mapping[key1] = feature_to_key_dict2[feature_tuple]
    mapping = {str(key): int(value) for key, value in mapping.items()}
    return mapping


def show_image(image):
    #plt.imshow(imagenet.idx2image(1))
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("figs/plot.png") 

def accuracy(predictions, ground_truth):
    correct = (predictions == ground_truth).sum().item()
    total = len(predictions)
    accuracy = correct / total * 100
    return accuracy


def downloadVOC():
    # Login URL and credentials
    login_url = "http://host.robots.ox.ac.uk/accounts/login/"
    download_url = "http://host.robots.ox.ac.uk/eval/downloads/VOC2012test.tar"
    session = requests.Session()

    # Login payload
    payload = {
        "username": "felixfritzen",
        "password": "nozkot-jixbo3-patqyW"
    }

    # Post login data
    session.post(login_url, data=payload)

    # Download file
    with session.get(download_url, stream=True) as response:
        with open("VOC2012test.tar", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def get_info(dataloaders):
    for key in dataloaders.keys():
        print(key, ' Loader ',len(dataloaders[key]),' Set ', len(dataloaders[key].dataset))


def kd_loss(student_logits, teacher_logits, temperature):
    """Compute the knowledge distillation (KD) loss using KL divergence."""
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss

def explanation_loss(student_explanation, teacher_explanation, similarity_metric='cosine'):
    """Compute explanation loss to match student and teacher explanations."""
    if similarity_metric == 'cosine':
        cos_sim = F.cosine_similarity(student_explanation, teacher_explanation, dim=1)
        loss = 1 - cos_sim.mean()
    else:
        raise ValueError("Only cosine similarity is currently supported.")
    return loss

def e2KD_loss(student_logits, teacher_logits, student_explanation, teacher_explanation, 
              temperature, lambda_weight):
    """Calculate the total e2KD loss combining KD loss and explanation loss."""
    kd = kd_loss(student_logits, teacher_logits, temperature)
    exp_loss = explanation_loss(student_explanation, teacher_explanation)
    total_loss = kd + lambda_weight * exp_loss
    return total_loss

#GradCAM Class
----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.
        Args:
            model (torch.nn.Module): The model to explain.
            target_layer (str): The name of the target layer for GradCAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Register hooks to save gradients and activations.
        """
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate a GradCAM heatmap.
        Args:
            input_tensor (torch.Tensor): Input image tensor (1, C, H, W).
            class_idx (int, optional): Target class index. Defaults to the predicted class.
        Returns:
            torch.Tensor: Resized GradCAM heatmap.
        """
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(input_tensor)  # (1, num_classes)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()

        # Compute gradients for the target class
        target_score = outputs[:, class_idx]
        target_score.backward()

        # Compute weights and GradCAM heatmap
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (batch, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU to keep positive values only

        # Normalize and resize the heatmap
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze(0).squeeze(0)  # Return single-channel heatmap
----------------------------------------------------------------------------------

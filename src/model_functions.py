import torch
import torch.nn.functional as F

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
#----------------------------------------------------------------------------
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
                module.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, device, class_idx=None, retain=False):
        """
        Generate a GradCAM heatmap.
        Args:
            input_tensor (torch.Tensor): Input image tensor (1, C, H, W).
            class_idx (int, optional): Target class index. Defaults to the predicted class.
        Returns:
            torch.Tensor: Resized GradCAM heatmap.
        """
        input_tensor = input_tensor.to(device)
        self.model.zero_grad()
        
        input_tensor.requires_grad = True

        # Forward pass
        outputs = self.model(input_tensor)  # (1, num_classes)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1)

        # Compute gradients for the target class
        batch_size = input_tensor.size(0)
        target_scores = outputs[torch.arange(batch_size), class_idx]  # (batch_size,)
        
        target_scores.backward(gradient=torch.ones_like(target_scores))# TODO retain_graph=retain, men ger bug 

        # Compute weights and GradCAM heatmap
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (batch, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU to keep positive values only

        # Normalize and resize the heatmap
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)  # (batch, 1, H, W)
        cam = cam - cam.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1, 1)
        cam = cam / cam.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1, 1)

        return cam.squeeze(1), outputs  # Return (batch_size, H, W)

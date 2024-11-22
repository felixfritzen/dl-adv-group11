from safetensors.torch import load_file
import torch
import torchvision.models as models


def pcam_teacher():
    checkpoint_path = "models/teachers/pcam/model.safetensors"
    state_dict = load_file(checkpoint_path)

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2) # edit output

    model.load_state_dict(state_dict)
    model.eval()
    return model

def main(): 
    model = pcam_teacher()
    print(model)

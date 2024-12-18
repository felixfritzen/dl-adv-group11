from safetensors.torch import load_file
import torch
import torchvision.models as models
from fixup_resnet import xfixup_resnet50 

def pcam_teacher():
    checkpoint_path = "/home/shared_project/dl-adv-group11/models/teachers/camelyon/model.safetensors"
    state_dict = load_file(checkpoint_path)

    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2, bias=True) # edit output

    model.load_state_dict(state_dict)
    model.eval()
    return model

def pcam_student():
    model = models.resnet18(pretrained=False)

    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features, 
        out_features=2,  # Binary
        bias=True
    )
    return model

def waterbirds_teacher():
    model = xfixup_resnet50(num_classes=2).to(torch.device('cuda'))  # Ensure this matches the fine-tuned model structure
    model.fc = torch.nn.Linear(model.fc.in_features, 2).to(torch.device('cuda'))   # Ensure the final layer is binary classification
    fine_tuned_weights_path = "/home/shared_project/dl-adv-group11/models/teachers/waterbirds/resnet50_augmented_fc_200epoker.pth"
    model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=torch.device('cuda')))
    return model

def waterbirds_student():
    return pcam_student() # also resnet 18, binary

def main(): 
    model = pcam_teacher()
    print(model)
    model = pcam_student()
    print(model)
if __name__ == "__main__":
    main()

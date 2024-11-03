
from torchvision import models, transforms
from tqdm import tqdm
import torch
import utils


def eval(device, dataloader, plot=True):
    #model.load_state_dict(torch.load('./models/teachers/resnet_34-a63425a03e.pth', weights_only=True))
    #NOT SAME AS STANDARD
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    model.eval()  # Set the model to evaluation mode
    #print(dir(model))
    all_predictions, all_labels = [], []
    for i, (images, labels, _) in enumerate(tqdm(dataloader)):
        #print(images.shape, labels.shape)
        #utils.show_image(images[0])
        with torch.no_grad():
            output = model(images.to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            best_class_indices = torch.argmax(probabilities, dim=1)
            if plot and i%10 ==0:
                print(best_class_indices)
                print(labels)
                print(utils.accuracy(best_class_indices, labels))
                print(labels[0], dataloader.dataset.id2class[int(labels[0])])
            all_predictions.append(best_class_indices)
            all_labels.append(labels)
        if i>1000:# if you want to test only a few
            break
    all_predictions, all_labels = torch.cat(all_predictions).cpu(), torch.cat(all_labels).cpu()
    acc = utils.accuracy(all_predictions, all_labels)
    print('Accuracy', acc) # 54.947

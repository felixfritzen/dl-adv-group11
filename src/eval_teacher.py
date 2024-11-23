import torch
from torchvision import models
from tqdm import tqdm
import datasets.datasets as datasets
import utils
# Device setup
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

def evaluate_teacher(model, dataloader):
    """
    Evaluate the teacher model on validation/test data.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating Teacher"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    print(correct, total)

    accuracy = correct / total
    print(f"Teacher Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def eval_imagenet(device, model, dataloader, plot=True):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    model.eval()  # Set the model to evaluation mode
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
        if i>10:# if you want to test only a few
            break
    all_predictions, all_labels = torch.cat(all_predictions).cpu(), torch.cat(all_labels).cpu()
    acc = utils.accuracy(all_predictions, all_labels)
    print('Accuracy', acc) # 54.947

def eval_cam(model):
    #test_acc_loaders()
    model.eval()
    all_predictions, all_labels = [], []

    dataloaders =datasets.get_dataloaders('camelyon')
    for i, (images, labels) in enumerate(tqdm(dataloaders['val'])):
        #print(images.shape, labels.shape)
        #utils.show_image(images[0])
        with torch.no_grad():
            output = model(images.to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            best_class_indices = torch.argmax(probabilities, dim=1)

            all_predictions.append(best_class_indices)
            all_labels.append(labels)
        if i>10:# if you want to test only a few
            break
    all_predictions, all_labels = torch.cat(all_predictions).cpu(), torch.cat(all_labels).cpu()
    acc = utils.accuracy(all_predictions, all_labels)
    print('Accuracy', acc)
    print(all_predictions)
    print(all_labels.sum(), all_labels.shape)

def test_waterbirds():
    dataloaders = datasets.get_dataloaders("waterbirds")  

    teacher = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    print(dataloaders)
    evaluate_teacher(teacher, dataloaders['val_id'])  #TODO broken id ood


def test_acc_loaders():
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    dataloaders =datasets.get_dataloaders('imagenet')
    for att in ['train', 'test', 'val']:
        eval_imagenet(device, model, dataloaders[att], plot=False)

def main():
    #test_waterbirds()
    test_acc_loaders()


if __name__ == "__main__":
    main()

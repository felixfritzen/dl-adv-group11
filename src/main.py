import torch 
import requests
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import pandas as pd
import scipy
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


def datasets():
    import os

    print("Absolute path:", os.path.abspath("./data/voc/test/VOCdevkit/VOC2012"))
    current_directory_files = os.listdir(".")
    print(os.path.exists('./data/voc/test/VOCdevkit/VOC2012'))
    print(current_directory_files)
    from torchvision.datasets import VOCDetection
    from torchvision import transforms

    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialize the dataset
    voc_train_val = VOCDetection(
        root="./data/voc/train",
        year="2012",
        image_set="trainval",
        download=False,
        transform=transform
    )

    image, target = voc_train_val[0]
    print("Image shape:", image.shape)
    print("Target:", target)

    voc_test = VOCDetection(
    root="./data/voc/test",
    download=False,
    image_set="trainval",
    transform=transform
    )
    image, target = voc_test[0]
    print("Image shape:", image.shape)
    print("Target:", target)

def waterbirds():
    #
    pass

def imagenet():
    labels_path = './data/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    images_path = './data/imagenet/images/' #ILSVRC2012_val_00000001.JPEG
    pass


class ImagenetDataset(Dataset):
    def __init__(self, labels_path, images_path):
        self.images_path = images_path
        self.transform =  transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        with open(labels_path, 'r') as f:
            labels = torch.tensor([int(line.strip()) for line in f], dtype=torch.long)
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = self.idx2image(idx)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx

    def idx2image(self, idx):
        image = Image.open(self.images_path+f'ILSVRC2012_val_{idx+1:08d}.JPEG').convert("RGB")
        return image
    
def show_image(image):
    #plt.imshow(imagenet.idx2image(1))
    plt.imshow(image.permute(1, 2, 0))
    plt.savefig("figs/plot.png") 
    
from torchvision import models, transforms

def imagenet_dataloader():
    labels_path = './data/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    images_path = './data/imagenet/images/'
    imagenet_dataset = ImagenetDataset(labels_path, images_path)
    dataloader = DataLoader(imagenet_dataset, batch_size=32, shuffle=False)
    return dataloader


def get_id2name2012(path):
    """Note that id 1 2012 does is not the same in 2017"""
    mat_data = scipy.io.loadmat(path)
    synsets = mat_data['synsets']
    ids = [entry[0][0][0][0] for entry in synsets[:1000]]
    names = [entry[0][2][0] for entry in synsets[:1000]]
    #df = pd.DataFrame({'ids':ids, 'names':names})
    id2name = dict(zip(ids, names))

    print(id2name[410])
    id2name[410] = 'cockatoo, Kakatoe galerita, Cacatua galerita'# only missmatch was sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
    return id2name

def getkey2key():
    sys.path.append('/home/shared_project/dl-adv-group11/external_libs/GoodTeachersExplain/bcos/data')
    from imagenet_classnames import name_map as id2name2017

    id2name2012 = get_id2name2012('/home/shared_project/dl-adv-group11/data/imagenet/ILSVRC2012_devkit_t12/data/meta.mat')
    print(id2name2012[490])
    print('diff ',set(id2name2017.values()) - set(id2name2012.values()), 'diff ')
    print(set(id2name2012.values())-set(id2name2017.values()))
    print(id2name2012)
    mapping = key2key(id2name2012, id2name2017)
    # Save the dictionary to a JSON file
    with open('mapping.json', 'w') as f:
        json.dump(mapping, f)
    return mapping


def key2key(dict1, dict2):
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


def forward():
    #model.load_state_dict(torch.load('./models/teachers/resnet_34-a63425a03e.pth', weights_only=True))
    #NOT SAME AS STANDARD
    model = models.resnet34(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    print(dir(model))

    for i, (images, labels, idx) in enumerate(imagenet_dataloader()):
        if i <1:
            print(idx, 'idx')
            print(images.shape, labels.shape)
            show_image(images[0])
            print(labels[0])

            with torch.no_grad():
                output = model(images)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                best_class_indices = torch.argmax(probabilities, dim=1)
                print(best_class_indices)
                preds = []
                for cl in best_class_indices:
                    preds.append(id2name2017[int(cl)])
                print(preds)
                print(labels)
                labs = []
                for cl in labels:
                    labs.append(id2name2012[int(cl)])
                print(labs)
        break

def main():
    #forward()
    getkey2key()

if __name__ == "__main__":
    main()

# show memory used: df -h, du -sh ./data
# untar: tar -xzvf ILSVRC2017_DET.tar.gz

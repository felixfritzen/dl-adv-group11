import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy
from torchvision import transforms

import utils, datasets #ours

from torch.utils.data import random_split
################## ImageNet #####################

class ImagenetDataset(Dataset):
    def __init__(self, transform, labels_path, images_path, mapping, id2name2017):
        self.images_path = images_path
        self.transform =  transform
        with open(labels_path, 'r') as f:
            self.labels = torch.tensor([int(line.strip()) for line in f], dtype=torch.long)
        for i, _ in enumerate(self.labels):
            self.labels[i]=mapping[str(int(self.labels[i]))]
        self.id2class = id2name2017
    
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

def get_id2name2012(path):
    """Id to class label, Note that id in 2012 dataset
      does is not the same as targets of resenet"""
    mat_data = scipy.io.loadmat(path)
    synsets = mat_data['synsets']
    ids = [entry[0][0][0][0] for entry in synsets[:1000]]
    names = [entry[0][2][0] for entry in synsets[:1000]]
    #df = pd.DataFrame({'ids':ids, 'names':names})
    id2name = dict(zip(ids, names))
    #print(id2name[410])
    id2name[410] = 'cockatoo, Kakatoe galerita, Cacatua galerita'
    # only missmatch was sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
    return id2name

def getkey2key():
    """Remap key in dataset to key in resnet with same target"""
    from datasets.imagenet_classnames import name_map as id2name2017
    id2name2012 = get_id2name2012('/home/shared_project/dl-adv-group11/data/imagenet/ILSVRC2012_devkit_t12/data/meta.mat')
    #print('diff ',set(id2name2017.values()) - set(id2name2012.values()), 'diff ')
    mapping = utils.key2key(id2name2012, id2name2017)
    # Save the dictionary to a JSON file
    with open('mapping.json', 'w') as f:
        json.dump(mapping, f)
    return mapping, id2name2017

def get_loaders_imagenet(transform, batch_size = 32):
    """Image and int target"""
    labels_path = './data/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    images_path = './data/imagenet/images/'
    mapping, id2name2017 = getkey2key()
    dataloaders = {}
    ds = {}
    imagenet_dataset = ImagenetDataset(transform, labels_path, images_path, mapping, id2name2017)
    
    train_size = int(0.8 * len(imagenet_dataset))
    val_size = int(0.1 * len(imagenet_dataset))
    test_size = len(imagenet_dataset) - train_size - val_size
    
    ds['train'], ds['val'], ds['test'] = random_split(imagenet_dataset, [train_size, val_size, test_size])
    for key in ds.keys():
        shuffle = True if key == 'train' else False
        dataloaders[key]=DataLoader(ds[key],batch_size=batch_size, shuffle=shuffle)
    return dataloaders


################## Waterbirds #####################


def get_loaders_waterbirds(transform, splits=['train', 'val', 'test'], batch_size=64):
    """Image and binary target"""
    dataloaders = {}
    for att in splits: 
        dataset = datasets.datasets_gte.WaterBirdsDataset(root= './data',image_set=att, 
                           transform=transform, target_transform=None,
                           target_name='waterbird_complete95',
                           confounder_names=None,
                           reverse_problem=False)
        
        shuffle = True if att == 'train' else False

        dataset.also_return_groups = True if  att == 'test' else False # for out/in dist

        dataloaders[att] =  DataLoader(dataset,batch_size=batch_size,
                                      shuffle=shuffle)
    return dataloaders



################## VOC #####################

def get_loaders_voc(transform, batch_size = 64):
    """Image and one/multi hot of 20 classes target"""
    ds = {}
    dataloaders = {}

    dataset = datasets.datasets_gte.VOCDataset(
        root="./data",
        year="2012",
        image_set="train",
        download=False,
        transform=transform)
    # now we split this into train/val 90/10"
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    ds['train'], ds['val'] = random_split(dataset, [train_size, val_size])

    ds['test']  = datasets.datasets_gte.VOCDataset(
        root="./data",
        year="2012",
        image_set="val", # uses the validation dataset
        download=False,
        transform=transform)
    
    for key in ds.keys():
        shuffle = True if key == 'train' else False
        dataloaders[key]=DataLoader(ds[key],batch_size=batch_size,
                                    shuffle=shuffle)
    return dataloaders





############### General #############


def get_dataloaders(ds):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)# from paper
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    if ds == 'imagenet':
        return get_loaders_imagenet(transform, batch_size = 64)
    elif ds == 'waterbirds':
        return get_loaders_waterbirds(transform, batch_size=64)
    elif ds == 'voc':
        return get_loaders_voc(transform, batch_size=64)
    else:
        print('Not valid dataset!')


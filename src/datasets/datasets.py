import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy
from torchvision import transforms

import utils, datasets #ours
from . import datasets_gte
import h5py
from torch.utils.data import random_split
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
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

def get_loaders_imagenet(transforms, batch_size=32):
    """
    ImageNet dataloaders with specified data augmentation.
    """
    labels_path = '/home/shared_project/dl-adv-group11/data/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    images_path = '/home/shared_project/dl-adv-group11/data/imagenet/images/'
    mapping, id2name2017 = getkey2key()
    dataloaders = {}
    ds = {}
    
    # Load the full dataset indices
    total_size = len(ImagenetDataset(None, labels_path, images_path, mapping, id2name2017))
    indices = list(range(total_size))
    np.random.shuffle(indices)  # Make sure to shuffle indices for randomness

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Split indices for reproducibility
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create separate datasets for each split with their own transforms
    train_dataset = ImagenetDataset(transforms['train'], labels_path, images_path, mapping, id2name2017)
    val_dataset = ImagenetDataset(transforms['eval'], labels_path, images_path, mapping, id2name2017)
    test_dataset = ImagenetDataset(transforms['eval'], labels_path, images_path, mapping, id2name2017)

    # Create Subsets
    ds['train'] = torch.utils.data.Subset(train_dataset, train_indices)
    ds['val'] = torch.utils.data.Subset(val_dataset, val_indices)
    ds['test'] = torch.utils.data.Subset(test_dataset, test_indices)
    
    for key in ds.keys():
        shuffle = True if key == 'train' else False
        if key == 'val':
            bs = TEST_BATCH_SIZE
        if key == 'test':
            bs = TEST_BATCH_SIZE
        else: 
            bs = batch_size
        dataloaders[key] = DataLoader(ds[key], batch_size=bs, shuffle=shuffle, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    return dataloaders


################## Waterbirds #####################


def get_loaders_waterbirds(transforms, splits=['train', 'val', 'test'], batch_size=64):
    """
    Loaders for Waterbirds with support for ID and OOD splits.
    """
    dataloaders = {}
    for att in splits:
        if att == 'train':
            transform = transforms['train']
        else:
            transform = transforms['eval']

        dataset = datasets.datasets_gte.WaterBirdsDataset(
            root='/home/shared_project/dl-adv-group11/data',
            image_set=att,
            transform=transform,
            target_transform=None,
            target_name='waterbird_complete95',
            confounder_names=None,
            reverse_problem=False
        )
        dataloaders[att]= DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
        if att in ['val', 'test']:
            # Filter directly on self.indices to ensure valid subset indices
            id_indices = np.where(
            (dataset.group_array[dataset.indices] == 3) | (dataset.group_array[dataset.indices] == 0)
            )[0] # Waterbird on water, landbird on land
            ood_indices = np.where(
            (dataset.group_array[dataset.indices] == 2) | (dataset.group_array[dataset.indices] == 1)
            )[0] # Waterbird on land,  landbird on water
            id_dataset = torch.utils.data.Subset(dataset, id_indices)
            ood_dataset = torch.utils.data.Subset(dataset, ood_indices)

            print(f"ID Dataset '{att}': {len(id_dataset)} samples")
            print(f"OOD Dataset '{att}': {len(ood_dataset)} samples")

            dataloaders[f'{att}_id'] = DataLoader(id_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
            dataloaders[f'{att}_ood'] = DataLoader(ood_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
        else:
            dataloaders[att] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)

    return dataloaders




################## VOC #####################

def get_loaders_voc(transforms, batch_size = 64):
    """Image and one/multi hot of 20 classes target"""
    ds = {}
    dataloaders = {}

    dataset = datasets.datasets_gte.VOCDataset(
        root="./data",
        year="2012",
        image_set="train",
        download=False,
        transform=transforms['train'])
    # now we split this into train/val 90/10"
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    ds['train'], ds['val'] = random_split(dataset, [train_size, val_size])

    ds['test']  = datasets.datasets_gte.VOCDataset(
        root="./data",
        year="2012",
        image_set="val", # uses the validation dataset
        download=False,
        transform=transforms['eval'])
    
    for key in ds.keys():
        
        shuffle = True if key == 'train' else False
        dataloaders[key]=DataLoader(ds[key],batch_size=batch_size,
                                    shuffle=shuffle)
    return dataloaders

############## CAMYLON ##############

class PatchCamelyon(Dataset):
    def __init__(self, root, split='train', transform=None, max_samples = 50000):
        self.root = root
        self.transform = transform
        self.split = split
        self.max_samples = max_samples
        with h5py.File(self.root + self.split2path('x'), "r") as h5:
            self.total_samples = h5["x"].shape[0]  # Total number of samples in the file
        # Limit the number of samples to max_samples or the total number available
        self.num_samples = min(self.total_samples, self.max_samples)
        
    def __len__(self):
        return self.num_samples
    
    def split2path(self, z):
        """choose z to x or y depending on image or label"""
        return f'camelyonpatch_level_2_split_{self.split}_{z}.h5'

    def __getitem__(self, idx):
        with h5py.File(self.root+self.split2path('x'), "r") as h5:
            image = h5["x"][idx] 
        with h5py.File(self.root+self.split2path('y'), "r") as h5:
            label = h5["y"][idx].squeeze() 

        if self.transform:
            image = self.transform(image)
        else:
             image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    


def get_loaders_patchcamelyon(transforms, splits=['train', 'valid', 'test'], batch_size=64, total_max_samples=50000):
    """Image and binary target"""
    dataloaders = {}
    split_ratios = {
        'train': 0.8,   
        'valid': 0.1,   
        'test': 0.1     
    }
    split_max_samples = {split: int(total_max_samples * ratio) for split, ratio in split_ratios.items()}

    for att in splits: 
        if att == 'train':
            transform = transforms['train']
        else:
            transform = transforms['eval']
            
        max_samples = split_max_samples.get(att, total_max_samples)
        
        dataset = PatchCamelyon('/home/shared_project/dl-adv-group11/data/pcamv1/', att, transform, max_samples=max_samples)
        
        shuffle = True if att == 'train' else False
        if att == 'valid':
            bs = TEST_BATCH_SIZE
            att ='val'
        else: 
            bs = batch_size
        dataloaders[att] =  DataLoader(dataset,batch_size=bs,
                                      shuffle=shuffle, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    return dataloaders



############### General #############
def get_transforms(ds):
    """Transformations used in dataloaders, no random in eval dataloaders"""
    transforms_dict={}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)# from paper
    if ds == 'imagenet':
        transforms_dict['train'] = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        transforms_dict['eval'] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif ds == 'waterbirds':
        bb= False
        if not bb:
            transforms_dict['train'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),#from paper
                transforms.RandomResizedCrop(224,scale=(0.7, 1.0),ratio=(0.75, 1.3333333333333333)),
                transforms.Normalize(mean=mean, std=std),
            ])
            transforms_dict['eval'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(224), #DEFAULT_CROP_SIZE = 224
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            transforms_dict['train'] = A.Compose([
            A.SmallestMaxSize(max_size=224),
            #A.RandomResizedCrop(height=224, width=224, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=mean),
            ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco")
            )
            transforms_dict['eval'] = A.Compose([
            #A.SmallestMaxSize(max_size=224),
            A.LongestMaxSize(max_size=256), 
            #A.CenterCrop(height=224, width=224),  
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco")
            )
    elif ds == 'voc':
        transforms_dict['train'] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
        transforms_dict['eval']=transforms_dict['train']  
    elif ds == 'camelyon':
        transforms_dict = { # incoming tensor
        'train': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(224), 
            transforms.RandomHorizontalFlip(),  
            transforms.RandomRotation(10), 
            transforms.ToTensor(),
            #transforms.Normalize(mean,std)
        ]),
        'eval': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(224), 
            transforms.ToTensor(),
            #transforms.Normalize(mean, std) #88.769% if no rezize nor normalize
        ])}

    else:
        print('Not valid dataset')
    return transforms_dict

def get_dataloaders(ds, do_aug=True):
    global NUM_WORKERS # parallell dataloading
    global PREFETCH_FACTOR # load in advance
    global TEST_BATCH_SIZE
    PREFETCH_FACTOR = 4 
    NUM_WORKERS = 8
    TEST_BATCH_SIZE = 256
    transforms_dict = get_transforms(ds)
    if not do_aug:
        transforms_dict = {}
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transforms_dict['train'] = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
        transforms_dict['eval'] = transforms_dict['train'] 

    if ds == 'imagenet':
        return get_loaders_imagenet(transforms_dict, batch_size=32)
    elif ds == 'waterbirds':
        return get_loaders_waterbirds(transforms_dict, batch_size=64)
    elif ds == 'voc':
        return get_loaders_voc(transforms_dict, batch_size=64)
    elif ds == 'camelyon':
        return get_loaders_patchcamelyon(transforms_dict, batch_size=256)
    else:
        print('Not valid dataset!')


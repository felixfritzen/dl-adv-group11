import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import pandas as pd
import scipy
from torchvision import models, transforms
from tqdm import tqdm
import importlib

import datasets.datasets
import datasets.datasets_gte
import model_functions, utils, training, datasets

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available(): # For apple silicon
    device = 'mps'
print("Using device:", device)


def demo():
    for ds in ['voc', 'waterbirds', 'camelyon', 'imagenet']:
        dataloaders = datasets.datasets.get_dataloaders(ds)
        print(ds)
        utils.get_info(dataloaders)

    training.eval(device, dataloaders['val'], False) # only imagenet atm

def inspect_dataloader(dataloaders):
    images, labels = next(iter(dataloaders['train']))
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print(labels)

def main(): 
    demo()
    #dataloaders = datasets.datasets.get_dataloaders('camelyon')
    #print(dataloaders)
    #inspect_dataloader(dataloaders)

if __name__ == "__main__":
    main()
# show memory used: df -h, du -sh ./data
# untar: tar -xzvf ILSVRC2017_DET.tar.gz

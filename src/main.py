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
import model_functions, utils, training, datasets, our_models

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available(): # For apple silicon
    device = 'mps'
print("Using device:", device)


def demo():
    for ds in ['waterbirds', 'camelyon', 'imagenet']:
        dataloaders = datasets.datasets.get_dataloaders(ds)
        print(ds)
        utils.get_info(dataloaders)

    #training.eval(device, dataloaders, False) # only imagenet atm

def inspect_dataloader(dataloaders):
    images, labels = next(iter(dataloaders['train']))
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print(labels)


def show_dataloader_demo():
    dataloaders = datasets.datasets.get_dataloaders('waterbirds')
    data_iter = iter(dataloaders['test_id'])
    images, labels = next(data_iter)
    image = utils.show_image(images[0], './figs/dataset_image.png')
    print(image)


def demo_waterbirds_acc():
    # values for ["700 epochs OOD", "700 epochs ID", "+5x training OOD", "+5x training ID"]
    accuracy_data = {
        "KD": [38.74, 92.72, 38.74, 92.72],
        "e$^2$KD": [37.3, 93.3, 37.3, 93.3]
    }
    agreement_data = {
        "KD": [59.96, 93.96, 59.96, 93.96],
        "e$^2$KD": [61.93, 95.13, 61.93, 95.13]
    }
    utils.plot_waterbirds_result(accuracy_data, agreement_data, '/home/shared_project/dl-adv-group11/src/experiments/waterbirds/final/wb_result.png')


def main(): 
    #model = models.pcam_teacher()
    #print(model)
    #demo()
    #dataloaders = datasets.datasets.get_dataloaders('camelyon')
    #print(dataloaders)
    #inspect_dataloader(dataloaders)
    #show_dataloader_demo()
    demo_waterbirds_acc()
    #demo()
    

if __name__ == "__main__":
    main()
# show memory used: df -h, du -sh ./data
# untar: tar -xzvf ILSVRC2017_DET.tar.gz

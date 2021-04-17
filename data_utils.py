
""" 
Data management and loaders
"""

import os
import numpy as np 
import torchvision.transforms as T 
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class DeepfakeDataset(Dataset):

    def __init__(self, paths, labels, transform=None):
        super().__init__()
        self.paths = paths 
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        img = self.transform(img)
        if self.labels is not None:
            trg = self.labels[index]
            return img, trg
        else:
            return img


def get_dataloaders(self, train_root, test_root, transforms, val_split, batch_size):
    """ 
    train_root will have two sub-directories: real and fake
    test_root will just have images
    """
    real_paths, fake_paths, test_paths = [], [], []

    for folder in os.listdir(train_root):
        for file in os.listdir(os.path.join(train_root, folder)):
            path = os.path.join(train_root, folder, file)
            if "real" in folder:
                real_paths.append(path)
            else:
                fake_paths.append(path)
    
    for file in os.listdir(test_root):
        path = os.path.join(test_root, file)
        test_paths.append(path)

    if isinstance(transforms, dict):
        train_transform, val_transform = transforms['train'], transforms['val']
    else:
        train_transform, val_transform = transforms, transforms

    val_real, val_fake = real_paths[:int(val_split*len(real_paths))], fake_paths[:int(val_split*len(fake_paths))]
    train_real, train_fake = real_paths[int(val_split*len(real_paths)):], fake_paths[int(val_split*len(fake_paths)):]
    
    train_dset = DeepfakeDataset(paths=train_real+train_fake, labels=[0]*len(train_real)+[1]*len(train_fake), transform=train_transform)
    val_dset = DeepfakeDataset(paths=val_real+val_fake, labels=[0]*len(val_real)+[1]*len(val_fake), transform=val_transform)
    test_dset = DeepfakeDataset(paths=test_paths, labels=None, transform=val_transform)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

""" 
Data management and loaders
"""

import os
import numpy as np 
import torchvision.transforms as T 
from PIL import Image
from ntpath import basename
from torch.utils.data import Dataset, DataLoader


TRANSFORM_HELPER = {
    'horizontal_flip': T.RandomHorizontalFlip,
    'vertical_flip': T.RandomVerticalFlip,
    'rotation': T.RandomRotation,
    'grayscale': T.RandomGrayscale,
    'tensor': T.ToTensor,
    'normalize': T.Normalize
}


def get_transform(transform_dict):
    transform_list = []
    for name, params in transform_dict.items():
        assert name in TRANSFORM_HELPER.keys(), f"Invalid transform {name}"
        if params is not None:
            transform_list.append(TRANSFORM_HELPER[name](**params))
        else:
            transform_list.append(TRANSFORM_HELPER[name]())
    return T.Compose(transform_list)


class DeepfakeDataset(Dataset):

    def __init__(self, paths, labels, transform=None):
        super().__init__()
        self.paths = paths 
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        img = self.transform(img)
        if self.labels is not None:
            trg = self.labels[index]
            return img, trg
        else:
            return img


def get_dataloaders(train_root, test_root, transforms, val_split, batch_size):
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
    test_ids = [int(basename(path).split(".")[0]) for path in test_paths]
    train_transform, val_transform = get_transform(transforms['train']), get_transform(transforms['val'])

    val_real, val_fake = real_paths[:int(val_split*len(real_paths))], fake_paths[:int(val_split*len(fake_paths))]
    train_real, train_fake = real_paths[int(val_split*len(real_paths)):], fake_paths[int(val_split*len(fake_paths)):]
    
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    for real, fake in zip(train_real, train_fake):
        train_paths.append(real), train_paths.append(fake)
        train_labels.append(0), train_labels.append(1)
    for real, fake in zip(val_real, val_fake):
        val_paths.append(real), val_paths.append(fake)
        val_labels.append(0), val_labels.append(1)
    
    train_dset = DeepfakeDataset(paths=train_paths, labels=train_labels, transform=train_transform)
    val_dset = DeepfakeDataset(paths=val_paths, labels=val_labels, transform=val_transform)
    test_dset = DeepfakeDataset(paths=test_paths, labels=test_ids, transform=val_transform)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader
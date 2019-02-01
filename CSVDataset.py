import pandas as pd
import numpy as np
from PIL import Image
import os
import torchvision.transforms as sttransforms
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_image(filepath, channels = 3):
    if channels == 1:
        image = Image.open(filepath).convert('YCbCr')
        image, _, _ = image.split()
    else:
        image = Image.open(filepath).convert('RGB')
    return image


class CSVDataset(Dataset):
    def __init__(self, dir, csv_path, channels, transforms):
        self.root_dir = dir
        self.channels = channels
        self.transforms = transforms
        self.data_info = pd.read_csv(csv_path, header=0)
        self.tags_list = self.data_info.columns.tolist()
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1:])
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        single_image_name = os.path.join(self.root_dir,self.image_arr[index])
        single_image_label = torch.from_numpy(self.label_arr[index]).float()
        img_as_img = load_image(single_image_name, self.channels)
        img_as_tensor = self.transforms(img_as_img)
        return img_as_tensor, single_image_label

    def __len__(self):
        return self.data_len

    def tags(self):
        return np.asarray(self.tags_list[1:])

    def files(self):
        return self.image_arr

def make_dataloaders (dataset, batch_size, splitratio = 0.2):
    print(' split ratio ', splitratio)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(splitratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # print(train_indices, val_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    print(train_sampler, valid_sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                                    sampler=valid_sampler)
    print(train_loader, validation_loader)
    dataloaders = {'train': train_loader, 'val': validation_loader}
    return dataloaders
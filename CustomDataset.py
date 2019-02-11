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

def CustomDataset(dir, csv_path, channels, transforms):
    return CSVDataset(dir, csv_path, channels, transforms)

def CustomDataset(dir, channels, transforms):
    return FolderDataset(dir, channels, transforms)


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


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images


class FolderDataset(Dataset):
    def __init__(self, root, channels = 1, transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx)
        self.channels = channels
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]

        sample = load_image(path, self.channels)
        if self.transform is not None:
            sample = self.transform(sample)
        target = torch.FloatTensor([target])
        #print(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


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
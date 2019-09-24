import pandas as pd
import numpy as np
from PIL import Image
import os
import torchvision
import torch
import random

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import ImageFilter, ImageEnhance, Image

from DeepImageRecognition import DIMENSION, CHANNELS, IMAGE_SIZE

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".bmp", ".jpg", ".jpeg"])

def load_image(filepath):
    if CHANNELS == 3:
        return Image.open(filepath).convert('RGB')
    else:
        return Image.open(filepath).convert('L')


def CustomDataset(image_dir, csv_path: str = "",  augmentation: bool = False):
    if csv_path is None:
        return FolderDataset(image_dir, augmentation)
    else:
        return CSVDataset(image_dir=image_dir, csv_path=csv_path, augmentation=augmentation )


class CSVDataset(Dataset):
    def __init__(self, image_dir, csv_path,  augmentation: bool = False):
        self.image_dir = image_dir
        transforms_list = [
            torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            torchvision.transforms.ToTensor(),
        ]

        if augmentation:
            transforms_list = [
                                  torchvision.transforms.RandomAffine(degrees=(-30, 30), scale=(0.8, 1.2), resample=Image.BICUBIC),
                                  torchvision.transforms.Resize((int(IMAGE_SIZE * 1.055), int(IMAGE_SIZE * 1.055)), interpolation=3),
                                  torchvision.transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
                                  torchvision.transforms.RandomHorizontalFlip(),
                                  torchvision.transforms.ColorJitter(0.2, 0.2, 0.2),
                                  RandomNoise(),
                                  RandomBlur(),
                              ] + transforms_list

        self.transforms = torchvision.transforms.Compose(transforms_list)
        self.data_info = pd.read_csv(csv_path, header=0)
        self.tags_list = self.data_info.columns.tolist()
        print(self.tags_list)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1:DIMENSION + 1])
        self.data_len = len(self.data_info.index)
        self.statistics()

    def __getitem__(self, index):
        single_image_name = os.path.join(self.image_dir,self.image_arr[index])
        img_as_img = load_image(single_image_name)
        image = self.transforms(img_as_img)
        target = torch.from_numpy(self.label_arr[index]).float()
        target = (target - self.min)/ (self.max - self.min)
        return image, target

    def __len__(self):
        return self.data_len

    def statistics(self):
        print('maximum value = ', np.max(self.label_arr))
        self.max = float(np.max(self.label_arr))
        print('minimum value = ', np.min(self.label_arr))
        self.min = float(np.min(self.label_arr))
        print('average value = ', np.mean(self.label_arr))
        self.mean = float(np.mean(self.label_arr))
        print('dispersion value = ', np.std(self.label_arr))
        self.std = float(np.std(self.label_arr))
        print(self.label_arr.shape)

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
    def __init__(self, image_dir, augmentation : bool = False):
        classes, class_to_idx = find_classes(image_dir)
        samples = make_dataset(image_dir, class_to_idx)
        self.image_dir = image_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        transforms_list = [
            torchvision.transforms.CenterCrop(IMAGE_SIZE),
            torchvision.transforms.ToTensor(),
        ]

        if augmentation:
            transforms_list = [
                                  torchvision.transforms.RandomHorizontalFlip(),
                                  torchvision.transforms.ColorJitter(0.2, 0.2, 0.1),
                                  torchvision.transforms.RandomRotation(10),
                              ] + transforms_list

        self.transforms = torchvision.transforms.Compose(transforms_list)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = load_image(path)
        sample = self.transform(sample)
        target = torch.FloatTensor([target])
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


class RandomBlur(object):
    def __init__(self):
        self.blurring_filters = [ImageFilter.GaussianBlur, ImageFilter.BoxBlur]
        self.radius = [0, 1]

    def __call__(self, input):
        index = int(random.uniform(0,  len(self.blurring_filters)))
        radius = np.random.choice(self.radius)
        blurring_filter = self.blurring_filters[index](radius)
        return input.filter(blurring_filter)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.blurring_filters)

class RandomNoise(object):
    def __init__(self):
        self.noises = [GaussianNoise, UniformNoise]
        self.factors = [0,0.02]

    def __call__(self, input):
        factor = np.random.choice(self.factors)
        index = int(random.uniform(0, len(self.noises)))
        noise = self.noises[index](factor)
        return noise(input)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.noises)


class GaussianNoise(object):
    def __init__(self, factor: float = 0.1):
        self.factor = factor

    def __call__(self, input):
        img = np.array(input)
        img = img.astype(dtype=np.float32)
        noisy_img = img + np.random.normal(0.0, 255.0 * self.factor, img.shape)
        noisy_img = np.clip(noisy_img, 0.0, 255.0)
        noisy_img = noisy_img.astype(dtype=np.uint8)
        return Image.fromarray(noisy_img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.factor)


class UniformNoise(object):
    def __init__(self, factor: float = 0.1):
        self.factor = factor

    def __call__(self, input):
        img = np.array(input)
        img = img.astype(dtype=np.float32)
        noisy_img = img + np.random.uniform(0.0, 255.0 * self.factor, img.shape)
        noisy_img = np.clip(noisy_img, 0.0, 255.0)
        noisy_img = noisy_img.astype(dtype=np.uint8)
        return Image.fromarray(noisy_img)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.factor)



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
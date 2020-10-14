import glob
import json
import random
import re
import os
from os.path import join

import cv2
import torch
import torchvision
from DLBio.helpers import check_mkdir, search_in_all_subfolders
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np

import user_config as config

CLASSES = [
    'IgA-Kappa',
    'IgA-Lambda',
    'IgG-Kappa',
    'IgG-Lambda',
    'IgM-Kappa',
    'IgM-Lambda',
    'Negative'
]


def get_dataloader(is_train, batch_size, split_index=0):
    split = _find_split_file(int(split_index))
    
    # get class distribution and build a data sampler
    weights = split['weights']
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    if is_train:
        dataset = ImfxImDataset(split['train'], _get_data_aug(is_train))
        return DataLoader(dataset, batch_size=batch_size, sampler = sampler)
    else:
        dataset = ImfxImDataset(split['test'], _get_data_aug(is_train))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _get_data_aug(is_train, crop_size=224):
    if is_train:
        aug = [
            torchvision.transforms.ToPILImage('RGB'),
            torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True),
            torchvision.transforms.Resize(
                (crop_size, crop_size), interpolation=2),
            torchvision.transforms.ToTensor(),
            # using imagenet normalization
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
    else:
        aug = [
            torchvision.transforms.ToPILImage('RGB'),
            #torchvision.transforms.Pad(crop_size // 2),
            # torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.Resize(
                (crop_size, crop_size), interpolation=2),
            torchvision.transforms.ToTensor(),
            # using imagenet normalization
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]

    return torchvision.transforms.Compose(aug)


class ImfxImDataset(Dataset):
    def __init__(self, paths, augmentation):
        def load_image(x):
            return cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)

        self.images = [load_image(x) for x in paths]
        self.labels = [_get_class(x) for x in paths]

        self.aug = augmentation

    def __getitem__(self, index):
        x = self.images[index]
        x = self.aug(x)

        y = torch.tensor([self.labels[index]]).long()

        return x, y

    def __len__(self):
        return len(self.labels)


def _find_split_file(index):
    def get_index(x):
        rgx = r'.*(\/|\\)(\d+).json'
        return int(re.match(rgx, x).group(2))

    for file in glob.glob(join(config.IMFX_IM_BASE, 'splits', '*.json')):
        if get_index(file) == index:
            with open(file, 'r') as file:
                split = json.load(file)
                return split

    raise ValueError(f'Could not find split{index}')


def _get_class(x):
    rgx = r'.*(\/|\\)(.*)_\d+.jpg'
    class_name = re.match(rgx, x).group(2)
    return CLASSES.index(class_name)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def create_splits(num_splits=5, split_perc=.8):
    images_ = _get_images_sorted_by_class()
    
    # calculate class distribution
    class_counts = [0] * len(images_)
    for key in images_:
        class_counts[key] = len(images_[key])
    N = sum(class_counts)
    class_weights = [0.] * len(images_)
    for i in range(len(class_weights)):
        class_weights[i] = N / float(class_counts[i])

    for i in range(num_splits):
        train_images = []
        train_weights = []
        test_images = []

        # for each class get split_perc % for training
        for key, val in images_.items():
            n = len(val)
            n_train = int(split_perc * n)

            # grab images without replacement
            tmp_train = random.sample(val, n_train)
            tmp_weights = [class_weights[key]] * n_train
            tmp_test = list(set(val) - set(tmp_train))

            train_images += tmp_train
            train_weights += tmp_weights
            test_images += tmp_test

        out_path = join(config.IMFX_IM_BASE, 'splits', f'{i}.json')
        out_path = out_path.replace('/', os.sep)
        check_mkdir(out_path)

        with open(out_path, 'w') as file:
            json.dump({
                'train': train_images,
                'weights' : train_weights,
                'test': test_images
            }, file)


def _get_images_sorted_by_class():
    all_images = search_in_all_subfolders(
        r'(.*)_(\d+).jpg', config.IMFX_IM_BASE)
    images_by_class = dict()
    for x in all_images:
        index = _get_class(x)
        if index not in images_by_class.keys():
            images_by_class[index] = [x]
        else:
            images_by_class[index].append(x)
    return images_by_class

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _test_splits():
    all_images = search_in_all_subfolders(
        r'(.*)_(\d+).jpg', config.IMFX_IM_BASE)

    for file in glob.glob(join(config.IMFX_IM_BASE, 'splits', '*.json')):
        with open(file, 'r') as file:
            split = json.load(file)

        assert set(split['train'] + split['test']) == set(all_images)
        assert not set(split['train']).intersection(set(split['test']))

    print('Test succeeded.')


def _debug_dataset():
    import matplotlib.pyplot as plt
    from DLBio.pytorch_helpers import cuda_to_numpy
    from DLBio.helpers import to_uint8_image

    data_loader = get_dataloader(True, 16, split_index=0)
    for x, y in data_loader:
        print(y)

        for b in range(x.shape[0]):
            tmp = cuda_to_numpy(x[b, ...])
            tmp = to_uint8_image(tmp)

            plt.imshow(tmp)
            plt.title(int(y[b]))
            plt.savefig('debug.png')
            plt.close()

            xxx = 0


if __name__ == "__main__":
    create_splits(num_splits=5, split_perc=0.8)
    _test_splits()
    # _debug_dataset()

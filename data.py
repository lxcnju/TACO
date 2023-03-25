import os
import copy
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms

import numpy as np

from paths import miniimagenet_split_path
from paths import miniimagenet_path

from rand_augment import RandAugment


def load_miniimagenet_fname_labels(setname):
    fpath = os.path.join(
        miniimagenet_split_path, "{}.csv".format(setname)
    )

    fnames, labels = [], []
    with open(fpath, "r", encoding="utf-8") as fr:
        for line in fr.readlines()[1:]:
            line = line.strip()
            if len(line) <= 0:
                continue
            fname, label = line.split(",")
            fnames.append(fname)
            labels.append(label)
    return fnames, labels


def get_miniimagenet_labels():
    all_labels = []
    for setname in ["train", "val", "test"]:
        _, labels = load_miniimagenet_fname_labels(setname)
        all_labels.extend(labels)
    all_labels = list(sorted(set(all_labels)))
    labels_dict = {
        label: i for i, label in enumerate(all_labels)
    }
    return labels_dict


def load_miniimagenet_infos(setname):
    fnames, labels = load_miniimagenet_fname_labels(setname)
    labels_dict = get_miniimagenet_labels()
    labels = [labels_dict[label] for label in labels]
    return fnames, labels


class MiniImagenetDataset(data.Dataset):
    def __init__(
        self, setname,
        read_mem=0,
        backbone="ConvNet",
        rand_aug=False,
        is_train=True
    ):
        self.read_mem = read_mem
        fnames, labels = load_miniimagenet_infos(setname)

        if self.read_mem:
            self.images = [
                Image.open(
                    os.path.join(miniimagenet_path, fname)
                ).convert("RGB") for fname in fnames
            ]
        else:
            self.images = fnames
        self.labels = labels

        if backbone == 'ConvNet':
            image_size = 84
        elif backbone == 'ResNet':
            image_size = 80
        else:
            raise ValueError("No such backbone: {}".format(backbone))

        self.transform_train = transforms.Compose([
            transforms.Resize(92),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225])
            )
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225])
            )
        ])

        if is_train is True:
            self.transform = self.transform_train
            print("Using train augmentation!")
            if rand_aug is True:
                self.transform.transforms.insert(
                    0, RandAugment(2, 9)
                )
                print("Using random augmentation!")
        else:
            self.transform = self.transform_test
            print("Using test augmentation!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.read_mem:
            img = self.images[index]
        else:
            img = Image.open(
                os.path.join(miniimagenet_path, self.images[index])
            ).convert("RGB")
        label = self.labels[index]

        img = self.transform(img)
        label = torch.LongTensor([label])[0]
        return img, label


class CategoriesSampler():

    def __init__(self, labels, n_task, n_c, n_per_class):
        self.n_task = n_task
        self.n_c = n_c
        self.n_per_class = n_per_class

        labels = np.array(labels)
        self.label_arr = []
        for c in sorted(np.unique(labels)):
            inds = np.argwhere(labels == c).reshape(-1)
            inds = torch.from_numpy(inds)
            self.label_arr.append(inds)

    def __len__(self):
        return self.n_task

    def __iter__(self):
        for i in range(self.n_task):
            batch = []

            # sample classes
            classes = torch.randperm(len(self.label_arr))[:self.n_c]
            for c in classes:
                c_inds = self.label_arr[c]
                sam_inds = torch.randperm(len(c_inds))[:self.n_per_class]
                batch.append(c_inds[sam_inds])

            # return [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, ...]
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class MiniImagenetUnlabelDataset(data.Dataset):
    def __init__(
        self,
        unlabel_setname,
        read_mem=0,
        backbone="ConvNet",
        rand_aug=False,
        is_train=True
    ):
        self.read_mem = read_mem
        fnames, _ = load_miniimagenet_infos(unlabel_setname)

        if self.read_mem:
            self.images = [
                Image.open(
                    os.path.join(miniimagenet_path, fname)
                ).convert("RGB") for fname in fnames
            ]
        else:
            self.images = fnames

        if backbone == 'ConvNet':
            image_size = 84
        elif backbone == 'ResNet':
            image_size = 80
        else:
            raise ValueError("No such backbone: {}".format(backbone))

        self.transform_train = transforms.Compose([
            transforms.Resize(92),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225])
            )
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225])
            )
        ])

        if is_train is True:
            self.transform = self.transform_train
            print("Using train augmentation!")
            if rand_aug is True:
                self.transform.transforms.insert(
                    0, RandAugment(2, 9)
                )
                print("Using random augmentation!")
        else:
            self.transform = self.transform_test
            print("Using test augmentation!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.read_mem:
            img = self.images[index]
        else:
            img = Image.open(
                os.path.join(miniimagenet_path, self.images[index])
            ).convert("RGB")

        img = self.transform(img)
        return img


class MiniImagenetAugDataset(data.Dataset):
    def __init__(
        self, setname,
        read_mem=0,
        backbone="ConvNet",
        rand_aug=False,
        is_train=True
    ):
        self.read_mem = read_mem
        fnames, labels = load_miniimagenet_infos(setname)

        if self.read_mem:
            self.images = [
                Image.open(
                    os.path.join(miniimagenet_path, fname)
                ).convert("RGB") for fname in fnames
            ]
        else:
            self.images = fnames
        self.labels = labels

        if backbone == 'ConvNet':
            image_size = 84
        elif backbone == 'ResNet':
            image_size = 80
        else:
            raise ValueError("No such backbone: {}".format(backbone))

        self.transform = transforms.Compose([
            transforms.Resize(92),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225])
            )
        ])

        if is_train is True:
            print("Using train augmentation!")
            if rand_aug is True:
                self.transform.transforms.insert(
                    0, RandAugment(2, 9)
                )
                print("Using random augmentation!")
        else:
            print("Using test augmentation!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.read_mem:
            img = self.images[index]
        else:
            img = Image.open(
                os.path.join(miniimagenet_path, self.images[index])
            ).convert("RGB")
        label = self.labels[index]

        img1 = self.transform(img)
        img2 = self.transform(img)
        label = torch.LongTensor([label])[0]
        return img1, img2, label


if __name__ == "__main__":
    labels_dict = get_miniimagenet_labels()
    print(labels_dict)

    fnames, labels = load_miniimagenet_infos("train")
    print(len(fnames))
    print(np.unique(labels))
    print(len(np.unique(labels)))

    fnames, labels = load_miniimagenet_infos("train_label_0.3_C")
    print(len(fnames))
    print(np.unique(labels))
    print(len(np.unique(labels)))

    fnames, labels = load_miniimagenet_infos("train_label_0.3_I")
    print(len(fnames))
    print(np.unique(labels))
    print(len(np.unique(labels)))

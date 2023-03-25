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


class SemiMiniImagenetDataset(data.Dataset):
    def __init__(
        self, setname,
        unlabel_setname,
        read_mem=False,
        backbone="ConvNet",
        rand_aug=False,
        is_train=True
    ):
        self.read_mem = read_mem

        # load labeled data
        fnames, labels = load_miniimagenet_infos(setname)

        if self.read_mem is True:
            self.images = [
                Image.open(
                    os.path.join(miniimagenet_path, fname)
                ).convert("RGB") for fname in fnames
            ]
        else:
            self.images = fnames
        self.labels = labels

        # load unlabeled data
        un_fnames, un_labels = load_miniimagenet_infos(unlabel_setname)

        if self.read_mem is True:
            self.un_images = [
                Image.open(
                    os.path.join(miniimagenet_path, fname)
                ).convert("RGB") for fname in un_fnames
            ]
        else:
            self.un_images = fnames
        self.un_labels = un_labels

        if backbone == 'ConvNet':
            image_size = 84
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
        elif backbone == 'ResNet':
            image_size = 80
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
        else:
            raise ValueError("No such backbone: {}".format(backbone))

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
        if index[1] == 1:
            if self.read_mem is True:
                img = self.images[index]
            else:
                img = Image.open(
                    os.path.join(miniimagenet_path, self.images[index])
                ).convert("RGB")
            label = self.labels[index]
        elif index[1] == -1:
            if self.read_mem is True:
                img = self.un_images[index]
            else:
                img = Image.open(
                    os.path.join(miniimagenet_path, self.un_images[index])
                ).convert("RGB")
            label = -1

        img = self.transform(img)
        label = torch.LongTensor([label])[0]
        return img, label


class SemiCategoriesSampler():

    def __init__(
        self, labels, n_un_label, n_task, n_c, n_per_class, n_un_per_class
    ):
        self.n_task = n_task
        self.n_c = n_c
        self.n_per_class = n_per_class
        self.n_un_per_class = n_un_per_class

        labels = np.array(labels)
        self.label_arr = []
        for c in sorted(np.unique(labels)):
            inds = np.argwhere(labels == c).reshape(-1)
            inds = torch.from_numpy(inds)
            self.label_arr.append(inds)

        self.un_labels = np.arange(n_un_label)

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

            # unlabel index
            pos = torch.randperm(len(self.un_labels))[
                0:self.n_un_per_class * self.n_c
            ]
            un_batch = torch.from_numpy(
                self.un_labels[pos]
            ).type(torch.LongTensor)

            batch = torch.cat([batch, un_batch], 0)
            index_label = torch.cat([
                torch.ones(self.n_c * self.n_per_class),
                -torch.ones(self.n_c * self.n_un_per_class)
            ], dim=0).type(torch.LongTensor)
            batch = torch.cat([
                batch.view(-1, 1), index_label.view(-1, 1)
            ], dim=1)

            yield batch


if __name__ == "__main__":
    dset = SemiMiniImagenetDataset(setname="train")

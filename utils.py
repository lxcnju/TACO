import os
import time
import pandas as pd
import pickle

import torch
import torch.nn as nn

try:
    import moxing as mox

    def read_file(path):
        with mox.file.File(path, 'r') as fr:
            da_df = pd.read_csv(
                fr, index_col=False, header=None
            )
        return da_df

    def save_data(da_df, path):
        with mox.file.File(path, 'w') as fr:
            da_df.to_csv(fr)
        print("File saved in {}.".format(path))

    def load_pickle(fpath):
        with mox.file.File(fpath, "rb") as fr:
            data = pickle.load(fr)
        return data

    def append_to_logs(fpath, logs):
        with mox.file.File(fpath, "a") as fa:
            for log in logs:
                fa.write("{}\n".format(log))
            fa.write("\n")

except Exception:
    def read_file(path):
        da_df = pd.read_csv(
            path, index_col=False, header=None
        )
        return da_df

    def save_data(da_df, path):
        da_df.to_csv(path)
        print("File saved in {}.".format(path))

    def load_pickle(fpath):
        with open(fpath, "rb") as fr:
            data = pickle.load(fr)
        return data

    def append_to_logs(fpath, logs):
        with open(fpath, "a", encoding="utf-8") as fa:
            for log in logs:
                fa.write("{}\n".format(log))
            fa.write("\n")


def listfiles(fdir):
    for root, dirs, files in os.walk(fdir):
        print(root, dirs, files)


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.1)
        except Exception:
            pass
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        try:
            m.bias.data.zero_()
        except Exception:
            pass
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        try:
            m.bias.data.zero_()
        except Exception:
            pass


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

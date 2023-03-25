import os
import copy
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CategoriesSampler

from protonet import ProtoNet

from paths import pretrain_fpaths
from paths import cur_dir

from utils import set_gpu, Averager, count_acc
from utils import append_to_logs
from utils import Timer


def load_pretrained_model(model, args):
    model_dict = model.state_dict()
    if args.pretrain is True:
        pretrain_fpath = pretrain_fpaths[args.backbone]
        pretrained_dict = torch.load(pretrain_fpath)['params']
        if args.backbone == "ConvNet":
            pretrained_dict = {
                "encoder." + k: v for k, v in pretrained_dict.items()
            }
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict
        }
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Pretrained model loaded!")
    return model


def construct_loaders(args):
    if args.dataset == "MiniImageNet":
        from data import MiniImagenetDataset as Dataset
    else:
        raise ValueError("No such dataset: {}".format(args.dataset))

    train_set = Dataset(
        setname=args.train_set,
        read_mem=args.read_mem,
        backbone=args.backbone,
        rand_aug=args.rand_aug,
        is_train=True,
    )
    print(np.unique(train_set.labels))
    print(np.unique(train_set.labels).shape)

    train_sampler = CategoriesSampler(
        labels=train_set.labels,
        n_task=args.n_train_task,
        n_c=args.n_way,
        n_per_class=2 * args.n_shot + args.n_query
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_set = Dataset(
        setname=args.val_set,
        read_mem=args.read_mem,
        backbone=args.backbone,
        rand_aug=args.rand_aug,
        is_train=False
    )
    print(len(val_set.labels))
    print(np.unique(val_set.labels))
    val_sampler = CategoriesSampler(
        labels=val_set.labels,
        n_task=args.n_val_task,
        n_c=args.n_way,
        n_per_class=args.n_shot + args.n_query
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_set = Dataset(
        setname=args.test_set,
        read_mem=args.read_mem,
        backbone=args.backbone,
        rand_aug=args.rand_aug,
        is_train=False
    )
    test_sampler = CategoriesSampler(
        labels=test_set.labels,
        n_task=args.n_test_task,
        n_c=args.n_way,
        n_per_class=args.n_shot + args.n_query
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    rets = (
        train_loader,
        val_loader, test_loader
    )
    return rets


def construct_optimizer(model, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr
        )
    else:
        raise ValueError("No such backbone: {}".format(args.backbone))
    return optimizer


def train(model, train_loader, optimizer, args):
    model.train()
    avg_loss = Averager()
    avg_acc = Averager()

    labels = torch.arange(args.n_way).repeat(args.n_query).long()
    if args.cuda:
        labels = labels.cuda()

    print_info = True
    for i, batch in enumerate(train_loader, 1):
        data = batch[0]

        if args.cuda:
            data = data.cuda()

        # First Encode
        embeds = model.encoder(data)

        p = args.n_shot * args.n_way
        data_shot_1 = embeds[:p]
        data_shot_2 = embeds[p:2 * p]
        data_query = embeds[2 * p:]

        if print_info is True:
            print(data_shot_1.shape, data_shot_2.shape)
            print(data_query.shape)
            print_info = False

        logits_1 = model.logits(data_shot_1, data_query)
        loss_1 = F.cross_entropy(logits_1, labels)

        logits_2 = model.logits(data_shot_2, data_query)
        loss_2 = F.cross_entropy(logits_2, labels)

        loss = 0.5 * (loss_1 + loss_2)

        acc_1 = count_acc(logits_1, labels)
        acc_2 = count_acc(logits_2, labels)
        acc = 0.5 * (acc_1 + acc_2)

        # TACO
        kl_loss_1 = F.kl_div(
            F.log_softmax(logits_1, dim=1),
            F.softmax(logits_2.detach() / args.kl_tau, dim=1),
            reduction="batchmean"
        )
        kl_loss_2 = F.kl_div(
            F.log_softmax(logits_2, dim=1),
            F.softmax(logits_1.detach() / args.kl_tau, dim=1),
            reduction="batchmean"
        )
        loss += args.kl_lamb * (kl_loss_1 + kl_loss_2) / 2.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss.add(loss.item())
        avg_acc.add(acc)

    loss = avg_loss.item()
    acc = avg_acc.item()
    return loss, acc


def test(model, loader, args):
    model.eval()

    avg_acc = Averager()

    labels = torch.arange(args.n_way).repeat(args.n_query).long()
    if args.cuda:
        labels = labels.cuda()

    print_info = True
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            data = batch[0]
            if args.cuda:
                data = data.cuda()

            p = args.n_shot * args.n_way
            data_shot, data_query = data[:p], data[p:]

            if print_info is True:
                print("Test: ", data_shot.shape, data_query.shape)
                print_info = False

            logits = model(data_shot, data_query)
            acc = count_acc(logits, labels)

            avg_acc.add(acc)

    acc = avg_acc.item()
    return acc


def main(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    (
        train_loader,
        val_loader, test_loader
    ) = construct_loaders(args)

    model = ProtoNet(args)
    model = load_pretrained_model(model, args)

    set_gpu(args.gpu)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    optimizer = construct_optimizer(model, args)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    best_val_epoch = 0
    best_val_acc = 0.0
    best_model = copy.deepcopy(model.cpu())

    timer = Timer()
    for epoch in range(1, args.max_epoch + 1):
        if args.cuda:
            model = model.cuda()

        train_loss, train_acc = train(
            model, train_loader, optimizer, args
        )
        val_acc = test(
            model, val_loader, args
        )
        lr_scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model.cpu())

        print("[Ep:{}]\t[TrainLoss:{}]\t[TrainAcc:{:.5f}]".format(
            epoch, train_loss, train_acc
        ))
        print("[Ep:{}]\t[BestEp:{}]\t[BeVaAc:{:.5f}]\t[Vacc:{:.5f}]".format(
            epoch, best_val_epoch, best_val_acc, val_acc
        ))
        print('ETA:{}/{}'.format(
            timer.measure(), timer.measure(epoch / args.max_epoch)
        ))

    if args.cuda:
        best_model = best_model.cuda()

    test_acc = test(best_model, test_loader, args)
    print("[BestEp:{}]\t[BestValAcc:{:.5f}]\t[TeAcc:{:.5f}]".format(
        best_val_epoch, best_val_acc, test_acc
    ))

    res = {
        "best_val_epoch": best_val_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc
    }
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--read_mem", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    candi_param_dict = {
        "dataset": ["MiniImageNet"],
        "backbone": ["ConvNet"],
        "similarity": ["cosine"],
        "pretrain": [True],
        "rand_aug": [False],
        "train_set": ["train"],
        "val_set": ["val"],
        "test_set": ["test"],
        "n_train_task": [100],
        "n_val_task": [100],
        "n_test_task": [1000],
        "n_way": [5],
        "n_shot": [1],
        "n_query": [15],
        "max_epoch": [200],
        "optimizer": ["SGD"],
        "momentum": [0.9],
        "lr": [0.0001],
        "kl_lamb": [1.0],
        "kl_tau": [1.0],
        "step_size": [50],
        "gamma": [0.2],
        "weight_decay": [5e-4],
        "dis_tau": [1.0],
        "cuda": [True],
        "gpu": [args.gpu],
        "read_mem": [args.read_mem],
        "num_workers": [args.num_workers],
        "fname": ["sup-taco.log"],
    }

    set_gpu(args.gpu)
    for backbone in ["ResNet", "ConvNet"]:
        for n_shot in [1, 5]:
            for rand_aug in [False, True]:
                for kl_lamb in [0.0, 1.0]:
                    if kl_lamb == 0.0:
                        kl_taus = [1.0]
                    else:
                        kl_taus = [0.25, 1.0, 4.0]
                    for kl_tau in kl_taus:
                        all_res = {}
                        for _ in range(1):
                            para_dict = {}
                            for k, vs in candi_param_dict.items():
                                para_dict[k] = random.choice(vs)

                                para_dict["backbone"] = backbone
                                para_dict["n_shot"] = n_shot
                                para_dict["rand_aug"] = rand_aug
                                para_dict["kl_tau"] = kl_tau
                                para_dict["kl_lamb"] = kl_lamb

                            res = main(para_dict)

                            for k, v in res.items():
                                if k in all_res.keys():
                                    all_res[k].append(v)
                                else:
                                    all_res[k] = [v]

                        # record
                        fpath = os.path.join(
                            cur_dir, "final_" + para_dict["fname"]
                        )
                        algoname = " ".join([
                            "{}:{}".format(
                                k, v
                            ) for k, v in para_dict.items()
                        ])

                        logs = []
                        logs.append(algoname)

                        for k, vs in all_res.items():
                            str_res = "[{}]\t[Mean={:.5f}]\t[Std={:.5f}]".format(
                                k, np.mean(vs), np.std(vs)
                            )
                            logs.append(str_res)

                        append_to_logs(fpath, logs)

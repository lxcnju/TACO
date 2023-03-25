import torch
import torch.nn as nn

from networks import AlexNetBackbone
from networks import ResNet50Backbone

from paths import alexnet_pretrain_fpath
from paths import resnet50_pretrain_fpath

from utils import weights_init


class FinetuneNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.backbone == "AlexNet":
            if args.pretrain is True:
                self.encoder = AlexNetBackbone(alexnet_pretrain_fpath)
            else:
                self.encoder = AlexNetBackbone()
            self.n_embedding = 256 * 6 * 6
        elif args.backbone == "ResNet50":
            if args.pretrain is True:
                self.encoder = ResNet50Backbone(resnet50_pretrain_fpath)
            else:
                self.encoder = ResNet50Backbone()
            self.n_embedding = 2048
        else:
            raise ValueError("No such backbone: {}".format(args.backbone))

        if args.use_bottleneck is True:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.n_embedding, 128),
                nn.ReLU()
            )

            self.fc = nn.Sequential(
                nn.Linear(128, args.n_classes),
            )

            self.bottleneck.apply(weights_init)
            self.fc.apply(weights_init)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.n_embedding, args.n_classes),
            )

            self.fc.apply(weights_init)

    def forward(self, xs):
        hs = self.encoder(xs)

        if self.args.use_bottleneck is True:
            hs = self.bottleneck(hs)

        logits = self.fc(hs)
        return logits

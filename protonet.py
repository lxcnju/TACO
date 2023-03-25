import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import euclidean_metric

class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone == 'ConvNet':
            from networks import ConvNet
            self.encoder = ConvNet()
        elif args.backbone == "ResNet":
            from networks import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError("No such backbone: {}".format(args.backbone))

    def logits(self, shot, query):
        n_hidden = shot.shape[-1]
        proto = shot.view((self.args.n_shot, -1, n_hidden)).mean(dim=0)

        if self.args.similarity == "distance":
            logits = euclidean_metric(query, proto) / self.args.dis_tau
        elif self.args.similarity == "cosine":
            proto = F.normalize(proto, p=2, dim=-1)
            logits = torch.mm(query, proto.t())
        return logits

    def forward(self, data_shot, data_query):
        shot = self.encoder(data_shot)
        query = self.encoder(data_query)
        logits = self.logits(shot, query)
        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

from utils import euclidean_metric


class TacoSemiProtoNet(nn.Module):

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

        self.detector = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )

    def logits(self, proto, query):
        if self.args.similarity == "distance":
            logits = euclidean_metric(query, proto) / self.args.dis_tau
        elif self.args.similarity == "cosine":
            proto = F.normalize(proto, p=2, dim=-1)
            logits = torch.mm(query, proto.t())
        return logits

    def get_proto(self, x_shot, x_pool):
        label_support = torch.arange(self.args.n_way).repeat(
            self.args.n_shot
        ).type(torch.LongTensor)
        label_support_onehot = torch.zeros(
            self.args.n_way * self.args.n_shot, self.args.n_way
        )
        label_support_onehot.scatter_(1, label_support.unsqueeze(1), 1)
        if self.args.cuda is True:
            label_support_onehot = label_support_onehot.cuda()

        proto_shot = x_shot.view((
            self.args.n_shot, self.args.n_way, -1
        )).mean(dim=0)

        dis = euclidean_metric(x_pool, proto_shot) / self.args.dis_tau

        n_pool = dis.shape[0]
        if self.args.use_detector:
            stats_dis = -1.0 * dis
            stats_dis = stats_dis / torch.mean(stats_dis, dim=0)
            stats_dis_off = stats_dis.detach().cpu().numpy()

            stats_min = np.min(stats_dis_off, axis=0)
            stats_max = np.max(stats_dis_off, axis=0)
            stats_var = np.var(stats_dis_off, axis=0)
            stats_skew = stats.skew(stats_dis_off, axis=0)
            stats_kurt = stats.kurtosis(stats_dis_off, axis=0)

            stats_data = np.stack([
                stats_min, stats_max, stats_var, stats_skew, stats_kurt
            ], axis=0).transpose()
            stats_data = torch.FloatTensor(stats_data)
            if self.args.cuda is True:
                stats_data = stats_data.cuda()

            outputs = self.detector(stats_data)
            beta = outputs[:, 0].view((1, -1)).repeat((n_pool, 1))
            gamma = outputs[:, 1].view((1, -1)).repeat((n_pool, 1))
            m_adjust = F.sigmoid(-1.0 * gamma * (stats_dis - beta))
        else:
            m_adjust = torch.ones((n_pool, self.args.n_way))
            if self.args.cuda is True:
                m_adjust = m_adjust.cuda()

        z_hat = nn.Softmax(dim=1)(dis) * m_adjust
        z = label_support_onehot.view((-1, self.args.n_way))

        z = torch.cat([z, z_hat], dim=0)
        h = torch.cat([x_shot, x_pool], dim=0)

        proto = torch.mm(z.t(), h)

        sum_z = z.sum(dim=0).view((-1, 1))
        proto = proto / sum_z
        return proto

    def semiproto_logits(self, embed_shot, embed_query, embed_unlabel):
        proto = self.get_proto(embed_shot, embed_unlabel)
        logits = self.logits(proto, embed_query)
        return logits

    def forward(self, data_shot, data_query, data_unlabel):
        embed_shot = self.encoder(data_shot)
        embed_query = self.encoder(data_query)
        embed_unlabel = self.encoder(data_unlabel)

        logits = self.semiproto_logits(embed_shot, embed_query, embed_unlabel)
        return logits

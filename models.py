import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.autograd import Variable


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        h1 = self.W(x)
        return h1


class SGC_multi_hid(nn.Module):
    """
    Morton added.
    """

    def __init__(self, nfeat, nclass, dropout):
        super(SGC_multi_hid, self).__init__()

        self.W1 = nn.Linear(nfeat, 256, bias=True)
        self.W2 = nn.Linear(256, nclass, bias=True)
        # self.W3 = nn.Linear(600, nclass, bias=True)
        self.dropout = dropout

    def forward(self, x):
        x = self.W1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.W2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.W3(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output


class GCN(nn.Module):
    """
    A Two-layer GCN.
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 1024)
        self.gc3 = GraphConvolution(1024, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return x


class HGNN(nn.Module):
    """
    A Two-layer HGNN.
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(HGNN, self).__init__()
        print("HGNN model starting...")

        self.cluster_W_1 = nn.Linear(nfeat, nfeat, bias=True)
        self.user_W_1 = nn.Linear(2 * nfeat, 2 * nfeat, bias=True)
        self.cluster_W_2 = nn.Linear(2 * nfeat, 2 * nfeat, bias=True)
        self.user_W_2 = nn.Linear(4 * nfeat, 4 * nfeat, bias=True)
        self.output_W = nn.Linear(4 * nfeat, nclass, bias=True)
        self.dropout = dropout
        # self.cluster_feat_new = Variable(torch.zeros(nclass, nfeat))

    def forward(self, features, node2cluster_arr, cluster_nodes=None, cluster_adj=None):
        if cluster_nodes is None and cluster_adj is None:
            feat_with_hops_1 = torch.mm(node2cluster_arr, self.cluster_feat_new_1)
            user_features_1 = self.user_W_1(torch.cat([features, feat_with_hops_1], dim=1))
            feat_with_hops_2 = torch.mm(node2cluster_arr, self.cluster_feat_new_2)
            user_features_2 = self.user_W_2(torch.cat([user_features_1, feat_with_hops_2], dim=1))

            out = self.output_W(user_features_2)
            out = F.dropout(out, self.dropout, training=self.training)
            return out
        else:
            cluster_feat_1 = torch.mm(cluster_nodes, features)
            self.cluster_feat_new_1 = self.cluster_W_1(torch.mm(cluster_adj, cluster_feat_1))
            feat_with_hops_1 = torch.mm(node2cluster_arr, self.cluster_feat_new_1)
            user_features_1 = self.user_W_1(torch.cat([features, feat_with_hops_1], dim=1))
            # user_features = F.dropout(user_features, self.dropout, training=self.training)

            cluster_feat_2 = torch.mm(cluster_nodes, user_features_1)
            self.cluster_feat_new_2 = self.cluster_W_2(torch.mm(cluster_adj, cluster_feat_2))
            feat_with_hops_2 = torch.mm(node2cluster_arr, self.cluster_feat_new_2)
            user_features_2 = self.user_W_2(torch.cat([user_features_1, feat_with_hops_2], dim=1))
            # user_features = F.dropout(user_features, self.dropout, training=self.training)

            out = self.output_W(user_features_2)
            # out = F.dropout(out, self.dropout, training=self.training)
            return out


def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0.1, usecuda=False):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)

    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass)

    elif model_opt == "SGC_multi_hid":
        model = SGC_multi_hid(nfeat=nfeat,
                              nclass=nclass,
                              dropout=dropout)
    elif model_opt == "HGNN":
        model = HGNN(nfeat=nfeat,
                     nhid=nhid,
                     nclass=nclass,
                     dropout=dropout)

    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if usecuda:
        model.cuda()
    return model

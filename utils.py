# -*- coding: UTF-8 -*-
import torch
import argparse
import numpy as np

from sklearn.metrics import f1_score


# from torch_sparse import spspmm


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro


def shufflelists(lists):  # 多个序列以相同顺序打乱
    ri = np.random.permutation(len(lists[1]))
    out = []
    for l in lists:
        out.append(l[ri])
    return out


def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """
    parser = argparse.ArgumentParser()

    # data_args: control the data loading
    parser.add_argument('-dir', metavar='str', help='the detail directory of dataset',
                        type=str, default='./data/cmu/')
    parser.add_argument('-dump_file', metavar='str', help='the dir to load file include name',
                        type=str, default='./data/cmu/dump_doc_dim_128_for_hgnn.pkl')
    parser.add_argument('-feature_norm', type=str, choices=['None', 'Standard', 'Mean'], default='Standard')

    parser.add_argument('-doc2vec_model_file', metavar='str', help='the dir to load doc2vec_model_file .bin',
                        type=str, default="./data/cmu/train_corpus/model_dim_128_epoch_40.bin")
    parser.add_argument('-res_file', metavar='str', help='the dir to save the result file',
                        type=str, default='./result/res_cmu_hgnn_dim_128.txt')

    # -dir ./data/cmu/ -bucket 50 -celebrity 5
    # -dir ./data/na/ -bucket 2400 -celebrity 15
    # -dir ./data/world/ -bucket 2400 -celebrity 5
    parser.add_argument('-edge_dis_file', metavar='str', help='the dir to load all edges distance file',
                        type=str, default='./my_assets/no_ues_now/edge_dis.pkl')
    parser.add_argument('-bucket', metavar='int', help='discretisation bucket size', type=int, default=50)
    parser.add_argument('-mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument('-encoding', metavar='str', help='Data Encoding (e.g.latin1, utf-8)', type=str,
                        default='latin1')
    parser.add_argument('-celebrity', metavar='int', help='celebrity threshold', type=int, default=5)
    parser.add_argument('-vis', metavar='str', help='visualise representations', type=str, default=None)
    parser.add_argument('-builddata', action='store_true', help='if true do not recreated dumped data', default=False)

    # process_args: control the data preprocess
    parser.add_argument('-degree', type=int, help='degree of the approximation.', default=2)
    parser.add_argument('-normalization', type=str, help='Normalization method for the adjacency matrix.',
                        choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN', 'AugNormAdj', 'NormAdj',
                                 'RWalk', 'AugRWalk', 'NoNorm'], default='AugNormAdj')

    # model_args: the hyper-parameter of model
    parser.add_argument('-model', type=str, help='model to use.', default="HGNN")
    parser.add_argument('-usecuda', action='store_true', help='Use CUDA for training.', default=False)
    parser.add_argument('-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-epochs', type=int, help='Number of epochs to train.', default=30000)
    parser.add_argument('-lr', type=float, help='Initial learning rate.', default=0.001)
    parser.add_argument('-weight_decay', type=float, help='Weight decay (L2 loss on parameters).', default=5e-7)
    parser.add_argument('-patience', help='max iter for early stopping', type=int, default=30)
    parser.add_argument('-batch', metavar='int', help='SGD batch size', type=int, default=1024)

    args = parser.parse_args(argv)
    return args


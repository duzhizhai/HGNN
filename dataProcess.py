# -*- coding: UTF-8 -*-
import math
import os
import re
import csv
import gzip
import pickle
import hickle
import kdtree
import torch
import scipy.sparse as spsp
import numpy as np
import pandas as pd
import networkx as nx

from haversine import haversine
from scipy._lib.six import xrange
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict, OrderedDict
from sklearn.neighbors import NearestNeighbors
from normalization import fetch_normalization

import matplotlib.pyplot as plt

from utils import sparse_mx_to_torch_sparse_tensor

import smart_open
import gensim


def dump_obj(obj, filename, protocol=-1, serializer=pickle):
    if serializer == hickle:
        serializer.dump(obj, filename, mode='w', compression='gzip')
    else:
        with gzip.open(filename, 'wb') as fout:
            serializer.dump(obj, fout, protocol)


def load_obj(filename, serializer=pickle):
    if serializer == hickle:
        obj = serializer.load(filename)
    else:
        with gzip.open(filename, 'rb') as fin:
            obj = serializer.load(fin)
    return obj


def efficient_collaboration_weighted_projected_graph2(B, nodes):
    # B:        the whole graph including known nodes and mentioned nodes   --large graph
    # nodes:    the node_id of known nodes                                  --small graph node
    nodes = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    all_nodes = set(B.nodes())
    for m in all_nodes:
        nbrs = B[m]
        target_nbrs = [t for t in nbrs if t in nodes]
        # add edge between known nodesA(m) and known nodesB(n)
        if m in nodes:
            for n in target_nbrs:
                if m < n:
                    if not G.has_edge(m, n):
                        # Morton added for exclude the long edges

                        G.add_edge(m, n)
        # add edge between known n1 and known n2,
        # just because n1 and n2 have relation to m
        for n1 in target_nbrs:
            for n2 in target_nbrs:
                if n1 < n2:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
    return G


# normalization according to row, each row represent a feature
def feature_normalization1(dt):
    mean_num = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mean_num) / sigma


# normalization according to row, each row represent a feature
def feature_normalization2(dt):
    mean_num = np.mean(dt, axis=0)
    max_num = np.max(dt, axis=0)
    min_num = np.min(dt, axis=0)
    return (dt - mean_num) / (max_num - min_num)


def preprocess_data(data_args):
    """ obtain the parameters """
    data_dir = data_args.dir
    dump_file = data_args.dump_file
    bucket_size = data_args.bucket
    encoding = data_args.encoding
    celebrity_threshold = data_args.celebrity
    mindf = data_args.mindf
    builddata = data_args.builddata
    doc2vec_model_file = data_args.doc2vec_model_file
    # vocab_file = os.path.join(data_dir, 'vocab.pkl')
    if os.path.exists(dump_file):
        if not builddata:
            print('loading data from file : {}'.format(dump_file))
            data = load_obj(dump_file)
            return data
    dl = DataLoader(data_home=data_dir, bucket_size=bucket_size, encoding=encoding,
                    celebrity_threshold=celebrity_threshold, mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()  # 'user'        df_train          df_dev          df_test
    dl.assignClasses()  # 'lat', 'lon'  train_classes     dev_classes     test_class
    # dl.tfidf()                # 'text'        X_train           X_dev           X_test        self.tf_idf_sum
    dl.doc2vec_feature(doc2vec_model_file)  # 'text'        X_train           X_dev           X_test

    # dl.encodingContent(vacab_size=80000, encod_size=512, padding=0)     # 'text'   X_train   X_dev      X_test
    # vocab = dl.vectorizer.vocabulary_
    # dump_obj(vocab, vocab_file)
    # print('successfully dump vocab in {}'.format(vocab_file))
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()

    dl.get_graph()

    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    Y_test = dl.test_classes

    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]

    classLatMedian = {str(c): dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c): dl.cluster_median[c][1] for c in dl.cluster_median}

    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]

    adj = nx.adjacency_matrix(dl.graph)
    print('adjacency matrix created.')

    '''get training node index of each set.'''
    cluster_number = len(set(Y_train))
    cluster_nodes = []
    for set_idx in range(0, cluster_number):
        cluster_nodes.append(np.where(Y_train == set_idx)[0])
    cluster_arr = np.zeros(shape=(cluster_number, Y_train.shape[0]))
    for c_i, nodes in enumerate(cluster_nodes):
        for j in nodes:
            cluster_arr[c_i][j] = 1

    '''build the cluster graph Adjacency Matrix.'''
    cluster_adj = np.zeros((cluster_number, cluster_number))
    for i in range(0, cluster_number):
        for j in range(i + 1, cluster_number):
            clui_loc = (classLatMedian[str(i)], classLonMedian[str(i)])
            cluj_loc = (classLatMedian[str(j)], classLonMedian[str(j)])
            dis = haversine(clui_loc, cluj_loc)
            cluster_adj[i][j] = dis
            cluster_adj[j][i] = dis
    cluster_adj = (cluster_adj.max(axis=0) - cluster_adj) / cluster_adj.max(axis=0)

    '''store the shortest path matrix of each pair of nodes.'''

    # def Floyd(d):
    #     n = d.shape[0]
    #     for k in range(n):
    #         for i in range(n):
    #             for j in range(n):
    #                 d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    #         if (k + 1) % 100 == 0:
    #             print(k + 1, "done.")
    #
    #     return d
    # shorted_matrix = Floyd(adj.A)

    '''calculate the shortest hop path of every pair of nodes on the mention graph.'''
    num_of_nodes = dl.graph.number_of_nodes()
    shorted_matrix = np.zeros((num_of_nodes, num_of_nodes))
    for node_i in range(0, num_of_nodes):
        for node_j in range(node_i + 1, num_of_nodes):
            try:
                path_len = nx.dijkstra_path_length(dl.graph, source=node_i, target=node_j)
            except Exception as e:
                shorted_matrix[node_i][node_j] = 100
                continue
            shorted_matrix[node_i][node_j] = path_len
        if (node_i + 1) % 10 == 0:
            print(node_i + 1, "shortest path done.")

    '''calculate the shortest hop path on the mention graph.'''
    num_of_nodes = dl.graph.number_of_nodes()
    node2cluster_arr = np.zeros((num_of_nodes, cluster_number))  # size=(num_of_nodes, cluster_number)
    for node_i in range(0, num_of_nodes):
        for cluster_j in range(0, cluster_number):
            if node_i in cluster_nodes[cluster_j]:
                '''node_i belong the cluster_j (only exist for training nodes)'''
                shortest_path = 0
            else:
                nodes = cluster_nodes[cluster_j]
                path_list = []
                for node_j in nodes:
                    path_len = shorted_matrix[min(node_i, node_j)][max(node_i, node_j)]
                    if path_len == 100:
                        continue
                    path_list.append(path_len)
                if len(path_list) == 0:
                    '''for the isolate nodes.'''
                    shortest_path = 10
                else:
                    shortest_path = np.mean(path_list)

            node2cluster_arr[node_i][cluster_j] = shortest_path + 1

        if (node_i + 1) % 100 == 0:
            print(node_i + 1, "have gotten the shortest path.")

    '''save into files in order to repeat calculate.'''
    data = (adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test,
            classLatMedian, classLonMedian, userLocation, cluster_arr, cluster_adj, node2cluster_arr)
    dump_obj(data, dump_file)
    print('successfully dump data in {}'.format(str(dump_file)))
    return data


def process_data(data, args):
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, \
    classLatMedian, classLonMedian, userLocation, cluster_nodes, cluster_adj, node2cluster_arr = data

    '''porting to pyTorch and concat the matrix'''
    normalization = args.normalization
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    # '''word embedding for features'''
    # X_train = sparse_mx_to_torch_sparse_tensor(X_train)  # torch.FloatTensor(X_train.data)
    # X_dev = sparse_mx_to_torch_sparse_tensor(X_dev)
    # X_test = sparse_mx_to_torch_sparse_tensor(X_test)
    # features = torch.cat((X_train, X_dev, X_test), 0)
    # features = torch.FloatTensor(features.to_dense())

    # features = np.vstack((X_train.toarray(), X_dev.toarray(), X_test.toarray()))
    features = np.vstack((X_train, X_dev, X_test))  # only for dec2vec_feature

    '''feature normalization'''
    if args.feature_norm == 'Standard':
        features = feature_normalization1(features)
        print("Standard: using feature_normalization1 ..")
    elif args.feature_norm == 'Mean':
        features = feature_normalization2(features)
        print("Mean: using feature_normalization2 ..")
    else:
        print("none feature normalization.")

    features = torch.FloatTensor(features)
    cluster_nodes = torch.FloatTensor(cluster_nodes)
    cluster_adj = torch.FloatTensor(cluster_adj)
    node2cluster_arr = torch.FloatTensor(node2cluster_arr)
    print("feature shape:{}".format(features.shape))

    '''get labels'''
    labels = torch.LongTensor(np.hstack((Y_train, Y_dev, Y_test)))

    '''get index of train val and test'''
    len_train = int(X_train.shape[0])
    len_val = int(X_dev.shape[0])
    len_test = int(X_test.shape[0])
    idx_train = torch.LongTensor(range(len_train))
    idx_val = torch.LongTensor(range(len_train, len_train + len_val))
    idx_test = torch.LongTensor(range(len_train + len_val, len_train + len_val + len_test))

    '''convert to cuda'''
    if args.usecuda:
        print("converting data to CUDA format...")
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    data = (adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test,
            classLatMedian, classLonMedian, userLocation, cluster_nodes, cluster_adj, node2cluster_arr)
    return data


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def similarity_weight_of_graph(whole_graph, doc2vec_model_file):
    # load model
    model = gensim.models.doc2vec.Doc2Vec.load(doc2vec_model_file)
    # load feature process file.
    # all_corpus_file = "./data/cmu/train_corpus/cmu_all_process.txt"
    # all_corpus = list(read_corpus(all_corpus_file))

    # start to modify the weight of edges.
    sim_list = list()
    edge_list = whole_graph.edges
    for i, item in enumerate(edge_list):
        # wm_dis = model.wv.wmdistance(all_corpus[item[0]].words, all_corpus[item[1]].words)
        sim = model.docvecs.similarity(item[0], item[1])
        if sim <= 0:
            whole_graph.add_edge(item[0], item[1], weight=abs(sim) + 1)
        else:
            whole_graph.add_edge(item[0], item[1], weight=sim * 10)
        sim_list.append(sim)
        if (i + 1) % 10000 == 0:
            print("{} is finished!".format(i + 1))

    print("min:{} \t max:{} \t mean:{}".format(np.min(sim_list), np.max(sim_list), np.mean(sim_list)))

    return whole_graph


def route_weight_of_graph(whole_graph):
    ret_graph = whole_graph
    edge_list = whole_graph.edges
    count = 0
    for item in edge_list:
        whole_graph.remove_edge(item[0], item[1])
        try:
            route_path = nx.shortest_path_length(G=whole_graph, source=item[0], target=item[1])
        except nx.NetworkXNoPath:
            route_path = 0
            count += 1
            print("No path.{}".format(count))
        whole_graph.add_edge(item[0], item[1])
        if route_path == 0:
            ret_graph.add_edge(item[0], item[1], weight=1)
        else:
            ret_graph.add_edge(item[0], item[1], weight=(1 / route_path) + 1)

    return ret_graph


def guess_virtual_nodes(whole_graph, userLocation, U_train, U_dev, U_test):
    index_to_userId = list(U_train + U_dev + U_test)
    edge_list_all = list(whole_graph.edges)
    edge_list = list()
    guess_nodes = list()  # {<(id_1,id1),(id_1,id2)>,<(id_2,id1),(id_2,id2),...>,...}
    node_temp = list()

    # remain the edges between train and val/test data.
    index_threshold = len(U_train)
    for item in edge_list_all:
        if (item[0] < index_threshold) and (item[1] >= index_threshold):
            edge_list.append(item[::-1])

    # sort by item[0]
    edge_list.sort(key=lambda x: x[0])

    A_index = edge_list[0][0]
    for item in edge_list:
        if item[0] == A_index:
            node_temp.append(item)
        else:
            guess_nodes.append(node_temp)
            node_temp = []
            node_temp.append(item)
            A_index = item[0]
    guess_nodes.append(node_temp)  # add the last node.
    print("guess_nodes data format:{<(id_1,id1),(id_1,id2)>,<(id_2,id1),(id_2,id2),...>,...}")

    # guess a virtual position of nodes.
    neighbour_lati = list()
    neighbour_long = list()
    for node in guess_nodes:
        for item in node:
            user_ID = index_to_userId[item[1]]  # get user_ID
            user_loca = userLocation[user_ID].split(',')  # get user_location
            neighbour_lati.append(float(user_loca[0]))
            neighbour_long.append(float(user_loca[1]))
        # updata the target node location information
        user_ID = index_to_userId[item[0]]
        userLocation[user_ID] = str(np.mean(neighbour_lati)) + ',' + str(np.mean(neighbour_long))
        neighbour_lati.clear()
        neighbour_long.clear()
    print("len(guess_nodes) is:{}".format(len(guess_nodes)))
    return userLocation


def add_lineEXP_only_for_train_data(whole_graph, userLocation, U_train, dis_mean):
    index_to_userId = list(U_train)
    edge_list_old = list(whole_graph.edges)

    '''only select train nodes and reproduce new edge_list'''
    edge_list = list()
    train_len = len(index_to_userId)
    for item in edge_list_old:
        if (item[0] < train_len) and (item[1] < train_len):
            edge_list.append(item)
        else:
            continue
    print("len of edge_list is:{}".format(len(edge_list)))

    for item in edge_list:
        # get the index of nodes in the whole graph
        A_index = item[0]
        B_index = item[1]
        # get user_id
        use_id_A = index_to_userId[A_index]
        use_id_B = index_to_userId[B_index]
        # get the location of user using user_id
        locationA = userLocation[use_id_A].split(',')
        locationB = userLocation[use_id_B].split(',')
        # calculate the distance
        distance = haversine((float(locationA[0]), float(locationA[1])),
                             (float(locationB[0]), float(locationB[1])))
        if distance <= dis_mean:
            weight = (dis_mean + 1 - distance) / 10
        else:
            weight = math.exp(dis_mean - distance)
        whole_graph.add_edge(A_index, B_index, weight=weight)
    return whole_graph


def add_lineEXP_value_of_edges(whole_graph, userLocation, U_train, U_dev, U_test, dis_mean):
    index_to_userId = list(U_train + U_dev + U_test)
    edge_list = list(whole_graph.edges)
    for item in edge_list:
        # get the index of nodes in the whole graph
        A_index = item[0]
        B_index = item[1]
        # get user_id
        use_id_A = index_to_userId[A_index]
        use_id_B = index_to_userId[B_index]
        # get the location of user using user_id
        locationA = userLocation[use_id_A].split(',')
        locationB = userLocation[use_id_B].split(',')
        # calculate the distance
        distance = haversine((float(locationA[0]), float(locationA[1])),
                             (float(locationB[0]), float(locationB[1])))
        if distance <= dis_mean:
            weight = dis_mean + 1 - distance
        else:
            weight = math.exp(dis_mean - distance)
        whole_graph.add_edge(A_index, B_index, weight=weight)
    return whole_graph


def add_RBF_value_of_edges(whole_graph, userLocation, U_train, U_dev, U_test, dis_mean, dis_var):
    index_to_userId = list(U_train + U_dev + U_test)
    edge_list = list(whole_graph.edges)
    for item in edge_list:
        # get the index of nodes in the whole graph
        A_index = item[0]
        B_index = item[1]
        # get user_id
        use_id_A = index_to_userId[A_index]
        use_id_B = index_to_userId[B_index]
        # get the location of user using user_id
        locationA = userLocation[use_id_A].split(',')
        locationB = userLocation[use_id_B].split(',')
        # calculate the distance
        distance = haversine((float(locationA[0]), float(locationA[1])),
                             (float(locationB[0]), float(locationB[1])))
        weight = (distance - dis_mean) / dis_var
        whole_graph.add_edge(A_index, B_index, weight=weight)
    return whole_graph


def del_long_edge_from_graph(whole_graph, userLocation, U_train, U_dev, U_test, threshold=2800):
    index_to_userId = list(U_train + U_dev + U_test)
    edge_list = list(whole_graph.edges)
    remove_count = 0
    for item in edge_list:
        # get the index of nodes in the whole graph
        A_index = item[0]
        B_index = item[1]
        # get user_id
        use_id_A = index_to_userId[A_index]
        use_id_B = index_to_userId[B_index]
        # get the location of user using user_id
        locationA = userLocation[use_id_A].split(',')
        locationB = userLocation[use_id_B].split(',')
        # calculate the distance
        distance = haversine((float(locationA[0]), float(locationA[1])),
                             (float(locationB[0]), float(locationB[1])))
        if distance >= threshold:
            whole_graph.remove_edge(A_index, B_index)
            remove_count += 1
    print("remove {} edges with threshold {}".format(remove_count, threshold))

    return whole_graph


def del_nodes_from_dataset(data_input, del_index_list):
    if type(data_input).__name__ == 'csr_matrix':
        # print("origin shape of csr_matrix:{}".format(data_input.shape))
        data_input = data_input.toarray()
        for i in range(len(del_index_list) - 1, -1, -1):
            data_input = np.delete(data_input, del_index_list[i], 0)
        data_input = spsp.csr_matrix(data_input)
        # print("now shape:{}\tdel_list len:{}".format((data_input.shape), len(del_index_list)))

    elif type(data_input).__name__ == 'ndarray':
        # print("origin shape of ndarray:{}".format(data_input.shape))
        data_input = data_input.tolist()
        for i in range(len(del_index_list) - 1, -1, -1):
            del data_input[del_index_list[i]]
        data_input = np.array(data_input)
        # print("now shape:{}\tdel_list len:{}".format((data_input.shape), len(del_index_list)))

    else:
        # print("origin len:{}".format(len(data_input)))
        for i in range(len(del_index_list) - 1, -1, -1):
            del data_input[del_index_list[i]]
        # print("now len:{}\tdel_list len:{}".format(len(data_input), len(del_index_list)))

    return data_input


def del_isolated_nodes_from_graph(whole_graph):
    isolated_nodes = list()
    for item in whole_graph.degree:
        if item[1] == 0:
            isolated_nodes.append(item[0])
    whole_graph.remove_nodes_from(isolated_nodes)
    print("remove {} isolated_nodes".format(len(isolated_nodes)))
    return whole_graph, isolated_nodes


def count_distance_of_every_two_joint_nodes(whole_graph, userLocation, U_train, U_dev, U_test, edge_dis_file):
    print('Nodes: %d, Edges: %d' % (nx.number_of_nodes(whole_graph), nx.number_of_edges(whole_graph)))
    index_to_userId = list(U_train + U_dev + U_test)
    edge_list = list(whole_graph.edges)
    dis_list = list()  # store the edge distance of the whole graph
    # dis_data = ([0, 3], 5)        # distance between user_id:0 and user_id:3 is 5,  5 = dis<0, 3>
    A_index = 0
    for item in edge_list:
        # add the nodes which has no edge
        if item[0] - A_index > 1:
            no_edge_index = range(A_index + 1, item[0])
            for temp in no_edge_index:
                use_id = index_to_userId[temp]
                dis_data = ([use_id, use_id], 0)
                dis_list.append(dis_data)

        # get the index of nodes in the whole graph
        A_index = item[0]
        B_index = item[1]
        # get user_id
        use_id_A = index_to_userId[A_index]
        use_id_B = index_to_userId[B_index]
        # get the location of user using user_id
        locationA = userLocation[use_id_A].split(',')
        locationB = userLocation[use_id_B].split(',')
        # calculate the distance
        distance = haversine((float(locationA[0]), float(locationA[1])),
                             (float(locationB[0]), float(locationB[1])))
        dis_data = ([use_id_A, use_id_B], distance)
        dis_list.append(dis_data)

    # add extra edge=0 for some large_index nodes which has no edge
    largest_index = edge_list[-1][0]
    node_num = len(whole_graph.nodes)
    if largest_index < node_num:
        no_edge_index = range(largest_index + 1, node_num)
        for temp in no_edge_index:
            use_id = index_to_userId[temp]
            dis_data = ([use_id, use_id], 0)
            dis_list.append(dis_data)

    # store the dump file
    dump_obj(dis_list, edge_dis_file)
    print("dis_list data format: [(<use_id_A, use_id_B>, dis), (<use_id_A, use_id_C>, dis), ...]")
    print("Done.")


def count_distance_for_train_data(whole_graph, userLocation, U_train, edge_dis_file):
    print('Nodes: %d, Edges: %d' % (nx.number_of_nodes(whole_graph), nx.number_of_edges(whole_graph)))
    index_to_userId = list(U_train)  # count the edges' information only for train userID
    edge_list_old = list(whole_graph.edges)

    '''only select train nodes and reproduce new edge_list'''
    edge_list = list()
    train_len = len(index_to_userId)
    for item in edge_list_old:
        if (item[0] < train_len) and (item[1] < train_len):
            edge_list.append(item)
        else:
            continue
    print("len of edge_list is:{}".format(len(edge_list)))

    dis_list = list()  # store the edge distance of the whole graph
    A_index = 0
    for item in edge_list:
        # add the nodes which has no edge
        if item[0] - A_index > 1:
            no_edge_index = range(A_index + 1, item[0])
            for temp in no_edge_index:
                use_id = index_to_userId[temp]
                dis_data = ([use_id, use_id], 0)
                dis_list.append(dis_data)

        # get the index of nodes in the whole graph
        A_index = item[0]
        B_index = item[1]
        # get user_id
        use_id_A = index_to_userId[A_index]
        use_id_B = index_to_userId[B_index]
        # get the location of user using user_id
        locationA = userLocation[use_id_A].split(',')
        locationB = userLocation[use_id_B].split(',')
        # calculate the distance
        distance = haversine((float(locationA[0]), float(locationA[1])),
                             (float(locationB[0]), float(locationB[1])))
        dis_data = ([use_id_A, use_id_B], distance)
        dis_list.append(dis_data)

    # add extra edge=0 for some large_index nodes which has no edge
    largest_index = edge_list[-1][0]
    node_num = len(index_to_userId)
    if largest_index < node_num:
        no_edge_index = range(largest_index + 1, node_num)
        for temp in no_edge_index:
            use_id = index_to_userId[temp]
            dis_data = ([use_id, use_id], 0)
            dis_list.append(dis_data)

    # store the dump file
    dump_obj(dis_list, edge_dis_file)
    print("dis_list data format: [(<use_id_A, use_id_B>, dis), (<use_id_A, use_id_C>, dis), ...]")
    print("Done.")


def analyze_distance(edge_dis_file='./my_assets/no_ues_now/edge_dis.pkl'):
    # load dis_list from file
    if os.path.exists(edge_dis_file):
        print('loading dis_list from {}'.format(edge_dis_file))
        dis_list = load_obj(edge_dis_file)
    else:
        exit("there is no {}".format(edge_dis_file))
    print("len(dis_list):{}".format(len(dis_list)))
    # gather all distance of every user_id
    user_dis = list()  # data format: [(user_id_A, disx, disy, disz), (user_id_B, 0), ...]
    one_user = list()  # data format: [user_id_A, disx, disy, disz]
    user_id_A = dis_list[0][0][0]

    for index, item in enumerate(dis_list):
        # judge whether is the same user with the last one.
        if item[0][0] != user_id_A:
            user_dis.append(one_user)
            one_user = list()
            user_id_A = item[0][0]

        # store only a user_id
        if len(one_user) == 0:
            one_user.append(item[0][0])  # add user_id
        one_user.append(item[1])  # add dis

        # store the last user_id information of dis_list
        if index == len(dis_list) - 1:
            user_dis.append(one_user)
    print("len(user_dis):{}".format(len(user_dis)))

    # count the average edge distance of every user_id
    ave_list = list()
    all_dis_list = list()  # in order to get the mean and variance of all edges.
    for item in user_dis:
        ave_list.append(np.mean(item[1:]))
        for edge_dis_temp in item[1:]:  # in order to get the mean and variance of all edges.
            all_dis_list.append(edge_dis_temp)  # in order to get the mean and variance of all edges.
    print("the average distance of all edges:{}".format(np.mean(ave_list)))
    print("len of all_dis_list:{}\t mean:{}\t variance:{}"
          .format(len(all_dis_list), np.mean(all_dis_list), np.var(all_dis_list)))

    # count the distance distribution of user's average dis
    dis_distribution = [0] * 7  # distribution list
    for item in ave_list:
        index = int(math.ceil(item / 1000))  # math.ceil(2.3)-->3
        if index > 6:
            dis_distribution[6] += 1
        else:
            dis_distribution[index] += 1

    # count the distance distribution of the whole edges
    dis_distri_all = [0] * 7  # distribution list	gap 1000
    for item in dis_list:
        index = int(math.ceil(item[1] / 1000))  # math.ceil(2.3)-->3
        if index > 6:
            dis_distri_all[6] += 1
        else:
            dis_distri_all[index] += 1

    # count the distance distribution of the whole edges
    dis_distri_all2 = [0] * 11  # distribution list gap 500
    for item in dis_list:
        index = int(math.ceil(item[1] / 500))  # math.ceil(2.3)-->3
        if index > 10:
            dis_distri_all2[10] += 1
        else:
            dis_distri_all2[index] += 1

    # draw the picture
    x_data_1 = range(0, len(ave_list))
    y_data_1 = ave_list
    x_data_2 = range(0, 7)
    y_data_2 = dis_distribution
    x_data_3 = range(0, 7)
    y_data_3 = dis_distri_all
    x_data_4 = range(0, 11)
    y_data_4 = dis_distri_all2
    name_list = ['0', '(0,1K]', '(1K,2K]', '(2K,3K]', '(3K,4K]', '(4K,5K]', '(5K,++)']
    name_list2 = ['0', '(0,0.5]', '(0.5,1]', '(1,1.5]', '(1.5,2]', '(2,2.5]', '(2.5,3]',
                  '(3,3.5]', '(3.5,4]', '(4,4.5]', '(4.5,++)']

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x_data_1, y_data_1, label='average distance', color='b', linestyle='solid', linewidth=1)
    plt.legend()
    plt.xlabel('nodes_id')
    plt.ylabel('aver_dis')
    plt.title("the average distance of every nodes' edges.")

    plt.subplot(2, 2, 2)
    plt.bar(x_data_2, y_data_2, label='dis_distribution', width=0.35, fc='r', tick_label=name_list)
    plt.legend()
    plt.xlabel('dis_distribution')
    plt.ylabel('number')
    plt.title("the distance distribution of nodes' average edges.")

    plt.subplot(2, 2, 3)
    plt.bar(x_data_4, y_data_4, label='dis_distri_all', width=0.35, fc='y', tick_label=name_list2)
    plt.legend()
    plt.xlabel('dis_distribution(Unit:K)')
    plt.ylabel('number')
    plt.title("the distance distribution of the whole edges in the graph(gap 500).")

    plt.subplot(2, 2, 4)
    plt.bar(x_data_3, y_data_3, label='dis_distri_all', width=0.35, fc='y', tick_label=name_list)
    plt.legend()
    plt.xlabel('dis_distribution')
    plt.ylabel('number')
    plt.title("the distance distribution of the whole edges in the graph(gap 1000).")

    plt.show()

    return np.mean(all_dis_list), np.var(all_dis_list)


class DataLoader:
    def __init__(self, data_home, bucket_size=50, encoding='utf-8', celebrity_threshold=10, one_hot_labels=False,
                 mindf=10, maxdf=0.2, norm='l2', idf=True, btf=True, tokenizer=None, subtf=False, stops=None,
                 token_pattern=r'(?u)(?<![#@])\b\w\w+\b', vocab=None):
        self.data_home = data_home
        self.bucket_size = bucket_size
        self.encoding = encoding
        self.celebrity_threshold = celebrity_threshold
        self.one_hot_labels = one_hot_labels
        self.mindf = mindf
        self.maxdf = maxdf
        self.norm = norm
        self.idf = idf
        self.btf = btf
        self.tokenizer = tokenizer
        self.subtf = subtf
        self.stops = stops if stops else 'english'
        self.token_pattern = r'(?u)(?<![#@|,.-_+^……$%&*(); :`，。？、：；;《》{}“”~#￥])\b\w\w+\b'
        self.vocab = vocab
        # self.biggraph = None

    def load_data(self):
        print('loading the dataset from: {}'.format(self.data_home))
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')

        df_train = pd.read_csv(train_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                               quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                             quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_test = pd.read_csv(test_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                              quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_train.dropna(inplace=True)
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)

        df_train['user'] = df_train['user'].apply(lambda x: str(x).lower())
        df_train.drop_duplicates(['user'], inplace=True, keep='last')
        df_train.set_index(['user'], drop=True, append=False, inplace=True)
        df_train.sort_index(inplace=True)

        df_dev['user'] = df_dev['user'].apply(lambda x: str(x).lower())
        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)

        df_test['user'] = df_test['user'].apply(lambda x: str(x).lower())
        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)

        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test

    def get_graph(self):
        g = nx.Graph()
        # 'user'
        nodes = set(self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + len(self.df_test), 'duplicate target node'
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        print('start adding the train graph')
        externalNum = 0
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_train.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(user_id, id)
        print('start adding the dev graph')
        externalNum = 0
        for i in range(len(self.df_dev)):
            user = self.df_dev.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_dev.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        print('start adding the test graph')
        externalNum = 0
        for i in range(len(self.df_test)):
            user = self.df_test.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_test.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
                    externalNum += 1
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        print('#nodes: %d, #edges: %d' % (nx.number_of_nodes(g), nx.number_of_edges(g)))

        celebrities = []
        for i in xrange(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)

        print('removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)
        print('projecting the graph')
        projected_g = efficient_collaboration_weighted_projected_graph2(g, range(len(nodes_list)))
        print('#nodes: %d, #edges: %d' % (nx.number_of_nodes(projected_g), nx.number_of_edges(projected_g)))
        self.graph = projected_g

    def get_graph_temp(self):
        g = nx.Graph()
        nodes = set(self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + len(self.df_test), 'duplicate target node'
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist()
        node_id = {node: id for id, node in enumerate(nodes_list)}
        g.add_nodes_from(node_id.values())
        train_locs = self.df_train[['lat', 'lon']].values
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        print('adding the train graph')
        for i in range(len(self.df_train)):
            user = self.df_train.index[i]
            user_id = node_id[user]
            mentions = [m for m in pattern.findall(self.df_train.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        celebrities = []
        for i in xrange(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg > self.celebrity_threshold:
                celebrities.append(i)
        # get neighbours of celebrities
        id_node = {v: k for k, v in node_id.iteritems()}

        degree_distmean = defaultdict(list)
        degree_distance = defaultdict(list)
        c_distmean = {}
        for c in celebrities:
            c_name = id_node[c]
            c_nbrs = g[c].keys()
            c_degree = len(c_nbrs)
            c_locs = train_locs[c_nbrs, :]
            c_lats = c_locs[:, 0]
            c_lons = c_locs[:, 1]
            c_median_lat = np.median(c_lats)
            c_median_lon = np.median(c_lons)
            distances = [haversine((c_median_lat, c_median_lon), tuple(c_locs[i].tolist())) for i in
                         range(c_locs.shape[0])]
            degree_distance[c_degree].extend(distances)
            c_meandist = np.mean(distances)
            degree_distmean[c_degree].append(c_meandist)
            c_distmean[c_name] = [c_degree, c_meandist]
        with open('celebrity.pkl', 'wb') as fin:
            pickle.dump((c_distmean, degree_distmean, degree_distance), fin)

        print('removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        self.biggraph = g

    def longest_path(self, g):
        nodes = g.nodes()
        pathlen_counter = Counter()
        for n1 in nodes:
            for n2 in nodes:
                if n1 < n2:
                    for path in nx.all_simple_paths(g, source=n1, target=n2):
                        pathlen = len(path)
                        pathlen_counter[pathlen] += 1
        return pathlen_counter

    def list_dats_save_to_txt(self, file_path, data):  # file_path 为写入文件的路径，data为要写入数据列表.
        file = open(file_path, 'a', encoding='utf-8')
        for i in range(len(data)):
            file.write(str(data[i]) + '\n')
        file.close()
        print("{} save success !".format(file_path))

        # .replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        # 	s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符

    def doc2vec_feature(self, doc2vec_model_file):
        # load model
        model = gensim.models.doc2vec.Doc2Vec.load(doc2vec_model_file)

        # train data features
        feature_list = list()
        index_l = 0
        index_r = len(self.df_train.text)
        for i in range(index_l, index_r):
            feature_list.append(model.docvecs[i])
        self.X_train = np.array(feature_list)

        # dev data features
        feature_list = list()
        index_l = len(self.df_train.text)
        index_r = len(self.df_train.text) + len(self.df_dev.text)
        for i in range(index_l, index_r):
            feature_list.append(model.docvecs[i])
        self.X_dev = np.array(feature_list)

        # test data features
        feature_list = list()
        index_l = len(self.df_train.text) + len(self.df_dev.text)
        index_r = len(self.df_train.text) + len(self.df_dev.text) + len(self.df_test.text)
        for i in range(index_l, index_r):
            feature_list.append(model.docvecs[i])
        self.X_test = np.array(feature_list)

        print("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        print("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        print("test        n_samples: %d, n_features: %d" % self.X_test.shape)

    def tfidf(self):
        # keep both hashtags and mentions
        # token_pattern=r'(?u)@?#?\b\w\w+\b'
        # remove hashtags and mentions
        # token_pattern = r'(?u)(?<![#@])\b\w+\b'
        # just remove mentions and remove hashsign from hashtags
        # token_pattern = r'(?u)(?<![@])\b\w+\b'
        # remove mentions but keep hashtags with their sign
        # token_pattern = r'(?u)(?<![@])#?\b\w\w+\b'
        # remove multple occurrences of a character after 2 times yesss => yess
        # re.sub(r"(.)\1+", r"\1\1", s)

        """
        # Morton add for save the words data into the file.
        self.list_dats_save_to_txt("./data/cmu/train_corpus/cmu_train.txt", list(self.df_train.text.values))
        self.list_dats_save_to_txt("./data/cmu/train_corpus/cmu_dev.txt",  list(self.df_dev.text.values))
        self.list_dats_save_to_txt("./data/cmu/train_corpus/cmu_test.txt",  list(self.df_test.text.values))
        """

        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern, use_idf=self.idf,
                                          norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                          min_df=self.mindf, max_df=self.maxdf, ngram_range=(1, 1),
                                          stop_words=self.stops,
                                          vocabulary=self.vocab, encoding=self.encoding, dtype=np.float32)
        self.X_train = self.vectorizer.fit_transform(self.df_train.text.values)
        self.X_dev = self.vectorizer.transform(self.df_dev.text.values)
        self.X_test = self.vectorizer.transform(self.df_test.text.values)
        print("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        print("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        print("test        n_samples: %d, n_features: %d" % self.X_test.shape)

        ''' drop out, never used. in order to calculate the  sum(tfidf) * context_features '''
        # X_train = self.vectorizer.fit_transform(self.df_train.text.values)
        # X_dev = self.vectorizer.transform(self.df_dev.text.values)
        # X_test = self.vectorizer.transform(self.df_test.text.values)
        # tf_idf_sum = np.vstack((X_train.toarray(), X_dev.toarray(), X_test.toarray()))
        # self.tf_idf_sum = np.sum(tf_idf_sum, axis=1)
        # print("tf_idf_sum done.")

    def assignClasses(self):
        clusterer = kdtree.KDTreeClustering(bucket_size=self.bucket_size)
        train_locs = self.df_train[['lat', 'lon']].values
        clusterer.fit(train_locs)
        clusters = clusterer.get_clusters()
        cluster_points = defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_points[cluster].append(train_locs[i])
        print('# the number of clusterer labels is: %d' % len(cluster_points))
        self.cluster_median = OrderedDict()
        for cluster in sorted(cluster_points):
            points = cluster_points[cluster]
            median_lat = np.median([p[0] for p in points])
            median_lon = np.median([p[1] for p in points])
            self.cluster_median[cluster] = (median_lat, median_lon)
        dev_locs = self.df_dev[['lat', 'lon']].values
        test_locs = self.df_test[['lat', 'lon']].values
        nnbr = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size=1, metric=haversine, n_jobs=4)
        nnbr.fit(np.array(list(self.cluster_median.values())))
        self.dev_classes = nnbr.kneighbors(dev_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.test_classes = nnbr.kneighbors(test_locs, n_neighbors=1, return_distance=False)[:, 0]

        self.train_classes = clusters

        if self.one_hot_labels:
            num_labels = np.max(self.train_classes) + 1
            y_train = np.zeros((len(self.train_classes), num_labels), dtype=np.float32)
            y_train[np.arange(len(self.train_classes)), self.train_classes] = 1
            y_dev = np.zeros((len(self.dev_classes), num_labels), dtype=np.float32)
            y_dev[np.arange(len(self.dev_classes)), self.dev_classes] = 1
            y_test = np.zeros((len(self.test_classes), num_labels), dtype=np.float32)
            y_test[np.arange(len(self.test_classes)), self.test_classes] = 1
            self.train_classes = y_train
            self.dev_classes = y_dev
            self.test_classes = y_test

    def encodingContent(self, vacab_size=80000, encod_size=500, padding=0):
        vocab = dict()
        total_words = list()
        ignore = [' ', '|||', 'RT']

        # get_vocab_and_total_words from content
        def get_vocab_and_total_words(all_content):
            for line in all_content:
                # get words list
                line = line.encode("utf-8")
                words = line.split()
                # remove the ignore words
                for ign in ignore:
                    condition = lambda t: t != ign
                    words = list(filter(condition, words))
                # save all words
                total_words.append(words)
                # count words and number in vocab dict
                for word in words:
                    if vocab.has_key(word):
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

        get_vocab_and_total_words(self.df_train.text.values)
        get_vocab_and_total_words(self.df_dev.text.values)
        get_vocab_and_total_words(self.df_test.text.values)

        # sort the vocab 'key' according to the 'value'
        vocab_list = sorted(vocab.items(), key=lambda d: d[1], reverse=True)
        vocab_list = vocab_list[0:vacab_size]
        vocab_new = list()
        for i in vocab_list:
            vocab_new.append(i[0])

        # show information of content
        words_count = 0
        for line in total_words:
            words_count += len(line)
        print("the number of total words:{}\nline number:{}\naverage of line:{}"
              .format(words_count, len(total_words), words_count / len(total_words)))

        # # normalization according to row, each row represent a feature
        # def feature_normalization1(dt):
        # 	mean_num = np.mean(dt, axis=0)
        # 	sigma = np.std(dt, axis=0)
        # 	return (dt-mean_num)/sigma
        #
        # # normalization according to row, each row represent a feature
        # def feature_normalization2(dt):
        # 	mean_num = np.mean(dt, axis=0)
        # 	max_num = np.max(dt, axis=0)
        # 	min_num = np.min(dt, axis=0)
        # 	return (dt-mean_num)/(max_num-min_num)

        # get_index_form_vocab, all_words:train, dev, test
        def get_index_form_vocab(all_words):
            print("start get index...")
            encoding_words = list()
            for line in all_words:
                line_list = [0] * encod_size
                # cut line down to encod_size or padding
                if len(line) >= encod_size:
                    line = line[0:encod_size]
                # get the true index
                for i, word in enumerate(line):
                    if word in vocab_new:
                        line_list[i] = vocab_new.index(word)
                encoding_words.append(line_list)
            return np.array(encoding_words, dtype=float)

        train_words = total_words[0:self.df_train.text.shape[0]]
        dev_words = total_words[self.df_train.text.shape[0]:self.df_train.text.shape[0] + self.df_dev.text.shape[0]]
        test_words = total_words[self.df_train.text.shape[0] + self.df_dev.text.shape[0]:]
        self.X_train = get_index_form_vocab(train_words)
        self.X_dev = get_index_form_vocab(dev_words)
        self.X_test = get_index_form_vocab(test_words)
        print("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        print("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        print("test        n_samples: %d, n_features: %d" % self.X_test.shape)


if __name__ == '__main__':
    dl = DataLoader(data_home='./data/cmu', bucket_size=50, encoding='latin1', celebrity_threshold=5, mindf=10,
                    token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()
    dl.get_graph()
    dl.assignClasses()  # create the label (129 for cmu dataset)

    # dl.tfidf()
    # dl.encodingContent()
    # dl.encodingContent(vacab_size=80000, encod_size=500, padding=0)
    # analyze_distance()

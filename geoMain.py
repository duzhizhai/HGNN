#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import torch
import time

import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from haversine import haversine
from models import get_model
from dataProcess import preprocess_data, process_data
from utils import sgc_precompute, parse_args


def train_regression(model, train_features, train_labels, val_features, U_dev, classLatMedian, classLonMedian,
                     userLocation, epochs=100, weight_decay=5e-6, lr=0.01, patience=10, model_file='myModel.pkl'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_patience = 0
    val_acc_best = -1
    epoch_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()

        '''verification, no gradient descent'''
        _, _, val_acc, _, _, _ = geo_eval(model, val_features, U_dev, classLatMedian, classLonMedian, userLocation)

        '''show val_acc,val_acc_best every 50 epoch'''
        if epoch % 50 == 0:
            print("epoch:{}\t \tval_acc:{}\t \tval_acc_best:{}".format(epoch, val_acc, val_acc_best))

        '''apply early stop using val_acc_best, and save model'''
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            epoch_best = epoch
            train_patience = 0
            torch.save(model.state_dict(), model_file)
        else:
            train_patience += 1
        if train_patience == patience:
            print("Early stop! \t epoch_best:{}\t \t \tval_acc_best:{}".format(epoch_best, val_acc_best))
            break
    return val_acc_best, epoch_best


def train_regression2(model, train_features, train_labels, val_features, U_dev, classLatMedian, classLonMedian,
                      userLocation, epochs=100, weight_decay=5e-6, lr=0.01, patience=10, model_file='myModel.pkl',
                      cluster_nodes=None, cluster_adj=None, node2cluster_arr=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_patience = 0
    val_acc_best = -1
    epoch_best = 0
    train_len = train_features.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(train_features, node2cluster_arr[0:train_len], cluster_nodes, cluster_adj)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()

        '''verification, no gradient descent'''
        _, _, val_acc, _, _, _ = geo_eval2(model, val_features, U_dev, classLatMedian, classLonMedian,
                                           userLocation, node2cluster_arr[train_len:train_len + val_features.shape[0]])

        '''show val_acc,val_acc_best every 50 epoch'''
        if epoch % 50 == 0:
            print("epoch:{}\t \tval_acc:{}\t \tval_acc_best:{}".format(epoch, val_acc, val_acc_best))

        '''apply early stop using val_acc_best, and save model'''
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            epoch_best = epoch
            train_patience = 0
            # torch.save(model.state_dict(), model_file)
            torch.save(model, model_file)
        else:
            train_patience += 1
        if train_patience == patience:
            print("Early stop! \t epoch_best:{}\t \t \tval_acc_best:{}".format(epoch_best, val_acc_best))
            break
    return val_acc_best, epoch_best


def geo_eval(model, features, U_test, classLatMedian, classLonMedian, userLocation):
    with torch.no_grad():
        model.eval()
        y_pred = model(features)
    y_pred = y_pred.data.cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)  # 1代表行

    assert len(y_pred) == len(U_test), "#preds: %d, #users: %d" % (len(y_pred), len(U_test))

    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred, y_pred[i]])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred


def geo_eval2(model, features, U_test, classLatMedian, classLonMedian, userLocation, node2cluster_arr):
    with torch.no_grad():
        model.eval()
        y_pred = model(features, node2cluster_arr)
    y_pred = y_pred.data.cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)  # 1代表行

    assert len(y_pred) == len(U_test), "#preds: %d, #users: %d" % (len(y_pred), len(U_test))

    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_test[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred, y_pred[i]])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))
    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred


def main():
    """
    preprocess_data() :     load data from dataset and precess data into numpy format
    process_data() :        port the data to pyTorch and convert to cuda
    U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation : only use when Valid and Test
    """
    data = preprocess_data(args)
    data = process_data(data, args)
    (adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test,
     classLatMedian, classLonMedian, userLocation, cluster_nodes, cluster_adj, node2cluster_arr) = data

    """
    get model and train
    input_dim: features.size(1)             # equal 9467 for ./data/cmu
    output_dim: labels.max().item() + 1     # equal 129 for ./data/cmu
    """
    model = get_model(args.model, features.shape[1], labels.max().item() + 1, usecuda=args.usecuda)
    model_file = "./result/model/{}_lr_{}.pkl".format(args.model, str(args.lr))

    if args.model == "SGC":
        if args.vis == "vis":
            from plotFunc import draw_representations
            for i in range(1, 5):
                draw_representations(features, labels, k=4, seed=77, do_pca=True,
                                     filename='./result/pic/dim_128_origin_{}.png'.format(i))

        # influence = torch.FloatTensor(influence).cuda()
        # features = torch.mm(influence, features)

        features = sgc_precompute(features, adj, args.degree)

        if args.vis == "vis":
            from plotFunc import draw_representations
            for i in range(1, 5):
                draw_representations(features, labels, k=4, seed=77, do_pca=True,
                                     filename='./result/pic/dim_128_sgc_{}.png'.format(i))
            exit(1)

        val_acc_best, epoch_best = train_regression(model, features[idx_train], labels[idx_train], features[idx_val],
                                                    U_dev, classLatMedian, classLonMedian, userLocation, args.epochs,
                                                    args.weight_decay, args.lr, args.patience, model_file)

    if args.model == "HGNN":
        val_acc_best, epoch_best = train_regression2(model, features[idx_train], labels[idx_train], features[idx_val],
                                                     U_dev, classLatMedian, classLonMedian, userLocation,
                                                     args.epochs, args.weight_decay, args.lr, args.patience,
                                                     model_file, cluster_nodes, cluster_adj, node2cluster_arr)

    # load model from file and test the model
    # my_model = get_model(args.model, features.shape[1], labels.max().item() + 1, usecuda=args.usecuda)
    # my_model.load_state_dict(torch.load(model_file))
    my_model = torch.load(model_file)
    csv_file_test = "./result/dim_{}_{}_{}_{}_test.csv".format(
        features.shape[1], args.feature_norm, args.weight_decay, args.lr)
    csv_file_train = "./result/dim_{}_{}_{}_{}_train.csv".format(
        features.shape[1], args.feature_norm, args.weight_decay, args.lr)
    if args.model == "HGNN":
        meanDis, MedianDis, accAT161, distances, latlon_true, latlon_pred = geo_eval2(my_model, features[idx_test],
                                                                                      U_test, classLatMedian,
                                                                                      classLonMedian, userLocation,
                                                                                      node2cluster_arr[idx_test])
        # save_coordinate_true_predict(distances, latlon_true, latlon_pred, labels[idx_test], classLatMedian, classLonMedian,
        #                              csv_file_test)

        _, _, train_acc_at_161, distances, latlon_true, latlon_pred = geo_eval2(my_model, features[idx_train], U_train,
                                                                                classLatMedian, classLonMedian,
                                                                                userLocation, node2cluster_arr[idx_train])

    else:
        meanDis, MedianDis, accAT161, distances, latlon_true, latlon_pred = geo_eval(my_model, features[idx_test],
                                                                                     U_test,
                                                                                     classLatMedian, classLonMedian,
                                                                                     userLocation)
        # save_coordinate_true_predict(distances, latlon_true, latlon_pred, labels[idx_test], classLatMedian, classLonMedian,
        #                              csv_file_test)

        _, _, train_acc_at_161, distances, latlon_true, latlon_pred = geo_eval(my_model, features[idx_train], U_train,
                                                                               classLatMedian, classLonMedian,
                                                                               userLocation)

    print("train_acc_at_161:{}\t\tTest:\tMean:{}\t\tMedian:{}\t\tAcc@161:{}\n"
          .format(train_acc_at_161, meanDis, MedianDis, accAT161))
    # save_coordinate_true_predict(distances, latlon_true, latlon_pred, labels[idx_train], classLatMedian, classLonMedian,
    #                              csv_file_train)

    # write time, args and results down to file, format is important.
    timeStr = time.strftime("%Y-%m-%d %H:%M:%S\t", time.localtime(time.time()))
    argsStr = "-dump_file:{}\t-degree:{}\t-lr:{}\t-decay:{}".format(
        args.dump_file, args.degree, args.lr, args.weight_decay)
    resultStr = "Dev:\tepoch_best:{}\t\tval_acc_best:{}\n".format(epoch_best, val_acc_best) + \
                "train_acc_at_161:{}\t\tTest:\tMean:{}\t\tMedian:{}\t\tAcc@161:{}".format(
                    train_acc_at_161, meanDis, MedianDis, accAT161)
    content = "\n" + timeStr + "\n" + argsStr + "\n" + resultStr + "\n"
    with open('result/influ_doc128_sgc.txt', 'a') as f:
        f.write(content)
    f.close()


def save_coordinate_true_predict(distances, latlon_true, latlon_pred, labels, classLatMedian, classLonMedian,
                                 coordinate_file):
    """
    :return: record the true and false users.
    """
    true_array, pred_array, dis_array = np.array(latlon_true), np.array(latlon_pred), np.array(distances)
    dis_array = dis_array.reshape(len(dis_array), 1)
    labels = labels.data.cpu().numpy().tolist()
    lab_list = list()
    for l in labels:
        lat_lab, lon_lab = classLatMedian[str(l)], classLonMedian[str(l)]
        lab_list.append([lat_lab, lon_lab, l])
    lab_array = np.array(lab_list)
    combine = np.hstack((true_array, pred_array, lab_array, dis_array))
    out_str = "id,true_lat,true_lon,pred_lat,pred_lon,pred_class,lab_lat,lab_lon,lab_class,distance,class_acc,acc@161\n"
    with open(coordinate_file, 'w') as f:
        for i, coor in enumerate(combine):
            if coor[4] == coor[7]:
                sign = "Yes"
            else:
                sign = "No"
            if coor[8] <= 161:
                sign2 = "In."
            else:
                sign2 = "Out"
            out_str += str(i) + ',' + str(coor[0]) + ',' + str(coor[1]) + ',' + str(coor[2]) + ',' + str(coor[3]) \
                       + ',' + str(coor[4]) + ',' + str(coor[5]) + ',' + str(coor[6]) + ',' + str(coor[7]) \
                       + ',' + str(coor[8]) + ',' + sign + ',' + sign2 + "\n"
            if (i % 1000 == 0) or (i == len(combine) - 1):
                f.write(out_str)
                out_str = ""
    f.close()


if __name__ == '__main__':
    # update learning rate and restart
    args = parse_args(sys.argv[1:])

    degree_rate = [2]
    weight_decay_rate = [5e-7]
    lr_rate = [0.0001, 0.0001, 0.001]
    for degree in degree_rate:
        args.degree = degree
        for weight_decay in weight_decay_rate:
            args.weight_decay = weight_decay
            for lr in lr_rate:
                args.lr = lr
                print("degree:{}\t\tweight_decay:{}\t\tlr:{}".format(args.degree, args.weight_decay, args.lr))
                main()

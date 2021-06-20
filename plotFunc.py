# coding='utf-8'

import matplotlib.pyplot as plt
import numpy as np
import os


def t_SNE_visualisations():
    from time import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold, datasets

    n_points = 1000
    # X 是一个(1000, 3)的 2 维数据，color 是一个(1000,)的 1 维数据
    X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
    n_neighbors = 10
    n_components = 2

    fig = plt.figure(figsize=(8, 8))
    # 创建了一个 figure，标题为"Manifold Learning with 1000 points, 10 neighbors"
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (1000, n_neighbors), fontsize=14)

    ''' 绘制 S 曲线的 3D 图像 '''
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)  # 初始化视角

    '''t-SNE'''
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(X)  # 转换后的输出
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    ax = fig.add_subplot(212)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    # plt.axis('tight')

    plt.show()


def draw_representations(X, y, k=4, seed=77, do_pca=True, filename='./Pics/output.png'):
    """
    :param X: features with high dimension.
    :param y: labels corresponding to the features, also called class.
    :param k: the kinds of class to show.
    :param seed: the random seed.
    :param do_pca: if true, use PCA to decrease the dimension.
    :param filename: the path and format(e.g. .png .pdf) to save the result.
    :return:
    """
    import matplotlib.pyplot as plt
    from collections import Counter
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import scipy as sp

    print("start draw_representations ...")
    class_count = Counter(y.tolist()).most_common(k)
    num_samples = class_count[3][1] - class_count[3][1] % 10
    all_lbls = []
    all_samples = []
    for i, cc in enumerate(class_count):
        lbl, _ = cc
        samples = X[y == lbl][0:num_samples, :]
        samples = samples.todense() if sp.sparse.issparse(samples) else samples
        lbls = y[y == lbl][0:num_samples]
        lbls[:] = i
        all_samples.append(samples)
        all_lbls.append(lbls)
    all_lbls = np.hstack(all_lbls)
    all_samples = np.vstack(all_samples)
    if do_pca:
        pca = PCA(n_components=50, random_state=seed)
        all_samples = pca.fit_transform(all_samples)
    tsne = TSNE(n_components=2, random_state=seed)
    embeddings = tsne.fit_transform(all_samples)
    chosen_indices = np.random.choice(np.arange(embeddings.shape[0]), size=k * min(50, num_samples), replace=False)
    chosen_embeddings = embeddings[chosen_indices, :]
    chosen_ys = all_lbls[chosen_indices]

    plt.axis('off')
    # plt.title("samples: {}    class: {}".format(min(50, num_samples), k))
    # plt.scatter(chosen_embeddings[:, 0], chosen_embeddings[:, 1], chosen_embeddings[:, 2],
    #             c=chosen_ys, cmap=plt.cm.Spectral)
    plt.scatter(chosen_embeddings[:, 0], chosen_embeddings[:, 1], c=chosen_ys, cmap=plt.cm.get_cmap("Set1", k))
    plt.savefig(filename)
    plt.close()


def read_y_data_from_file(fPath):
    y_data = []
    with open(fPath, 'r') as f:
        for line in f.readlines():
            if len(line) > 2:
                accData = line.split("\t")[-1].split("\n")[0].split(":")
                if len(accData) == 2 and accData[0] == 'Acc@161':
                    y_data.append(round(float(accData[1]), 2))
    f.close()
    return y_data


def read_x_data_from_file(fPath):
    x_data = []
    with open(fPath, 'r') as f:
        for line in f.readlines():
            if len(line) > 2:
                lrData = line.split("\t")[-2].split(":")
                if len(lrData) == 2 and lrData[0] == '-lr':
                    x_data.append(round(float(lrData[1]), 3))
    f.close()
    return x_data


def read_data_from_file(file_name, match_str):
    """
    :param file_name: (str) the path and name of file
    :param match_str: (str) the match string
    :return:(list) the match data
    """
    match_data = list()
    with open(file_name, 'r') as fin:
        for line in fin:
            line = line.split()
            for word in line:
                if match_str in word:
                    num = float(word.split(":")[-1])
                    match_data.append(round(num, 3))
                    break
    fin.close()
    return match_data


def get_files_name(file_dir):
    files_list = []
    for file in os.listdir(file_dir):
        files_list.append(os.path.join(file_dir, file))
    return files_list


def main():
    """choose the result file"""
    # cmu:
    # file_dir = "./res_cmu_ProNE/try_all/"

    # na:
    # file_dir = "./res_na/try_all/"

    '''na_new: (modify some parameters of Doc2Vec)'''
    # file_dir = "./res_na/res_na_new/"

    '''multi_hid: (multi hidden layers.)'''
    # file_dir = "./res_na/multi_hid/"

    '''multi_hid: (multi hidden layers with 0.1 dropout.)'''
    # file_dir = "./res_na/multi_hid_drop01/"
    # file_dir = "./res_na/multi_hid512_drop01/"
    # file_dir = "./res_na/multi_hid1024_drop01/"

    '''na_dim_64:(features dimension: 512-->64)'''
    # file_dir = "./res_na/dim64_degree4_no_hid/"

    '''na_dim_128:(features dimension: 512-->128)'''
    # file_dir = "./res_na/dim128_degree4_no_hid/"

    '''na_dim_256:(features dimension: 512-->256)'''
    # file_dir = "./res_na/dim256_degree4_no_hid/"

    '''na_dim_1024:(features dimension: 512-->1024)'''
    # file_dir = "./res_na/dim1024_degree4_no_hid/"

    '''na_dim_1024_drop01:(features dimension: 512-->1024+drop01)'''
    # file_dir = "./res_na/dim1024_degree4_no_hid_drop01/"

    '''na_dim_2048:(features dimension: 512-->2048)'''
    # file_dir = "./res_na/dim2048_degree4_no_hid/"

    '''na_dim_3072:(features dimension: 512-->3072)'''
    # file_dir = "./res_na/dim3072_degree4_no_hid/"

    '''degree 3::na_dim_64:(features dimension: 512-->64)'''
    # file_dir = "./res_na/dim64_degree3_no_hid/"

    '''degree 3::na_dim_128:(features dimension: 512-->128)'''
    # file_dir = "./res_na/dim128_degree3_no_hid/"

    '''degree 3::na_dim_256:(features dimension: 512-->256)'''
    # file_dir = "./res_na/dim256_degree3_no_hid/"

    '''degree 3::na_dim_512:(features dimension: 512-->512)'''
    # file_dir = "./res_na/dim512_degree3_no_hid/"

    '''degree 3::na_dim_1024:(features dimension: 512-->1024)'''
    # file_dir = "./res_na/dim1024_degree3_no_hid/"

    '''degree 3::na_dim_1024_drop01:(features dimension: 512-->1024+drop01)'''
    # file_dir = "./res_na/dim1024_degree3_no_hid_drop01/"

    '''degree 3::na_dim_2048:(features dimension: 512-->2048)'''
    # file_dir = "./res_na/dim2048_degree3_no_hid/"

    '''degree 3::na_dim_3072:(features dimension: 512-->3072)'''
    # file_dir = "./res_na/dim3072_degree3_no_hid/"

    '''sgc_dim1024:(features dimension: 512-->1024)'''
    # file_dir = "./res_na/sgc_dim1024/"

    '''sgc_dim2048:(features dimension: 512-->2048)'''
    # file_dir = "./res_na/sgc_dim2048/"

    '''batch size:[64, 128, 256, 512, 1024, 2048, 4096, 8192]'''
    # file_dir = "./res_na/dim2048_degree3_batch/"

    '''batch size:[4096, 8192, 16384, 32768] add small learning rate'''
    # file_dir = "./res_na/dim2048_degree3_batch_lr0.0001/"

    '''batch size:[4096, 8192, 16384, 32768] add small learning rate and softmax layer'''
    # file_dir = "./res_na/dim2048_degree3_batch_lr0.0001_soft/"

    # world:
    # file_dir = "./res_world/try_all/"

    '''res_world_no_process:(retain the stop words and single punctuation, the dim=128)'''
    # file_dir = "./res_world/res_world_no_process_no_sum/"

    '''add multi hidden layer'''
    # file_dir = "./res_world/dim256_multi_hid512_drop01/"
    # file_dir = "./res_world/dim256_multi_hid1024_drop01/"

    '''world_dim_256:(features dimension: 128-->256)'''
    # file_dir = "./res_world/dim256_degree4_no_hid/"

    '''world_dim_512:(features dimension: 128-->512)'''
    # file_dir = "./res_world/dim512_degree4_no_hid/"

    '''world_dim_1024:(features dimension: 128-->1024)'''
    # file_dir = "./res_world/dim1024_degree4_no_hid/"

    '''world_dim_2048:(features dimension: 128-->2048)'''
    # file_dir = "./res_world/dim2048_degree4_no_hid/"

    '''world_dim_4096:(features dimension: 128-->4096)'''
    # file_dir = "./res_world/dim4096_degree4_no_hid/"

    '''degree 3::world_dim_64:(features dimension: 128-->64)'''
    # file_dir = "./res_world/dim64_degree3_no_hid/"

    '''degree 3::world_dim_128:(features dimension: 128-->128)'''
    # file_dir = "./res_world/dim128_degree3_no_hid/"

    '''degree 3::world_dim_256:(features dimension: 128-->256)'''
    # file_dir = "./res_world/dim256_degree3_no_hid/"

    '''degree 3::world_dim_512:(features dimension: 128-->512)'''
    # file_dir = "./res_world/dim512_degree3_no_hid/"

    '''degree 3::world_dim_1024:(features dimension: 128-->1024)'''
    # file_dir = "./res_world/dim1024_degree3_no_hid/"

    '''degree 3::world_dim_2048:(features dimension: 128-->2048)'''
    # file_dir = "./res_world/dim2048_degree3_no_hid/"

    '''degree 3::world_dim_4096:(features dimension: 128-->4096)'''
    file_dir = "./res_world/dim4096_degree3_no_hid/"

    '''sgc_dim1024:(features dimension: 128-->1024)'''
    # file_dir = "./res_world/sgc_dim1024/"

    '''sgc_dim2048:(features dimension: 128-->2048)'''
    # file_dir = "./res_world/sgc_dim2048/"

    '''batch size:[64, 128, 256, 512, 1024, 2048, 4096, 8192]'''
    # file_dir = "./res_world/dim2048_degree3_batch/"

    files = get_files_name(file_dir)
    for file_name in files:
        plot_from_the_res_file(file_name)


def plot_from_the_res_file(file_name):
    # get data from file_name
    lr = read_data_from_file(file_name, "-lr")
    epoch_best = read_data_from_file(file_name, "epoch_best")
    val_acc_best = read_data_from_file(file_name, "val_acc_best")
    train_acc_at_161 = read_data_from_file(file_name, "train_acc_at_161")
    mean = read_data_from_file(file_name, "Mean")
    median = read_data_from_file(file_name, "Median")
    test_acc_at_161 = read_data_from_file(file_name, "Acc@161")

    # show information
    print("{}: num of lr:{}".format(file_name, len(lr)))
    sorted_test_acc = sorted(test_acc_at_161, reverse=True)[0:5]
    ave_value = np.mean(sorted_test_acc)
    print("Acc@161 Top5:{}\t\t Mean:{}\t\t Best:{}".format(
        sorted_test_acc, round(ave_value, 2), sorted_test_acc[0], ))
    index = test_acc_at_161.index(max(test_acc_at_161))
    max_acc_corres_mean = mean[index]
    max_acc_corres_median = median[index]
    print("when Acc@161 is best, Mean:{}\t\t median:{}\n\n".format(
        max_acc_corres_mean, max_acc_corres_median))

    # exit(0)

    # plt data
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(lr, train_acc_at_161, label='train_Acc@161', color='b', linestyle='solid', linewidth=1)
    ax1.plot(lr, val_acc_best, label='val_acc_best', color='g', linestyle='solid', linewidth=1)
    ax1.plot(lr, test_acc_at_161, label='test_Acc@161', color='r', linestyle='solid', linewidth=1)
    ax1.set_ylabel('Acc@161')
    ax1.legend(loc='upper left')
    for a, b in zip(lr, test_acc_at_161):  # set num tag
        ax1.text(a, b, round(b, 1), ha='center', va='bottom', fontsize=8)
    for a, b in zip(lr, train_acc_at_161):  # set num tag
        ax1.text(a, b, round(b, 1), ha='center', va='bottom', fontsize=8)

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(lr, epoch_best, label='epoch_best', color='c', linestyle='solid', linewidth=1)
    ax2.plot(lr, mean, label='mean', color='m', linestyle='solid', linewidth=1)
    ax2.plot(lr, median, label='median', color='y', linestyle='solid', linewidth=1)
    ax2.set_ylabel('epoch and error')
    ax2.legend(loc='upper right')
    for a, b in zip(lr, mean):  # set num tag
        ax2.text(a, b, round(b, 1), ha='center', va='bottom', fontsize=8)

    for a, b in zip(lr, median):  # set num tag
        ax2.text(a, b, round(b, 1), ha='center', va='bottom', fontsize=8)

    plt.title('Acc@161(train,val,test) change with the learning rate\n{}'.format(file_name))
    plt.show()


# plt.figure(1)   # 创建图表1
# plt.title('Acc@161(train,val,test) change with the learning rate\n{}'.format(file_name))
# plt.xlabel('learning rate')
# plt.ylabel('Acc@161')
# plt.plot(lr, train_acc_at_161, label='train_Acc@161', color='b', linestyle='solid', linewidth=1)
# plt.plot(lr, val_acc_best, label='val_acc_best', color='g', linestyle='solid', linewidth=1)
# plt.plot(lr, test_acc_at_161, label='test_Acc@161', color='r', linestyle='solid', linewidth=1)
# plt.legend()
# # plt.xticks(x_data[9:len(x_data):10])  # 横轴显示间隔
# plt.show()

# plt.plot(x_data, y_data4, label='L2 : 5e-10', color='c', linestyle='solid', linewidth=1)
# plt.plot(x_data, y_data5, label='d:3 L:5e-8', color='m', linestyle='solid', linewidth=1)
# plt.plot(x_data, y_data6, label='d:3 L:5e-9', color='y', linestyle='solid', linewidth=1)

# plot方法的关键字参数color(或c)用来设置线的颜色,可取值为:
# b: blue
# g: green
# r: red
# c: cyan
# m: magenta
# y: yellow
# k: black
# w: white

# plot方法的关键字参数linestyle(或ls)用来设置线的样式,可取值为:
# -, solid
# --, dashed
# -., dashdot
# :, dotted
# '', ' ', None

# plt.legend('best')接受一个loc关键字参数来设定图例的位置
# 0: 'best'
# 1: 'upper right'
# 2: 'upper left'
# 3: 'lower left'
# 4: 'lower right'
# 5: 'right'
# 6: 'center left'
# 7: 'center right'
# 8: 'lower center'
# 9: 'upper center'
# 10: 'center'


def aug_normalized_adjacency(adj):
    import scipy.sparse as sp
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


# normalization according to row, each row represent a feature
def feature_normalization1(dt):
    mean_num = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mean_num) / sigma


if __name__ == '__main__':
    # main()

    import pickle
    from dataProcess import load_obj

    data = load_obj("./data/cmu/dump_doc_dim_128.pkl")
    adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, \
    classLatMedian, classLonMedian, userLocation = data

    labels = np.hstack((Y_train, Y_dev, Y_test))

    features = np.vstack((X_train, X_dev, X_test))  # doc2vec features.
    # features = np.load("./data/cmu/bert_cmu_dim768.emd.npy")			# bert features.

    # features standardizing normalization.
    features = feature_normalization1(features)

    # for i in range(0, 5):
    # 	draw_representations(features, labels, k=4, seed=77, do_pca=True,
    # 					 filename='./Pics/doc2vec_std_{}.png'.format(i))
    # exit(1)

    # degree of SGC.
    adj = aug_normalized_adjacency(adj)
    adj = adj.todense()

    features = np.dot(adj, features)
    print("done, degree 1.")
    for i in range(0, 5):
        draw_representations(features, labels, k=4, seed=77, do_pca=True,
                             filename='./result/pic/d2v_std_degree1_{}.png'.format(i))

    features = np.dot(adj, features)
    print("done, degree 2.")
    for i in range(0, 5):
        draw_representations(features, labels, k=4, seed=77, do_pca=True,
                             filename='./result/pic/d2v_std_degree2_{}.png'.format(i))

    features = np.dot(adj, features)
    print("done, degree 3.")
    for i in range(0, 5):
        draw_representations(features, labels, k=4, seed=77, do_pca=True,
                             filename='./result/pic/d2v_std_degree3_{}.png'.format(i))

    features = np.dot(adj, features)
    print("done, degree 4.")
    for i in range(0, 5):
        draw_representations(features, labels, k=4, seed=77, do_pca=True,
                             filename='./result/pic/d2v_std_degree4_{}.png'.format(i))

import random
import time
import csv
import numpy as np
import math
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  # 训练集，测试集划分函数
import torch
import argparse
from readdata import readdata
from bbo2_ds_dnm import bbo2_ds_dnm
from bbo2_ds_dnm import md_DNM
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="iris")
    parser.add_argument('--opt', type=str, default="BBO")
    parser.add_argument('--M', type=int, default=12)
    parser.add_argument('--k', type=float, default=10)
    parser.add_argument('--ks', type=float, default=1)
    parser.add_argument('--iter', type=int, default=300)
    parser.add_argument('--maxrun', type=int, default=1)
    parser.add_argument('--popsize', type=int, default=100)
    parser.add_argument('--test_size', type=float, default=0.4)
    parser.add_argument('--pMutation', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--dtype', type=str, default="different")
    parser.add_argument('--somafunc', type=str, default="tanh")
    parser.add_argument('--normalization', type=bool, default=True)
    parser.add_argument('--random_seed', type=int, default=100)
    args = parser.parse_args()

    s_time = time.time()
    # read data
    '''二分类: breast, blood-transfusion, heart, ionosphere, parkinsons, raisin, caesarian;
       多分类: glass, wine, car, iris, seeds, ecoli'''
    '''opt: BBO， BP， GA， PSO， PBIL， IA， ES， AFSA'''

    # DATA_PATH = "breast"
    DATA = args.dataset
    feature, target = readdata(DATA)

    # 设置训练集数据80%，测试集20%
    x_train0, x_test0, y_train, y_test = train_test_split(feature, target, test_size=args.test_size, random_state=22)
    # 归一化(也就是所说的min-max标准化)通过调用sklearn库的标准化函数
    if args.normalization:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x_train0)
        x_test = min_max_scaler.fit_transform(x_test0)
    else:
        x_train = x_train0
        x_test = x_test0

    # 将数据类型转换为tensor方便pytorch使用
    train_data = torch.FloatTensor(x_train)
    # y_train = torch.FloatTensor(y_train)
    test_data = torch.FloatTensor(x_test)
    # y_test = torch.FloatTensor(y_test)

    # class_num = np.max(target) + 1
    class_num = len(np.unique(target))
    train_num = np.size(np.array(train_data), 0)
    train_dim = np.size(np.array(train_data), 1)
    test_num = np.size(np.array(test_data), 0)
    # y_train = torch.reshape(y_train, (train_num, 1))
    # y_test = torch.reshape(y_test, (test_num, 1))

    train_target = torch.zeros(train_num, class_num)
    test_target = torch.zeros(test_num, class_num)
    for i in range(train_num):
        train_target[i, y_train[i]] = 1

    for i in range(test_num):
        test_target[i, y_test[i]] = 1

    D = train_dim * args.M * 2


    class DNMpara:
        def __init__(self, M, k, ks, D, dtype, popsize, iter, somafunc):
            self.M = M
            self.k = k
            self.ks = ks
            self.D = D
            self.dtype = dtype
            self.popsize = popsize
            self.somafunc = somafunc
            self.iter = iter


    BestChart = torch.zeros(1, args.maxrun)
    acc = torch.zeros(1, args.maxrun)
    auc = torch.zeros(1, args.maxrun)
    for runnum in range(args.maxrun):
        random_seed = args.random_seed + runnum
        setup_seed(random_seed)
        # train
        net = DNMpara(args.M, args.k, args.ks, D, args.dtype, args.popsize, args.iter, args.somafunc)
        w, q, ws, qs, ds, best, out = bbo2_ds_dnm(net, train_data, train_target, class_num, random_seed)

        # test
        test_fito = md_DNM(test_data, net, w, q, ws, qs, class_num, ds)

        y_t = np.reshape(np.array(test_target), (test_num * class_num))
        y_score = np.reshape(np.array(test_fito), (test_num * class_num))
        test_fit = torch.max(test_fito, 1)[1].numpy()
        gt = torch.max(test_target, 1)[1].numpy()
        acc[0, runnum] = (test_fit == gt).sum() / test_num
        fpr, tpr, thresholds = metrics.roc_curve(y_t.ravel(), y_score.ravel())
        auc[0, runnum] = metrics.auc(fpr, tpr)
        print("runtime:", runnum, "acc:", acc[0, runnum])

    e_time = time.time()

    print('dataset:', args.dataset, 'opt:', args.opt, '\n',
          auc, 'mean:', torch.std_mean(auc, 1), 'auc' '\n',
          acc, 'mean:', torch.std_mean(acc, 1), 'acc')
    print('runtime:', e_time - s_time)


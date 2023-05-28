#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
import argparse
import os
import pickle
import gzip
import copy
import itertools
import time
import torch.nn.functional as nnf

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report

import collections
import pickle
import ast
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import scipy.stats
import io

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from matplotlib import cm

from sklearn.model_selection import StratifiedKFold

import statistics

import __main__
import consts
from model import *
from tempfile import TemporaryFile


def output_scores(y_true, y_pred, only_accuracy=False, average='binary'):
    print("\ny_tryue = ", y_true)
    print("y_pred = ", y_pred)
    metrics = [accuracy_score(y_true, y_pred)]
    if not only_accuracy:
        metrics.extend([
            precision_score(y_true, y_pred, average=average),
            recall_score(y_true, y_pred, average=average),
            f1_score(y_true, y_pred, average=average),
            balanced_accuracy_score(y_true, y_pred, adjusted=True)
        ])
    names = ['Accuracy', 'Precision', 'Recall', 'F1',
             'Youden'] if not only_accuracy else ["Accuracy"]
    print(('{:>11}'*len(names)).format(*names))
    print((' {:.8f}'*len(metrics)).format(*metrics))
    return {name: metric for name, metric in zip(names, metrics)}


class OurDataset(Dataset):
    def __init__(self, data, labels=None, attack_vector=None, multiclass=None):
        self.data = data
        assert not np.isnan(self.data).any(
        ), "data is nan: {}".format(self.data)
        print("IN DATASET ", attack_vector)
        classes = consts.classes

        if multiclass:
            self.labels = attack_vector
            assert (self.data.shape[0] == self.labels.shape[0])
            lb_style = LabelBinarizer()
            lb_style.fit(classes)
            print("Label binarizer classes = ", lb_style.classes_)
            self.labels = lb_style.transform(self.labels)
            print("IN DATASET labels = ", self.labels)
            self.attacks = lb_style.classes_

        else:
            self.labels = labels
            assert not np.isnan(labels).any(
            ), "labels is nan: {}".format(labels)
            assert (self.data.shape[0] == self.labels.shape[0])

    def __getitem__(self, index):
        data, labels = torch.FloatTensor(
            self.data[index, :]), torch.FloatTensor(self.labels[index, :])
        return data, labels

    def __len__(self):
        return self.data.shape[0]


# def get_logdir(fold, n_fold):
#     return os.path.join('runs', current_time + '_' + socket.gethostname() + "_" + str(fold) +"_"+str(n_fold))


# Deep Learning
############################


def create_binary_prediction_eager(test_indices):
    """ Get binary performance on test-set"""

    test_data = torch.utils.data.Subset(dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=opt.batchSize, shuffle=False)

    samples = 0
    n_outputs = opt.nLayers+2

    y_pred_list = [[] for _ in range(n_outputs)]
    y_list = []

    net.eval()
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data = data.to(device)
            samples += data.shape[0]
            labels = labels.to(device)
            print("Labels = ", labels)
            outputs, _ = net(data)
            # Tensor (dimensions, classes_count) -> [ [0.1, 0.2, 0.1 ... 0.6] ]
            index_pred = outputs
            print("outputs = ", index_pred)

            for output_index, output in enumerate(outputs):
                y_pred_list[output_index].append(torch.round(
                    torch.sigmoid(output.detach()).squeeze()).cpu().numpy())

            y_list.append(labels.cpu().numpy())
        y_list = [a.squeeze().tolist() for a in y_list]
        y_list = [item for sublist in y_list for item in sublist]
        scores = {}
        print("y_pred_list = ", y_pred_list)
        print("y_list = ", y_list)

        for i, output in enumerate(y_pred_list):
            y_pred_list[i] = [a.squeeze().tolist() for a in output]
            y_pred_list[i] = [item for sublist in y_pred_list[i]
                              for item in sublist]
            scores['Layer_{}'.format(i)] = output_scores(
                y_list, y_pred_list[i])
        print("scores = ", scores)


def multiclass_eager(net, dataset, device, evaluate=False):
    """ Get test-set performance if evaluate is True
    else create accuracy per layer and class plot """

    batchSize = 1  # FIXME: this is a workaround
    # print("evaluate = ", evaluate, " test indeces multi = ", test_indices)
    # test_data = torch.utils.data.Subset(dataset, test_indices)
    test_data = dataset
    # print("test data = ", list(test_data))
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batchSize, shuffle=False)
    # print("test loadert = ", list(test_loader))
    print("len test data = ", len(test_loader))

    n_outputs = 5
    y_pred_list = [[] for _ in range(n_outputs)]
    y_list = []
    prob_list = []
    with torch.no_grad():
        net.eval()
        for data, labels in tqdm(test_loader):
            X_batch = data.to(device)
            # print("X_batch = ", X_batch)
            outputs, _ = net(X_batch)
            # print("outputs = ", outputs)
            probs = []
            for output_index, y_test_pred in enumerate(outputs):
                y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
                # print("вероятности классов y_pred_softmax = ", y_pred_softmax)
                # print("output_index = ", output_index, " y_pred_softmax = ", y_pred_softmax)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                # print("y_pred_tags = ", y_pred_tags.cpu().numpy())
                y_pred_list[output_index].append(y_pred_tags.cpu().numpy())
                # res = torch.sigmoid(y_test_pred.detach()).squeeze().detach()
                prob = nnf.softmax(y_test_pred, dim=1)
                # print("PREDS = ", torch.exp(y_test_pred))
                # print("PROB = ", prob)
                probs.append(torch.round(prob, decimals=4))
                # print("ROUNDED PROBS = ", torch.round(prob, decimals=4))

            max_prob, _ = torch.max(probs[-1], dim=1)
            # print("max probs = ", torch.max(probs[-1], dim=1))
            prob_list.append(max_prob.item())
            _, labels = torch.max(labels, dim=1)  # максимум в первом столбце
            # print("метка класса (label) = ", labels.cpu().numpy())
            y_list.append(labels.cpu().numpy())

    print("final prob_list = ", prob_list)
    y_list = [a.squeeze().tolist() for a in y_list]
    print("final y list = ", y_list)
    for i, output in enumerate(y_pred_list):
        y_pred_list[i] = [a.squeeze().tolist() for a in output]

    print("final y_pred = ", y_pred_list[-1])
    return y_pred_list[-1], prob_list


def make_predictions(input_csv_name):
    df = pd.read_csv(input_csv_name, nrows=100000).fillna(0)
    # df = df[df['flowDurationMilliseconds'] < 1000 * 60 * 60 * 24 * 10]

    del df['flowStartMilliseconds']
    del df['sourceIPAddress']
    del df['destinationIPAddress']
    attack_vector = np.array(list(df['Attack']))
    assert len(attack_vector.shape) == 1
    # print("df = ", df)
    # print("attack_vector = ", attack_vector, "\n")
    # print("unique attacks = ", np.unique(attack_vector))

    del df['Attack']
    features = df.columns[:-1]
    # print("features = ", features)

    data = df.values
    # print('#'*40)
    # print(">> Data shape ", data.shape)
    # print('#'*40)

    assert len(attack_vector) == len(data)
    columns = list(df)
    # print('#'*40)
    # print(">> Features ", columns)
    # print('#'*40)

    x, y = data[:, :-1].astype(np.float32), data[:, -1:].astype(np.uint8)
    # print("\n y = ", y)
    file_name = input_csv_name[:-4]+"_normal"
    # print("file_name = ", file_name)
    # if opt.normalizationData == "":
    file_name_for_normalization_data = file_name+"_normalization_data.pickle"
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    stds[stds == 0.0] = 1.0
    print("x = ", x)
    print("means = ", means)
    print("stds = ", stds)

    # meanDf = pd.DataFrame(means)
    # meanDf.to_csv("means.csv")
    # stdDf = pd.DataFrame(stds)
    # stdDf.to_csv("std.csv")
    means = pd.read_csv("means_main.csv")["values"].to_numpy()
    stds = pd.read_csv("std_main.csv")["values"].to_numpy()
    # print("HERE")
    # means = means["values"].to_numpy()

    print('mneans  after load  = ', means)

    assert means.shape[0] == x.shape[1], "means.shape: {}, x.shape: {}".format(
        means.shape, x.shape)
    assert stds.shape[0] == x.shape[1], "stds.shape: {}, x.shape: {}".format(
        stds.shape, x.shape)
    assert not (stds == 0).any(), "stds: {}".format(stds)
    x = (x-means)/stds

    dataset = OurDataset(x, y, np.expand_dims(
        attack_vector, axis=1), multiclass=True)
    print("dataset  = ", dataset, "\n")
    n_classes = dataset.labels.shape[-1]
    print("nclasses =   = ", n_classes, "\n")

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    net = EagerNet(x.shape[-1], n_classes, 3, 128, device).to(device)
    setattr(__main__, "EagerNet", EagerNet)
    net_path = "runs/May16_20-38-09_MacBook-Air-Anton.local_0_3/net_m_250335522.pth"
    net = torch.load(net_path, map_location=device)

    res, prob_list = multiclass_eager(net, dataset, device)
    return res, prob_list

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os, re, sys
import random
import math
import pandas as pd
import tensorflow as tf
import warnings
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#------------------------------------------------------------------------------
def init():
    np.set_printoptions(threshold=sys.maxsize)
    sys.setrecursionlimit(40000)
    random.seed(42)                             # For reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

# ----------------------------------------------------------------------------
def print_error(str):
    print('\033[91m' + str + '\033[0m')

# ----------------------------------------------------------------------------
def print_tabulated(list):
    print('\t'.join('%.4f' % x if type(x) is np.float64 or type(x) is float else str(x) for x in list))

# ----------------------------------------------------------------------------
def print_stats(var_name, var):
    print(' - {}: {} - shape {} - min {:.2f} - max {:.2f} - mean {:.2f} - std {:.2f}'.format(
            var_name, type(var), var.shape, np.min(var), np.max(var), np.mean(var), np.std(var)))

# ----------------------------------------------------------------------------
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

# ----------------------------------------------------------------------------
# Return the list of files in folder
def list_dirs(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))]

# ----------------------------------------------------------------------------
# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

#------------------------------------------------------------------------------
def generateLabelNoise(Y, percent):
    assert percent >= 0 and percent <= 100
    assert Y.ndim == 1 and len(Y) > 0

    arrayIndexes = range(len(Y))
    nb_changes = (percent * len(Y) / 100) / 2

    for i in range(0, nb_changes):
        posit1 = random.randrange(len(arrayIndexes))
        index1 = arrayIndexes[posit1]
        del arrayIndexes[posit1]

        while True:  # search a distinct label
            posit2 = random.randrange(len(arrayIndexes))
            index2 = arrayIndexes[posit2]
            if Y[index1] != Y[index2]:
                del arrayIndexes[posit2]
                break
        aux = Y[index1]
        Y[index1] = Y[index2]
        Y[index2] = aux

#------------------------------------------------------------------------------
def generateAttributeNoise(X, percent):
    assert percent >= 0 and percent <= 100
    assert X.ndim == 2 and len(X) > 0 and len(X[0]) > 0

    minVal = np.zeros((len(X[0])))
    maxVal = np.zeros((len(X[0])))
    for c in range(len(X[0])):
        minVal[c] = min(X[:,c])
        maxVal[c] = max(X[:,c])
        #print(c, 'min', minVal[c], 'max', maxVal[c])

    numChanges = 0
    total = 0
    for r in range(len(X)):
        for c in range(len(X[r])):
            total += 1
            if random.randint(0, 100) < percent:
                X[r,c] = random.uniform(minVal[c], maxVal[c])
                numChanges += 1


#------------------------------------------------------------------------------
def l2norm(X):
    norm = 0
    for i in range(len(X)):
        if X[i] < 0:
            X[i] = 0
        else:
            norm += X[i] * X[i]
    if norm != 0:
        norm = math.sqrt(norm)
        X /= norm


#------------------------------------------------------------------------------
def get_sufix_noise(config):
    sufix_noise = ''
    if config.lnoise > 0 or config.anoise > 0:
        noise_type = '_label_noise' if config.lnoise > 0 else '_attr_noise'
        noise_level = config.lnoise if config.lnoise > 0 else config.anoise
        sufix_noise = noise_type + str(noise_level)
    return sufix_noise


# -----------------------------------------------------------------------------
def calculate_best_k(X_train, Y_train, arrayKValues, n_jobs, maxK=-1):
    if maxK == -1:
        maxK = np.max(arrayKValues)

    clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=(maxK+1), leaf_size=1, n_jobs=n_jobs)
    clf.fit( X_train, Y_train )
    neighbors_list = clf.kneighbors(X_train, n_neighbors=None, return_distance=False)

    index = neighbors_list[:, 1:]           # Remove the own prototype
    y_pred = Y_train[ index ]

    best_f1 = -1
    best_k = -1
    for k in arrayKValues:
        if k > maxK:
                continue

        m, nb = stats.mode( y_pred[:, :k], axis=1 )

        with warnings.catch_warnings(record=True): # ignore warnings
            precision, recall, f1, support = precision_recall_fscore_support(Y_train, m.flatten(), average=None)
        f1 = np.average(f1, None, support)

        if f1 > best_f1:
            #print(f1, k)
            best_f1 = f1
            best_k = k

    assert best_k != -1
    return best_k


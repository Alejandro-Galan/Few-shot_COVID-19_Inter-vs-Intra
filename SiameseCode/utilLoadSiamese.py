# -*- coding: utf-8 -*-
from __future__ import print_function
import math, random
import cv2, copy
import keras
import numpy as np
import keras.backend as K
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist, fashion_mnist, cifar10


#------------------------------------------------------------------------------
def __normalize(x_t, y_t):
    channels = 1 if len(x_t.shape) == 3 else x_t.shape[3]
    if K.image_data_format() == 'channels_first':
        x_t = x_t.reshape(x_t.shape[0], channels, x_t.shape[1], x_t.shape[2])
        y_t = y_t.reshape((y_t.shape[0],))
    else:
        x_t = x_t.reshape(x_t.shape[0], x_t.shape[1], x_t.shape[2], channels)
        y_t = y_t.reshape((y_t.shape[0],))
    x_t = x_t.astype('float32')
    x_t /= 255.0
    return x_t, y_t


#------------------------------------------------------------------------------
def fillMinorityClass(vector, n_t):
    if n_t > len(vector):
        aux_vector = np.zeros( ((n_t,) + vector.shape[1:]), dtype=type(vector) )
        aux_vector[:len(vector)] = [elem for elem in vector]
        for i in range(len(vector), n_t):
            aux_vector[i] = random.choice(vector)
        return aux_vector
    else:
        return vector

#------------------------------------------------------------------------------
def __transform_to_matrix(x_t, y_t, n_t=None, start=0, num_samples_classes=None):
    # Obtains classes (health and disease in this case)
    keys = np.sort(np.unique(y_t))
    # Divides the instances
    dict_t = {key: x_t[y_t[:, int(key)] == 1.0] for key in keys}
    sum = len(dict_t[0]) + len(dict_t[1])
    
    # Stores [Original distribution, weight of each class per sample] 
    weight_bal = [num_samples_classes, {int(key): sum / (2 * len(dict_t[key])) for key in dict_t} ]
    # Fill vector of minority class
    if num_samples_classes:
        m_t = max(num_samples_classes.values())
        dict_t_red = {key: dict_t[key][start:num_samples_classes[key]+start] for key in keys}
        dict_t_f = {key: dict_t[key][:num_samples_classes[key]] if len(dict_t_red[key]) != num_samples_classes[key] else dict_t_red[key] for key in keys}
        dict_t = {key: fillMinorityClass(dict_t_f[key], m_t) for key in keys}
    else:    
        m_t = min(len(dict_t[key]) for key in keys)
        # n_t = m_t if n_t is None else n_t #max(n_t, m_t)
        dict_t = {key: dict_t[key][start:m_t+start] for key in keys}
    mat_t = np.concatenate([[dict_t[key]] for key in keys])
    return mat_t, weight_bal


#------------------------------------------------------------------------------
# Avoid duplication of minority class
def __transform_to_matrix_val(x_t, y_t, n_t=None, start=0, num_samples_classes=None):
    keys = np.sort(np.unique(y_t))
    dict_t = {key: x_t[y_t[:, int(key)] == 1.0] for key in keys}
    sum = len(dict_t[0]) + len(dict_t[1])
    
    m_t = max(len(dict_t[key]) for key in keys)
    # n_t = m_t if n_t is None else max(n_t, m_t)
    
    # weight_bal = [{0:len(dict_t[0]), 1:len(dict_t[1])}, {int(key): sum / (2 * len(dict_t[key])) for key in dict_t} ]

    # n_t = n_t // 3 if n_t > 3 else 1
    offset_start, original_samples = copy.deepcopy(num_samples_classes), copy.deepcopy(num_samples_classes)
    original_samples[1] = original_samples[1] // 3 if original_samples[1] > 3 else 1
    original_samples[0] = original_samples[0] // 3 if original_samples[0] > 3 else 1


    # Fill vector of minority class
    m_t = max(original_samples.values())

    dict_t_red = {key: dict_t[key][offset_start[key]+start:offset_start[key]+original_samples[key]+start] for key in keys}
    dict_t_f = {key: dict_t[key][:original_samples[key]] if len(dict_t_red[key]) != original_samples[key] else dict_t_red[key] for key in keys}
    dict_t = {key: fillMinorityClass(dict_t_f[key], m_t) for key in keys}

    # dict_t = {key: dict_t[key][start:m_t+start] for key in keys}

    mat_t = np.concatenate([[dict_t[key]] for key in keys])
    return mat_t, None


#------------------------------------------------------------------------------
def add_data_augmentation(mat_tr, aug):
    assert aug > 0
    datagen1 = ImageDataGenerator( rotation_range=1,
                                                                        width_shift_range=0.01,
                                                                        height_shift_range=0.01,
                                                                        zoom_range=0.01 )
    new_mat_tr = []

    for l in range(len(mat_tr)):
        x_label = mat_tr[l]
        y_label = l * np.ones((len(x_label)))

        NUM_AUGS = aug
        mat_aug = x_label.copy()
        for r in range(NUM_AUGS):
            x_aug, y_aug = next( datagen1.flow(x_label, y_label, shuffle=False, batch_size=len(x_label)) )

            """for i in range(len(y_label)):
                print('Label:', y_label[i], y_aug[i])
                cv2.imshow("img_o", x_label[i])
                cv2.imshow("img_a", x_aug[i])
                cv2.waitKey(0)"""

            mat_aug = np.concatenate((mat_aug, x_aug), axis=0)
            #print(' - Aug shape:', np.array(mat_aug).shape, x_aug.shape, y_aug.shape)

        new_mat_tr.append( mat_aug )

    return np.array(new_mat_tr)



#------------------------------------------------------------------------------
def load(dataset, n_tr, aug=0, start=0):
    if dataset == 'mnist':
        (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
        #x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1], x_tr.shape[2], 1)
        #x_te = x_te.reshape(x_te.shape[0], x_te.shape[1], x_te.shape[2], 1)
    elif dataset == 'fashion_mnist':
        (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
    elif dataset == 'usps':
        X, y = fetch_openml('usps', return_X_y=True)
        size = int( math.sqrt( X.shape[1] ) )
        X = X.reshape(X.shape[0], size, size, 1)
        #print(X.dtype, np.min(X), np.max(X))
        X = cv2.normalize(X, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(X.dtype, np.min(X), np.max(X))
        #X = np.array(X, dtype='int')
        #print(X.dtype, np.min(X), np.max(X))
        y = np.array(map(int, y))
        y -= 1
        x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    else:
        raise Exception('Unknown dataset name')

    """import sys
    np.set_printoptions(threshold=sys.maxsize)
    print(y_te)"""

    print(' - x_train:', x_tr.shape)
    print(' - y_train:', y_tr.shape)
    print(' - x_test:', x_te.shape)
    print(' - y_test:', y_te.shape)

    x_tr, y_tr = __normalize(x_tr, y_tr)
    x_te, y_te = __normalize(x_te, y_te)

    mat_tr = __transform_to_matrix(x_tr, y_tr, n_tr, start)
    mat_te = __transform_to_matrix(x_te, y_te)

    print(' - mat_tr:', mat_tr.shape)

    if aug > 0:
        print('# Add data augmentation...')
        mat_tr = add_data_augmentation(mat_tr, aug)
        print(' - Aug tr shape:', mat_tr.shape)

    return mat_tr, mat_te

# -*- coding: utf-8 -*-
from __future__ import print_function
from SiameseCode.utilGeneratorPair import apply_data_augmentation
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import os
import utils.util as util
import tensorflow as tf

# ----------------------------------------------------------------------------
def __run_validations(dataset):
    input_shape = None
    nb_classes = None
    assert dataset is not None
    assert len(dataset) > 0

    assert 'name' in dataset and dataset['name'] is not None
    assert 'x_train' in dataset and dataset['x_train'] is not None
    assert 'y_train' in dataset and dataset['y_train'] is not None
    assert 'x_test' in dataset and dataset['x_test'] is not None
    assert 'y_test' in dataset and dataset['y_test'] is not None

    if input_shape is None:
        input_shape = dataset['x_train'].shape[1:]
    assert input_shape == dataset['x_train'].shape[1:]
    assert input_shape == dataset['x_test'].shape[1:]

    if nb_classes is None:
        nb_classes = len(dataset['y_train'][1])
    assert nb_classes == len(dataset['y_train'][1])
    assert nb_classes == len(dataset['y_test'][1])

    return input_shape, nb_classes


# ----------------------------------------------------------------------------
def __smooth_labels(dataset):
    # print('Smoothing labels...')
    dataset['y_train'] = util.smooth_labels(dataset['y_train'].astype('float32'))


# ----------------------------------------------------------------------------
def __normalize(dataset, norm_type):
    x_train = dataset['x_train']
    x_test = dataset['x_test']
    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')

    # print('Normalize dataset', dataset['name'])
    # print(' - Min / max / avg train:', np.min(x_train), ' / ', np.max(x_train), ' / ', np.mean(x_train))
    # print(' - Min / max / avg test:', np.min(x_test), ' / ', np.max(x_test), ' / ', np.mean(x_test))

    if norm_type == '255':
        x_train /= 255.
        x_test /= 255.
    elif norm_type == 'standard':
        mean = np.mean(x_train)
        std = np.std(x_train)
        x_train = (x_train - mean) / (std + 0.00001)
        x_test = (x_test - mean) / (std + 0.00001)
    elif norm_type == 'mean':
        mean = np.mean(x_train)
        x_train -= mean
        x_test -= mean
    elif norm_type == 'imagenet':
        x_train = tf.keras.applications.resnet_v2.preprocess_input(
            x_train, data_format=None
        )
        x_test = tf.keras.applications.resnet_v2.preprocess_input(
            x_test, data_format=None
        )

    # print(' After norm...')
    # print(' - Min / max / avg train:', np.min(x_train), ' / ', np.max(x_train), ' / ', np.mean(x_train))
    # print(' - Min / max / avg test:', np.min(x_test), ' / ', np.max(x_test), ' / ', np.mean(x_test))

    dataset['x_train'] = x_train
    dataset['x_test'] = x_test


# -----------------------------------------------------------------------------
#     CSV: column "finding"
#             COVID-19
#             COVID-19, ARDS
#     filename
def __loadCSV(csv_file):
    # print(' - Reading', csv_file, '...')
    labels = {}
    csv_data = util.load_csv(csv_file, sep=',', header=0, usecols = ['filename', 'finding'])
    for l in csv_data:
        key = os.path.splitext(l[1])[0]
        if l[0] == 'COVID-19' or l[0] == 'COVID-19, ARDS':
            labels[key] = 1
        else:
            labels[key] = 0
    return labels

# -----------------------------------------------------------------------------
def load_one_dataset(path, part, label, truncate=-1, countImages=None, limit_size = None, offset = 0, load_by_disk=True):
    # print(' - Loading the', part, 'set of', path, '...')
    X = []
    Y = []
    fullpath = os.path.join('datasets', path, part)
    if label==-1:
        labels = __loadCSV(os.path.join('datasets', path,'metadata.csv'))

    cv_counter, first_elems = -1, -1
    images_explored = [0,0]
    positive = negative = 0
    all_files = len(util.list_files(fullpath))
    loops = 1
    
    SAVE_LIMIT_LOADED = 2
    # Assuming minority class will always be negative images
    if (part != "test" and (all_files <= limit_size[label] * SAVE_LIMIT_LOADED or offset >= all_files )  ): # if the full dataset is not enough for partitions and need to be itered
        full_load = limit_size[label] + limit_size[label] // 3 if limit_size[label] // 3 > 0 else limit_size[label] + 1  

        offset = offset % all_files

        # offset = offset / full_load * all_files / SAVE_LIMIT_LOADED # Assuming 5cv
        if offset + full_load >= all_files:
            first_elems =  offset + full_load - all_files # Total of elements not available so will be needed to pick from beggining

        loops = int((limit_size[label] * 1.4) // all_files) + 1


    for _ in range(loops):
        for fname in util.list_files(fullpath):
            # Cross validation we choose a set in train
            cv_counter += 1
            if cv_counter < offset and cv_counter >= first_elems:
                continue

            # Limit training size at earlier time, reduces execution time 
            # and allows execution on some limited machines. In this case whole folder is same label
            if label != -1 and limit_size and limit_size[label] != -1 and countImages[label] >= limit_size[label] * 10: # Train and val sets
                break

            if not load_by_disk:
                img = cv2.imread(fname, cv2.IMREAD_COLOR)

            # if size != -1:
            #     img = cv2.resize(img, (size, size), interpolation = cv2.INTER_CUBIC )
            #print(label, fname, img.shape)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)
            c = label
            if label == -1:
                #print(fname, os.path.splitext( os.path.basename(fname) )[0])
                c = labels[ os.path.splitext( os.path.basename(fname) )[0] ]
                if c == 0:
                    negative += 1
                else:
                    positive += 1
            assert c==0 or c==1
            
            # Avoid loading too many images to reduce comp. time. Train and val
            if limit_size and limit_size[label] != -1 and countImages[c] >= limit_size[c] * SAVE_LIMIT_LOADED:
                continue            

            if not load_by_disk:
                X.append( img )
            else:
                X.append( fname )
            Y.append( c )

            if countImages:
                countImages[c] += 1 # Depending of the positive/negative image
                images_explored[c] += 1


            # if truncate>0 and len(Y) == truncate:
            #     break

        # In case repetition is needed
        cv_counter = 0
    # print( positive, negative )

    return X, Y, countImages, images_explored

# -----------------------------------------------------------------------------
def load_one_dataset_test(path, part, label, truncate=-1, limitImages=None, load_by_disk=True):
    # print(' - Loading the', part, 'set of', path, '...')
    X = []
    Y = []
    fullpath = os.path.join('datasets', path, part)
    if label==-1:
        labels = __loadCSV(os.path.join('datasets', path,'metadata.csv'))

    images_counter = [0,0] # Start after train
    positive = negative = 0

    for fname in util.list_files(fullpath):
        if not load_by_disk:
            img = cv2.imread(fname, cv2.IMREAD_COLOR)

        c = label
        if label == -1:
            c = labels[ os.path.splitext( os.path.basename(fname) )[0] ]
        assert c==0 or c==1
        
        images_counter[c] += 1

        # if images_counter[c] < limitImages[c]:
        #     continue
        
        if not load_by_disk:
            X.append( img )
        else:
            X.append( fname )
        Y.append( c )

    return X, Y


# -----------------------------------------------------------------------------
def getProportionImbalance(pathNeg, pathPos, part, label1, label2):
    fullpathNeg = os.path.join('datasets', pathNeg, part)
    fullpathPos = os.path.join('datasets', pathPos, part)

    # For now, only label 2 can be different
    if label2==-1:
        labels = __loadCSV(os.path.join('datasets', pathPos,'metadata.csv'))
        negatives = sum(v == 0 for v in labels.values())
        positives = sum(v == 1 for v in labels.values())
        negatives += len(util.list_files(fullpathNeg))
        # No positives with route as that information is in "labels"
    else:
        negatives = len(util.list_files(fullpathNeg))
        positives = len(util.list_files(fullpathPos))
    
    imbalance = positives / negatives
    return imbalance, negatives, positives

# -----------------------------------------------------------------------------
def load_datasets(path1, label1, path2, label2, truncate, imbalance_to_dataset = 100, limit_train_size = None, offset = 0, load_by_disk=True):
    # In case we wanted to reduce the data base on real imbalance
    # imbalancePerc, negative_imgs, positive_imgs = getProportionImbalance(path1, path2, 'train', label1, label2)
    countImages = [0,0] # Counter to control number of positive and negative samples

    limit_size = [limit_train_size, imbalance_to_dataset]
    if limit_train_size == -1:
        limit_size = [170, 170] # Multiplied by 10 inside

    x_train1, y_train1, countImages, images_exp1 = load_one_dataset(path1, 'train', label1, truncate, 
                                    countImages, limit_size, offset, load_by_disk )

    x_train2, y_train2, countImages, images_exp2 = load_one_dataset(path2, 'train', label2, truncate, 
                                    countImages, limit_size, offset, load_by_disk )

    x_test1, y_test1 = load_one_dataset_test(path1, 'test', label1, truncate, limitImages=[0,0], load_by_disk=load_by_disk) # No need to assign limit, there are 2 different paths
    x_test2, y_test2 = load_one_dataset_test(path2, 'test', label2, truncate, limitImages=[0,0], load_by_disk=load_by_disk)

    return np.concatenate((x_train1,x_train2), axis=0),\
                    np.concatenate((y_train1,y_train2),axis=0),\
                    np.concatenate((x_test1,x_test2), axis=0),\
                    np.concatenate((y_test1,y_test2), axis=0)

# -----------------------------------------------------------------------------
# 1. ChestX-ray (-) and Github-COVID (+-)
# 2. PadChest (-) and BIMCV+
# 3. BIMCV- and BIMCV+
def load(config, verbose):

    # Just one
    # for bd_name in config.dbs:
    # print('Loading dataset', config.dbs, '...')
    if int(config.dbs)==1:
        x_train, y_train, x_test, y_test = load_datasets('ChestX-ray14', 0, 'GitHub-COVID', -1, 
                                        -1, imbalance_to_dataset = config.samples, limit_train_size = config.limit_train_size, 
                                        offset = config.start, load_by_disk=config.load_by_disk)
    elif int(config.dbs)==2:
        x_train, y_train, x_test, y_test = load_datasets('PadChest', 0, 'bimcv+', 1, 
                                        -1, imbalance_to_dataset = config.samples, limit_train_size = config.limit_train_size, 
                                        offset = config.start, load_by_disk=config.load_by_disk)
    elif int(config.dbs)==3:
        x_train, y_train, x_test, y_test = load_datasets('bimcv-', 0, 'bimcv+', 1, 
                                        -1, imbalance_to_dataset = config.samples, limit_train_size = config.limit_train_size, 
                                        offset = config.start, load_by_disk=config.load_by_disk)
    else:
        raise Exception('Unknowm dataset')

    if verbose:
        # print(' - Clases:', np.unique(y_dis))
        print(' - Samples per class in train:', np.unique(y_train, return_counts=True)[1])
        print(' - Samples per class in test:', np.unique(y_test, return_counts=True)[1])

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    # if verbose:
    #     print(' - X Train:', x_train.shape)
    #     print(' - Y Train:', y_train.shape)
    #     print(' - X Test:', x_test.shape)
    #     print(' - Y Test:', y_test.shape)

    dataset = {'name': config.dbs,
                'x_train': x_train, 'y_train': y_train,
                'x_test': x_test, 'y_test': y_test
            } 

    input_shape, num_labels = __run_validations(dataset)

    # # Label smooth
    # if config.lsmooth:
    #     __smooth_labels(dataset)

    if not config.load_by_disk:
        # Normalize
        norm = "standard"
        if config.pretrained_weights:
            norm = config.pretrained_weights
        __normalize(dataset, norm)

    # Shuffle
    dataset['x_train'], dataset['y_train'] = shuffle(dataset['x_train'], dataset['y_train'])
    dataset['x_test'], dataset['y_test'] = shuffle(dataset['x_test'], dataset['y_test'])

    # Truncate?
    """if config.truncate:
        print('Truncate...')
        for i in range(len(datasets)):
            factor = 0.2 # 0.1
            new_len_tr = int( factor * len(datasets[i]['x_train']) )
            new_len_te = int( factor * len(datasets[i]['x_test']) )
            datasets[i]['x_train'] = datasets[i]['x_train'][:new_len_tr]
            datasets[i]['y_train'] = datasets[i]['y_train'][:new_len_tr]
            datasets[i]['x_test'] = datasets[i]['x_test'][:new_len_te]
            datasets[i]['y_test'] = datasets[i]['y_test'][:new_len_te]"""

    return dataset, input_shape, num_labels

# Read a set of img from disk and normalize
def read_set_img(x_paths, labels_augm, config):
    imgs = []

    if type(x_paths) is np.str_ and x_paths.size != 1:
        breakpoint()
    if type(x_paths) is np.str_ and x_paths.size == 1:
        imgs.append(cv2.imread(x_paths, cv2.IMREAD_COLOR))
    elif type(x_paths) is str:
        imgs.append(cv2.imread(x_paths, cv2.IMREAD_COLOR))
    else:
        for path in x_paths:
            imgs.append(cv2.imread(path, cv2.IMREAD_COLOR))

    for i, path in enumerate(imgs):
        if labels_augm[i]:
            imgs[i] = apply_data_augmentation(imgs[i], labels_augm[i])


    norm_type = "standard"
    if config.pretrained_weights:
        norm_type = config.pretrained_weights
    imgs = np.asarray(imgs).astype('float32')


    if norm_type == '255':
        imgs /= 255.
    elif norm_type == 'standard':
        mean = np.mean(imgs)
        std = np.std(imgs)
        imgs = (imgs - mean) / (std + 0.00001)
    elif norm_type == 'mean':
        mean = np.mean(imgs)
        imgs -= mean
    elif norm_type == 'imagenet':
        imgs = tf.keras.applications.resnet_v2.preprocess_input(
            imgs, data_format=None
        )

    if len(imgs) == 1:
        return imgs[0]
    return imgs

# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings
gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')

import numpy as np
from random import randint
import argparse
import gc
import matplotlib
matplotlib.use('Agg')
import util
import utilLoad
import utilCNN
import utilDANN
import utilDANNModel
from tensorflow.keras import backend as K

util.init()
K.set_image_data_format('channels_last')

"""if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
"""

WEIGHTS_CNN_FOLDERNAME = 'WEIGHTS_CNN'
WEIGHTS_DANN_FOLDERNAME = 'WEIGHTS_DANN'
util.mkdirp( WEIGHTS_CNN_FOLDERNAME + '/truncated')
util.mkdirp( WEIGHTS_DANN_FOLDERNAME + '/truncated')


# -----------------------------------------------------------------------------
def train_cnn(datasets, input_shape, num_labels, WEIGHTS_CNN_FOLDERNAME, config):
    utilCNN.train_cnn(datasets, input_shape, num_labels, WEIGHTS_CNN_FOLDERNAME, config)


# -----------------------------------------------------------------------------
def train_dann(datasets, input_shape, num_labels, weights_foldername, config):
    for i in range(len(datasets)):
        if config.from_db is not None and config.from_db != datasets[i]['name']:
            continue

        for j in range(len(datasets)):
            if i == j or (config.to_db is not None and config.to_db != datasets[j]['name']):
                continue

            print(80*'-')
            print('SOURCE: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                datasets[i]['name'], datasets[i]['x_train'].shape, datasets[i]['y_train'].shape,
                datasets[i]['x_test'].shape, datasets[i]['y_test'].shape))
            print('TARGET: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                datasets[j]['name'], datasets[j]['x_train'].shape, datasets[j]['y_train'].shape,
                datasets[j]['x_test'].shape, datasets[j]['y_test'].shape))

            dann = utilDANNModel.DANNModel(config.model, input_shape, num_labels, config.batch)
            #dann_model = dann.build_tsne_model()
            #dann_vis = dann.build_dann_model()

            weights_filename = utilDANN.get_dann_weights_filename( weights_foldername,
                                                                                    datasets[i]['name'], datasets[j]['name'], config)

            if config.load == False:
                utilDANN.train_dann(dann,
                                                            datasets[i]['x_train'], datasets[i]['y_train'],
                                                            datasets[j]['x_train'], datasets[j]['y_train'],
                                                            config.epochs, config.batch, weights_filename)

            print('# Evaluate...')
            dann.load( weights_filename )  # Load the last save weights...
            source_loss, source_acc = dann.label_model.evaluate(datasets[i]['x_test'], datasets[i]['y_test'], batch_size=32, verbose=0)
            target_loss, target_acc = dann.label_model.evaluate(datasets[j]['x_test'], datasets[j]['y_test'], batch_size=32, verbose=0)
            print('Result: {}\t{}\t{:.4f}\t{:.4f}'.format(datasets[i]['name'], datasets[j]['name'], source_acc, target_acc))

            gc.collect()


# -----------------------------------------------------------------------------
def train_dann2(datasets, input_shape, num_labels, weights_foldername, config):
    for i in range(len(datasets)):
        if config.from_db is not None and config.from_db != datasets[i]['name']:
            continue

        for j in range(len(datasets)):
            if i == j or (config.to_db is not None and config.to_db != datasets[j]['name']):
                continue

            print(80*'-')
            print('SOURCE: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                datasets[i]['name'], datasets[i]['x_train'].shape, datasets[i]['y_train'].shape,
                datasets[i]['x_test'].shape, datasets[i]['y_test'].shape))
            print('TARGET: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                datasets[j]['name'], datasets[j]['x_train'].shape, datasets[j]['y_train'].shape,
                datasets[j]['x_test'].shape, datasets[j]['y_test'].shape))

            dann = utilDANNModel.DANNModel(config.model, input_shape, num_labels, config.batch)
            #dann_model = dann.build_tsne_model()
            #dann_vis = dann.build_dann_model()

            weights_filename = utilDANN.get_dann_weights_filename( weights_foldername,
                                                                                    datasets[i]['name'], datasets[j]['name'], config)

            if config.load == False:
                utilDANN.train_dann(dann,
                                                            datasets[i]['x_train'], datasets[i]['y_train'],
                                                            datasets[j]['x_train'], datasets[j]['y_train'],
                                                            config.epochs, config.batch, weights_filename)

            print('# Evaluate...')
            dann.load( weights_filename )  # Load the last save weights...
            source_loss, source_acc = dann.label_model.evaluate(datasets[i]['x_test'], datasets[i]['y_test'], batch_size=32, verbose=0)
            target_loss, target_acc = dann.label_model.evaluate(datasets[j]['x_test'], datasets[j]['y_test'], batch_size=32, verbose=0)
            print('Result: {}\t{}\t{:.4f}\t{:.4f}'.format(datasets[i]['name'], datasets[j]['name'], source_acc, target_acc))

            gc.collect()


# ----------------------------------------------------------------------------
def menu():
    parser = argparse.ArgumentParser(description='DANN')

    group0 = parser.add_argument_group('Training type')
    group0.add_argument('-type',   default='dann', type=str,     choices=['dann', 'dann2', 'cnn'],  help='Training type')
    group0.add_argument('-model', default=1,         type=int,        help='Model number (1,2,3,4,...)')

    group1 = parser.add_argument_group('Input data parameters')
    group1.add_argument('-dbs',   default=None, type=str, help='Comma separated list of BD names to load. Set None to load all.')
    group1.add_argument('-from',   default=None, dest='from_db', type=str, help='Use only this dataset as source.')
    group1.add_argument('-to',   default=None, dest='to_db', type=str, help='Use only this dataset as target.')
    group1.add_argument('-norm',   default='standard',                 type=str,
                                                choices=['mean', 'standard', '255', 'keras'], help='Input data normalization type')
    group1.add_argument('-size',   default=-1, type=int, help='Scale to this size. -1 to use default size.')
    group1.add_argument('--truncate',  default=-1,     type=int,      help='Truncate datasets to this size.')

    group3 = parser.add_argument_group('Training parameters')
    group3.add_argument('-e',          default=200,    type=int,         dest='epochs',         help='Number of epochs')
    group3.add_argument('-b',          default=64,     type=int,       dest='batch',            help='Batch size')
    group3.add_argument('--wloss',  action='store_true',               help='Activate class weight')
    group3.add_argument('--aug',    action='store_true',               help='Use data augmentation')

    group3.add_argument('-lr',        default=1.0,     type=float,    help='Learning rate')
    group3.add_argument('--lsmooth', action='store_true', help='Activate label smoothing.')
    group3.add_argument('-lda',      default=0.01,    type=float,    help='Reversal gradient lambda')

    parser.add_argument('--tsne',    action='store_true',               help='Activate TSNE')
    parser.add_argument('--load',  action='store_true',               help='Load weights.')
    parser.add_argument('--v',         action='store_true',      help='Activate verbose.')
    parser.add_argument('-gpu',    default='0',       type=str,         help='GPU')
    args = parser.parse_args()

    if args.dbs is not None and args.dbs.lower() != 'none':
        args.dbs = [int(x) for x in args.dbs.split(',')]
    else:
        args.dbs = [1,2,3]

    if args.from_db is not None and args.from_db.lower() == 'none':
        args.from_db = None
    if args.to_db is not None and args.to_db.lower() == 'none':
        args.to_db = None

    assert isinstance(args.dbs, (list, tuple, np.ndarray))

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    config = menu()

    datasets, input_shape, num_labels = utilLoad.load(config)

    print(' - Input shape:', input_shape)
    print(' - Num labels:', num_labels)

    if config.type == 'cnn':
        print('Train CNN...')
        train_cnn(datasets, input_shape, num_labels, WEIGHTS_CNN_FOLDERNAME, config)
    elif config.type == 'dann':
        print('Train DANN...')
        train_dann(datasets, input_shape, num_labels, WEIGHTS_DANN_FOLDERNAME, config)
    elif config.type == 'dann2':
        print('Train DANN 2...')
        train_dann2(datasets, input_shape, num_labels, WEIGHTS_DANN_FOLDERNAME, config)
    else:
        raise Exception('Unknown training type')

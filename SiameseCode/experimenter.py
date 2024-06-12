# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings
gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')
import numpy as np
import math
import cv2
import argparse
import util
import utilLoad
import utilGeneratorCNN
import utilGeneratorPair, utilGeneratorPairClass
import utilGeneratorTriplet, utilGeneratorTripletClass
import utilGeneratorTripletMem
import utilModels, utilModelCNN
import utilModelSiamese, utilModelSiameseClass
import utilModelTriplet, utilModelTripletClass
import utilModelTripletMem
import utilEvaluate
from scipy import stats
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

util.init()
K.set_image_data_format('channels_last')

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)

DATASETS = ['cifar10', 'mnist', 'fashion_mnist', 'usps']
MODES = ['cnn', 'pair', 'pairclass', 'triplet', 'tripletclass', 'tripletmem']


#------------------------------------------------------------------------------
def load_data(config):
    mat_tr, mat_te = utilLoad.load(config.db, config.train_size, config.augtr, config.start)

    mat_me = np.zeros((mat_tr.shape[0], config.dim))     # Para el modelo con memoria...

    if config.mode == 'cnn':
        gen_tr = utilGeneratorCNN.catgen(mat_tr, config.batch)
        gen_te = utilGeneratorCNN.catgen(mat_te, config.batch)
    elif config.mode == 'pair':
        gen_tr = utilGeneratorPair.pairgen(mat_tr, config.pos, config.neg, config.batch)
        gen_te = utilGeneratorPair.pairgen(mat_te, config.pos, config.neg, config.batch)
    elif config.mode == 'pairclass':
        gen_tr = utilGeneratorPairClass.pairgen(mat_tr, config.pos, config.neg, config.batch)
        gen_te = utilGeneratorPairClass.pairgen(mat_te, config.pos, config.neg, config.batch)
    elif config.mode == 'triplet':
        trip = 3
        gen_tr = utilGeneratorTriplet.tripletgen(mat_tr, trip, config.batch)
        gen_te = utilGeneratorTriplet.tripletgen(mat_te, trip, config.batch)
    elif config.mode == 'tripletclass':
        trip = 3
        gen_tr = utilGeneratorTripletClass.tripletgen(mat_tr, trip, config.batch)
        gen_te = utilGeneratorTripletClass.tripletgen(mat_te, trip, config.batch)
    elif config.mode == 'tripletmem':
        trip = 3
        gen_tr = utilGeneratorTripletMem.tripletgen(mat_tr, mat_me, trip, config.batch)
        gen_te = utilGeneratorTripletMem.tripletgen(mat_te, mat_me, trip, config.batch)
    else:
        raise Exception('Unknown generator mode')

    return mat_tr, mat_te, mat_me, gen_tr, gen_te


#------------------------------------------------------------------------------
def get_loss(config):
    if config.mode == 'cnn':
        return None
    elif config.mode == 'pair':
        if config.loss == 'a':
            return utilModelSiamese.get_loss_a(config.margin)
        elif config.loss == 'b':
            return utilModelSiamese.get_loss_b(config.margin)
        else: # c
            return utilModelSiamese.get_loss_c()
    elif config.mode == 'pairclass':
        return utilModelSiameseClass.contrastive_loss1
    elif config.mode == 'triplet':
        return utilModelTriplet.contrastive_loss1
    elif config.mode == 'tripletclass':
        return utilModelTripletClass.contrastive_loss1
    elif config.mode == 'tripletmem':
        return utilModelTripletMem.contrastive_loss1
    else:
        raise Exception('Unknown loss mode')

#------------------------------------------------------------------------------
def train_model(input_shape, nb_classes, gen_tr, mat_tr, mat_me, config):
    network_name = config.db
    if config.db == 'usps':
        network_name = 'mnist'
    base_model = getattr(utilModels, "base_" + network_name)(input_shape, config.dim)

    if config.mode == 'cnn':
        model = getattr(utilModelCNN, "cnn_" + network_name)(nb_classes, input_shape)
        emb_model = model
        utilModelCNN.compile(model, config.optimizer)
    elif config.mode == 'pair':
        model, emb_model = utilModelSiamese.create(base_model)
        utilModelSiamese.compile(model, config.loss, config.optimizer)
    elif config.mode == 'pairclass':
        model, emb_model = utilModelSiameseClass.create(base_model, nb_classes)
        utilModelSiameseClass.compile(model, config.loss, config.optimizer, config.contrib)
    elif config.mode == 'triplet':
        model, emb_model = utilModelTriplet.create(base_model)
        utilModelTriplet.compile(model, config.loss, config.optimizer)
    elif config.mode == 'tripletclass':
        model, emb_model = utilModelTripletClass.create(base_model, nb_classes)
        utilModelTripletClass.compile(model, config.loss, config.optimizer)
    elif config.mode == 'tripletmem':
        model, emb_model = utilModelTripletMem.create(base_model, config.dim)
        utilModelTripletMem.compile(model, config.loss, config.optimizer)
    else:
        raise Exception('Unknown network mode')

    if config.load == True:
        model.load_weights(config.weights_filename)
    elif config.mode != 'tripletmem':
        verbose = 2 if mat_tr.shape[1] >= 50 else 0
        model.fit_generator(gen_tr,  epochs=config.epochs, verbose=verbose)
        model.save_weights(config.weights_filename, overwrite=True)
    else:
        for e in range(config.epochs):
            print("Epoch {}".format(e + 1))
            loss = 0.0
            cont = 0
            for x, y in gen_tr:
                loss += model.train_on_batch(x, y)
                cont += 1
            for i in range(mat_tr.shape[0]):
                new_me = np.average(emb_model.predict(mat_tr[i]), axis=0)
                mat_me[i] = mat_me[i] * (1.0 - e / config.epochs) + new_me * (e / config.epochs)
            sys.stdout.flush()
        model.save_weights(config.weights_filename, overwrite=True)

    return emb_model


#------------------------------------------------------------------------------
def evaluate_knn(e_tr, y_tr, e_te, y_te):
    utilEvaluate.svr(e_tr, y_tr, e_te, y_te, 'linear')
    utilEvaluate.svr(e_tr, y_tr, e_te, y_te, 'poly')
    utilEvaluate.svr(e_tr, y_tr, e_te, y_te, 'rbf')

    utilEvaluate.knn(e_tr, y_tr, e_te, y_te, 1)
    utilEvaluate.knn(e_tr, y_tr, e_te, y_te, 5)
    utilEvaluate.knn(e_tr, y_tr, e_te, y_te, 15)
    utilEvaluate.knn(e_tr, y_tr, e_te, y_te, 25)

    utilEvaluate.rf(e_tr, y_tr, e_te, y_te, 50)
    utilEvaluate.rf(e_tr, y_tr, e_te, y_te, 100)
    utilEvaluate.rf(e_tr, y_tr, e_te, y_te, 200)


#------------------------------------------------------------------------------
def calculate_mode_acc(name, y_mode, y_te):
    m, nb = stats.mode( y_mode, axis=1 )
    acc = (y_te == m.flatten()).mean()
    print('Mode {} with {} preds -> Accuracy on test set: {:.2f}%'.format(
                    name, y_mode.shape[1], acc * 100))


#------------------------------------------------------------------------------
def run_evaluation_with_augmentation(emb_model, e_tr, y_tr, x_te, e_te, y_te):
    print('--')
    print('Evaluate with data augmentation...')

    NUM_PRED = 19
    svr_best_kernel = 'linear'
    knn_best_k = 1
    raf_best_estimators = 100

    y_mode_hist = np.zeros((len(e_te), NUM_PRED+1), dtype=int)
    y_mode_hist[:,0] = utilEvaluate.hist(e_tr, y_tr, e_te, y_te)

    y_mode_svr = np.zeros((len(e_te), NUM_PRED+1), dtype=int)
    y_mode_svr[:,0] = utilEvaluate.svr(e_tr, y_tr, e_te, y_te, svr_best_kernel, True)

    y_mode_knn = np.zeros((len(e_te), NUM_PRED+1), dtype=int)
    y_mode_knn[:,0] = utilEvaluate.knn(e_tr, y_tr, e_te, y_te, knn_best_k, True)

    y_mode_rf = np.zeros((len(e_te), NUM_PRED+1), dtype=int)
    y_mode_rf[:,0] = utilEvaluate.rf(e_tr, y_tr, e_te, y_te, raf_best_estimators, True)


    if False:
        for i in range(len(x_te)):
            print('Label:', y_te[i])
            #cv2.imwrite('IMGS/usps'+str(i)+'.png',x_train[i])
            cv2.imshow("img", x_te[i])
            cv2.waitKey(0)

    datagen = ImageDataGenerator( rotation_range=10,
                                                                    width_shift_range=0.05,
                                                                    height_shift_range=0.05,
                                                                    zoom_range=0.05)
    """datagen = ImageDataGenerator( rotation_range=5,
                                                                        width_shift_range=0.02,
                                                                        height_shift_range=0.02,
                                                                        zoom_range=0.02 )"""
    """datagen1 = ImageDataGenerator( rotation_range=20,
                                                                        width_shift_range=0.1,
                                                                        height_shift_range=0.1,
                                                                        zoom_range=0.1 )"""

    for r in range(NUM_PRED):
        print(r)
        x_ate, y_ate = next( datagen.flow(x_te, y_te, shuffle=False, batch_size=len(x_te)) )
        #print(' - Aug te shape:', x_ate.shape, y_ate.shape)

        """for i in range(len(x_te)):
            print('Label:', y_te[i], y_ate[i])
            #cv2.imwrite('IMGS/usps'+str(i)+'.png',x_train[i])
            cv2.imshow("img_o", x_te[i])
            cv2.imshow("img_a", x_ate[i])
            cv2.waitKey(0)"""

        e_ate = emb_model.predict(x_ate)

        y_mode_hist[:, r+1] = utilEvaluate.hist(e_tr, y_tr, e_ate, y_ate)
        y_mode_svr[:, r+1] = utilEvaluate.svr(e_tr, y_tr, e_ate, y_ate, svr_best_kernel, True)
        y_mode_knn[:, r+1] = utilEvaluate.knn(e_tr, y_tr, e_ate, y_ate, knn_best_k, True)
        y_mode_rf[:, r+1] = utilEvaluate.rf(e_tr, y_tr, e_ate, y_ate, raf_best_estimators, True)

    print('Y_mode shape:', y_mode_hist.shape)
    for r in range(1, NUM_PRED+1):
        calculate_mode_acc('hist', y_mode_hist[:, :r+1], y_te)
        calculate_mode_acc('svr', y_mode_svr[:, :r+1], y_te)
        calculate_mode_acc('knn', y_mode_knn[:, :r+1], y_te)
        calculate_mode_acc('rf', y_mode_rf[:, :r+1], y_te)


#------------------------------------------------------------------------------
def run_evaluation(emb_model, mat_tr, mat_te, config):
    print('Evaluate...')
    x_tr = np.reshape(mat_tr, (-1,) + mat_tr.shape[2:])
    e_tr = emb_model.predict(x_tr)
    y_tr = np.repeat(np.arange(mat_tr.shape[0]), mat_tr.shape[1])

    x_te = np.reshape(mat_te, (-1,) + mat_te.shape[2:])
    e_te = emb_model.predict(x_te)
    y_te = np.repeat(np.arange(mat_te.shape[0]), mat_te.shape[1])

    print(' - Train shape:', e_tr.shape, y_tr.shape)
    print(' - Test shape:', e_te.shape, y_te.shape)

    utilEvaluate.hist(e_tr, y_tr, e_te, y_te)
    if config.knn:
        evaluate_knn(e_tr, y_tr, e_te, y_te)

    """print('With l2...')
    for i in range(len(e_tr)):
        util.l2norm(e_tr[i,:])
    for i in range(len(e_te)):
        util.l2norm(e_te[i,:])

    utilEvaluate.hist(e_tr, y_tr, e_te, y_te)
    if config.knn:
        evaluate_knn(e_tr, y_tr, e_te, y_te)"""

    if args.augte == True:
        run_evaluation_with_augmentation(emb_model, e_tr, y_tr, x_te, e_te, y_te)



#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Siamese experimenter')
parser.add_argument('-db',  required=True, type=str, help='Dataset', choices=DATASETS)
parser.add_argument('-mode',  required=True, type=str, help='Mode', choices=MODES)

parser.add_argument('-size',   default=100,  dest='train_size', type=int,   help='Number of train samples per class')
parser.add_argument('-start',   default=0,  type=int,   help='Offset to get train samples')

parser.add_argument('-dim',   default=512,  type=int,   help='Embedding dimensions')
parser.add_argument('-opt',   default='adam', dest='optimizer', type=str, help='Optimizer')
parser.add_argument('-loss',   default='a', type=str, choices=['a', 'b', 'c'], help='Loss function')
parser.add_argument('-margin', default=1.0, type=float, help='Loss margin (only valid for loss a and b)')
parser.add_argument('-pos',   default=1,  type=int,   help='Number of positive pairs')
parser.add_argument('-neg',   default=5,  type=int,   help='Number of negative pairs')

parser.add_argument('-contrib', default=0.25, type=float, help='Class contribution (ONLY for the Siamese class network)')

parser.add_argument('-epoch',   default=200,  dest='epochs',  type=int,   help='Number of epochs')
parser.add_argument('-batch',   default=32,  dest='batch',   type=int,   help='Mini batch size')

parser.add_argument('-augtr',  default=0,  type=int,   help='Use train data augmentation')
parser.add_argument('--augte',  action='store_true',   help='Use test data augmentation')
parser.add_argument('--knn',  action='store_true',        help='Run validation with kNN, SVM and RaF')
parser.add_argument('--load',  action='store_true',        help='Load data')
parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')
args = parser.parse_args()

print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

util.mkdirp('WEIGHTS')
args.weights_filename = 'WEIGHTS/weights_db' + args.db \
                                                        + '_m' + args.mode \
                                                        + '_size' + str(args.train_size) \
                                                        + '_dim' + str(args.dim) \
                                                        + '_pos' + str(args.pos) \
                                                        + '_neg' + str(args.neg) \
                                                        + '_opt' + args.optimizer \
                                                        + '_loss' + args.loss \
                                                        + '_margin' + str(args.margin) \
                                                        + ('_augtr' + str(args.augtr) if args.augtr>0 else '' ) \
                                                        + ('_augte' if args.augte else '' ) \
                                                        + '_e' + str(args.epochs) \
                                                        + '_b' + str(args.batch) \
                                                        + '.h5'

# Parameters
# TODO - revisar si el loss es el mismo en todos............................................
args.loss = get_loss(args)


# Load data...
mat_tr, mat_te, mat_me, gen_tr, gen_te = load_data(args)

if mat_tr.shape[1] < args.batch:
    args.batch =  mat_tr.shape[1]
    print(' * New batch:', args.batch)


# Train model...
nb_classes = mat_tr.shape[0]
input_shape = mat_tr.shape[2:]
emb_model = train_model(input_shape, nb_classes, gen_tr, mat_tr, mat_me, args)


# Evaluate
if args.mode != 'cnn':
    run_evaluation(emb_model, mat_tr, mat_te, args)
else:
    print('Evaluate...')
    score, acc = emb_model.evaluate_generator(gen_te)
    print('Test loss:', score)
    print('Test accuracy:', acc * 100)


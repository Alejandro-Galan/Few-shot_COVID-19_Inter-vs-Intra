import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
import tensorflow as tf


#------------------------------------------------------------------------------
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


#------------------------------------------------------------------------------
# L_a
def get_loss_a(weights_per_sample = None, margin=1.0):
    def contrastive_loss1(y_true, y_pred):
        #margin = 1.0
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        error = K.mean((1 - y_true) * square_pred + y_true * margin_square)
        # if weights_per_sample:
        #     return weights_per_sample * error
        return error
    return contrastive_loss1


#------------------------------------------------------------------------------
# L_b
def get_loss_b(margin=1.0):
    def contrastive_loss2(y_true, y_pred):
        #margin = 1.0
        square_pred = K.square(y_pred)
        margin_square = K.maximum(margin - K.square(y_pred), 0)
        return K.mean((1 - y_true) * square_pred + y_true * margin_square)
    return contrastive_loss2


#------------------------------------------------------------------------------
# L_c
def get_loss_c():
    def contrastive_loss3(y_true, y_pred):
        N = 512
        beta = N
        epsilon = 1e-8
        square_pred = -K.log((-(K.sum(y_pred)) / beta) + 1 + epsilon)
        margin_square = -K.log((-(N - K.sum(y_pred)) / beta) + 1 + epsilon)
        return K.mean((1 - y_true) * square_pred + y_true * margin_square)
    return contrastive_loss3


#------------------------------------------------------------------------------
def pair_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))


#------------------------------------------------------------------------------
def create(base_model):
    i1 = Input(shape=base_model.input_shape[1:], name='i1')
    i2 = Input(shape=base_model.input_shape[1:], name='i2')
    p1 = base_model(i1)
    p2 = base_model(i2)
    o = Lambda(euclidean_distance)([p1, p2])
    siam_model = Model([i1, i2], o)
    emb_model = Model(i1, p1)
    return siam_model, emb_model


#------------------------------------------------------------------------------
def compile(siam_model, loss, optimizer):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=100,
                decay_rate=0.96)

    opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    siam_model.compile(loss=loss, optimizer=opt, metrics=[pair_accuracy])


#------------------------------------------------------------------------------
def compileFreeze(siam_model, loss, unfreeze = False):
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    lr = 0.1 # Initial lr
    decay_steps = 20
    # if unfreeze:
    #     lr = 0.00001
    #     decay_steps = 10000

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=decay_steps,
                decay_rate=0.96)

    opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    siam_model.compile(loss=loss, optimizer=opt, metrics=[pair_accuracy])

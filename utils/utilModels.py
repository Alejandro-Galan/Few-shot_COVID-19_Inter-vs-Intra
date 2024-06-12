from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
import tensorflow as tf
from utils import utilConfig


#------------------------------------------------------------------------------
def base_resnet(config, pretrained_weights):
    emb_d = config.dim
    if config.type=="transfer-learning":
        emb_d = 1
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(weights=pretrained_weights, include_top=False)
    

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    predictions = Dense(emb_d, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # Transfer learning from other dataset
    if config.init_dbs:
        utilConfig.loadModelNp(model, config.init_weights)        

    return model

#------------------------------------------------------------------------------
def cnn_resnet(config, pretrained_weights):
    emb_d = config.dim

    model = tf.keras.applications.resnet_v2.ResNet50V2(weights=pretrained_weights, include_top=False)
    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    predictions = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=model.input, outputs=predictions)     

    return model

#------------------------------------------------------------------------------
def cnn_compile(siam_model, loss):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=100,
                decay_rate=0.96)

    opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    siam_model.compile(loss=loss, optimizer=opt)
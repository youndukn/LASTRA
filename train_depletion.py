import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Concatenate, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,GlobalAveragePooling2D, \
    Conv3D, AveragePooling2D, MaxPooling2D, Reshape, Multiply, LeakyReLU, LSTM, GRU

from keras.losses import categorical_crossentropy
from keras.models import Model, load_model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
# from matplotlib.pyplot import imshow
from keras.optimizers import Adam
import os
import pickle
import random

import keras.backend as K
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys

import math

import glob
import queue
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
path_checkpoint = './check_p/train_checkpoint.keras'
path_checkpoint_pre = './check_p/pre_checkpoint.keras'

from distribution import Distribution

xs_batch_input_temp_q = []
bs_batch_input_temp_q = []
ps_batch_output_temp_q = []
dp_batch_output_temp_q = []
bs_batch_output_temp_q = []

for i in range(20):
    xs_batch_input_temp_q.append([])
    bs_batch_input_temp_q.append([])
    ps_batch_output_temp_q.append([])
    dp_batch_output_temp_q.append([])
    bs_batch_output_temp_q.append([])

ad = [
    [0, 9, 9],
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
    [11, 10, 10],
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [22, 11, 11],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [33, 12, 12],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [44, 13, 13],
    [45, 13, 14],
    [46, 13, 15],
    [55, 14, 14],
]
sep_cen = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26]
sep_out = [7, 14, 20, 24, 27, 28]
import tensorflow as tf
b_norm = 1000
dp_norm = (1000, 100, 1000, 100, 1, 1, 100, 1)

def get_files_old():
    files = []
    for j in range(1, 4):
        for i in range(0, 14):
            to_string = '/media/youndukn/lastra/depl_data_{}/data_{}_s'.format(j, i)

            file_read = open(to_string, 'rb')
            files.append(file_read)

    return files

def get_files(excludes, shuffled=""):
    files = []
    for path in glob.glob("/media/youndukn/lastra/plants_data/*{}.inp".format(shuffled)):
        try:

            isExclude = False
            for exclude in excludes:
                if exclude in path:
                    isExclude = True
            if not isExclude:
                file_read = open(path, 'rb')
                files.append(file_read)
        except:
            pass

    return files

def get_files_with(includes, excludes, shuffled="", folders=["/media/youndukn/lastra/plants_data/"]):
    files = []
    for folder in folders:
        for path in glob.glob("{}*{}*".format(folder, shuffled)):
            try:

                isInclude = False
                for include in includes:
                    if include in path:
                        isInclude = True

                if len(includes) == 0:
                    isInclude = True

                isExclude = False
                for exclude in excludes:
                    if exclude in path:
                        isExclude = True

                if isInclude and not isExclude:
                    file_read = open(path, 'rb')
                    files.append(file_read)
            except:
                pass

    return files
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def identity_block_se(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def identity_block_se_lk(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X = Add()([X, X_shortcut])
    X = LeakyReLU()(X)

    return X



def identity_block_se_l(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolution_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X

def get_decoder(encoder_output, dim, state_size = 1280):

    decoder_initial_state = Input(shape=(state_size,),
                                  name='decoder_initial_state')

    decoder_input = Input(shape=(None, 1), name='decoder_input')

    decoder_gru1 = GRU(state_size, activation='relu', name='decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, activation='relu', name='decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, activation='relu', name='decoder_gru3',
                       return_sequences=True)

    decoder_dense = Dense(dim, name='decoder_output_pre')

    def connect_decoder(initial_state):
        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect all the GRU-layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)
        decoder_output = LeakyReLU(name='decoder_output')(decoder_output)

        return decoder_output

    decoder_output = connect_decoder(initial_state=encoder_output)

    return decoder_initial_state, decoder_input, decoder_output, connect_decoder


def get_decoder_xs(encoder_output, decoder_input, dim, state_size = 1280):

    decoder_initial_state = Input(shape=(state_size,),
                                  name='xs_decoder_initial_state')

    #decoder_input = Input(shape=(None, dim), name='r_decoder_input')

    decoder_gru1 = GRU(state_size, activation='relu', name='xs_decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, activation='relu', name='xs_decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, activation='relu', name='xs_decoder_gru3',
                       return_sequences=True)

    decoder_dense = Dense(dim, name='xs_decoder_output_pre')

    def connect_decoder(initial_state):
        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect all the GRU-layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)
        decoder_output = LeakyReLU(name='xs_decoder_output')(decoder_output)

        return decoder_output

    decoder_output = connect_decoder(initial_state=encoder_output)

    return decoder_initial_state, decoder_input, decoder_output, connect_decoder

def get_decoder_xs_real(encoder_output, i_dim, dim, state_size = 1280):

    decoder_initial_state = Input(shape=(state_size,),
                                  name='xs_decoder_initial_state_real')

    decoder_input = Input(shape=(None, i_dim), name='xs_decoder_input_real')

    decoder_gru1 = GRU(state_size, activation='relu', name='xs_decoder_gru1_real',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, activation='relu', name='xs_decoder_gru2_real',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, activation='relu', name='xs_decoder_gru3_real',
                       return_sequences=True)

    decoder_dense = Dense(dim, name='xs_decoder_output_pre_real')

    def connect_decoder(initial_state):
        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect all the GRU-layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)
        decoder_output = LeakyReLU(name='xs_decoder_output_real')(decoder_output)

        return decoder_output

    decoder_output = connect_decoder(initial_state=encoder_output)

    return decoder_initial_state, decoder_input, decoder_output, connect_decoder

def get_decoder_r(encoder_output, decoder_input, dim, state_size = 1280):

    decoder_initial_state = Input(shape=(state_size,),
                                  name='r_decoder_initial_state')

    #decoder_input = Input(shape=(None, dim), name='r_decoder_input')

    decoder_gru1 = GRU(state_size, activation='relu', name='r_decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, activation='relu', name='r_decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, activation='relu', name='r_decoder_gru3',
                       return_sequences=False)

    decoder_dense = Dense(dim, name='r_decoder_output_pre')

    def connect_decoder(initial_state):
        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect all the GRU-layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)
        decoder_output = LeakyReLU(name='r_decoder_output')(decoder_output)

        return decoder_output

    decoder_output = connect_decoder(initial_state=encoder_output)

    return decoder_initial_state, decoder_input, decoder_output, connect_decoder

def get_decoder_r_real(encoder_output, i_dim, dim, state_size = 1280):

    decoder_initial_state = Input(shape=(state_size,),
                                  name='r_decoder_initial_state_real')

    decoder_input = Input(shape=(None, i_dim), name='r_decoder_input_real')

    decoder_gru1 = GRU(state_size, activation='relu', name='r_decoder_gru1_real',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, activation='relu', name='r_decoder_gru2_real',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, activation='relu', name='r_decoder_gru3_real',
                       return_sequences=False)

    decoder_dense = Dense(dim, name='r_decoder_output_pre_real')

    def connect_decoder(initial_state):
        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect all the GRU-layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)
        decoder_output = LeakyReLU(name='r_decoder_output_real')(decoder_output)

        return decoder_output

    decoder_output = connect_decoder(initial_state=encoder_output)

    return decoder_initial_state, decoder_input, decoder_output, connect_decoder


def get_decoder_r_real_embedding(encoder_output, i_dim, dim, state_size = 1280):

    decoder_initial_state = Input(shape=(state_size,),
                                  name='r_decoder_initial_state_real')

    decoder_input = Input(shape=(None, i_dim), name='r_decoder_input_real')

    decoder_gru1 = GRU(state_size, activation='relu', name='r_decoder_gru1_real',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, activation='relu', name='r_decoder_gru2_real',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, activation='relu', name='r_decoder_gru3_real',
                       return_sequences=False)

    decoder_dense = Dense(dim, name='r_decoder_output_pre_real')

    def connect_decoder(initial_state):
        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect all the GRU-layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)
        decoder_output = LeakyReLU(name='r_decoder_output_real')(decoder_output)

        return decoder_output

    decoder_output = connect_decoder(initial_state=encoder_output)

    return decoder_initial_state, decoder_input, decoder_output, connect_decoder

def convolution_block_se(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X

def convolution_block_se_lk(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = LeakyReLU()(X)

    return X


def convolution_block_se_l(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X

def convolution_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X


def convolution_block_no(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    # X_shortcut = BatchNormalization(axis=3, name = bn_name_base + '1')(X_shortcut)

    X = Activation("relu")(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    # X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis= 3, name = 'bn_conv1')(X)
    # X = Activation('relu')(X)

    stage_filters = [64, 64, 256]
    X = convolution_block(X_input, f=3, filters=stage_filters, stage=1, block='a', s=1)
    X = identity_block(X, 3, stage_filters, stage=1, block='b')
    X = identity_block(X, 3, stage_filters, stage=1, block='c')

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f=3, filters=stage_filters, stage=2, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=2, block='b')
    X = identity_block(X, 3, stage_filters, stage=2, block='c')
    X = identity_block(X, 3, stage_filters, stage=2, block='d')

    stage_filters = [256, 256, 1024]
    X = convolution_block(X, f=3, filters=stage_filters, stage=3, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=3, block='b')
    X = identity_block(X, 3, stage_filters, stage=3, block='c')
    X = identity_block(X, 3, stage_filters, stage=3, block='d')
    X = identity_block(X, 3, stage_filters, stage=3, block='e')
    X = identity_block(X, 3, stage_filters, stage=3, block='f')

    stage_filters = [256, 256, 2048]
    X = convolution_block(X, f=3, filters=stage_filters, stage=4, block='a', s=3)
    X = identity_block(X, 3, stage_filters, stage=4, block='b')
    X = identity_block(X, 3, stage_filters, stage=4, block='c')

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def FullyConnected(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(128, name='fc1' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(4000, name='fc2' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(8000, name='fc3' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(4000, name='fc4' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(3000, name='fc5' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(classes, name='fc6' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="FullyConnected")

    return model


def FullyConnected(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(256 * 2, name='fc1' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(256 * 3, name='fc2' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(256 * 3, name='fc3' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(256 * 2, name='fc4' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(classes, name='fc6' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="FullyConnected")

    return model


def FullyConnected_Small_3(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(3, name='fc1' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('sigmoid')(X)
    X = Dense(classes, name='fc6' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="FullyConnected")

    return model


def FullyConnected_Small_6(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(3, name='fc1' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('sigmoid')(X)
    X = Dense(classes, name='fc6' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="FullyConnected")

    return model


def FullyConnected_Small_9(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(3, name='fc1' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('sigmoid')(X)
    X = Dense(classes, name='fc6' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="FullyConnected")

    return model


def FullyConnected_Small_3_3(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(3, name='fc1' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('sigmoid')(X)
    X = Dense(3, name='fc2' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('sigmoid')(X)
    X = Dense(classes, name='fc6' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="FullyConnected")

    return model


def ResNet10(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f=3, filters=stage_filters, stage=stage, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')
    X = identity_block(X, 3, stage_filters, stage=stage, block='d')
    stage = stage + 1

    stage_filters = [256, 256, 1024]
    X = convolution_block(X, f=3, filters=stage_filters, stage=3, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')
    X = identity_block(X, 3, stage_filters, stage=stage, block='d')
    X = identity_block(X, 3, stage_filters, stage=stage, block='e')
    X = identity_block(X, 3, stage_filters, stage=stage, block='f')
    stage = stage + 1

    stage_filters = [256, 256, 2048]
    X = convolution_block(X, f=3, filters=stage_filters, stage=stage, block='a', s=3)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNet5(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f=3, filters=stage_filters, stage=stage, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')
    X = identity_block(X, 3, stage_filters, stage=stage, block='d')
    stage = stage + 1

    stage_filters = [256, 256, 1024]
    X = convolution_block(X, f=3, filters=stage_filters, stage=3, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')
    X = identity_block(X, 3, stage_filters, stage=stage, block='d')
    X = identity_block(X, 3, stage_filters, stage=stage, block='e')
    X = identity_block(X, 3, stage_filters, stage=stage, block='f')
    stage = stage + 1

    stage_filters = [256, 256, 2048]
    X = convolution_block(X, f=3, filters=stage_filters, stage=stage, block='a', s=3)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNet2(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(2):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f=3, filters=stage_filters, stage=stage, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')
    X = identity_block(X, 3, stage_filters, stage=stage, block='d')
    stage = stage + 1

    stage_filters = [256, 256, 1024]
    X = convolution_block(X, f=3, filters=stage_filters, stage=3, block='a', s=2)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')
    X = identity_block(X, 3, stage_filters, stage=stage, block='d')
    X = identity_block(X, 3, stage_filters, stage=stage, block='e')
    X = identity_block(X, 3, stage_filters, stage=stage, block='f')
    stage = stage + 1

    stage_filters = [256, 256, 2048]
    X = convolution_block(X, f=3, filters=stage_filters, stage=stage, block='a', s=3)
    X = identity_block(X, 3, stage_filters, stage=stage, block='b')
    X = identity_block(X, 3, stage_filters, stage=stage, block='c')

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNet20(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI1(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(1):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI3(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(3):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI7(input_shape=(64, 64, 3), xs_shape=(8,), classes=6):
    X_input = Input(input_shape)
    XS_input = Input(xs_shape)
    stage = 0;

    for i in range(7):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Concatenate()([X, XS_input])
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=[X_input, XS_input], outputs=X, name="ResNet0")

    return model


def ResNetI14(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(14):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI40(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(40):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetC1(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(1):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetC3(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(3):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetC7(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(7):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetC14(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(14):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetC40(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(40):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_64(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 64]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_128(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 128]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_256(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_512(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 512]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_1024(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 1024]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_f2(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=2, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 2, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 2, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_f4(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=4, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 4, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 4, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI5_f5(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=5, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 5, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 5, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet0")

    return model


def ResNetI7_Center(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=[X_input], outputs=X, name="ResNet0")

    return model


def ResNetI7_Good(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 3, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    output_X = []
    for oo in range(29):
        output_X.append(
            Dense(classes, name='fc' + str(classes) + str(oo), kernel_initializer=glorot_uniform(seed=0))(X))

    model = Model(inputs=[X_input], outputs=output_X, name="ResNet0")

    return model

def ResNetI7_Good_5(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f=3, filters=stage_filters, stage=stage, block='a_{}'.format(i), s=1)
        X = identity_block(X, 5, stage_filters, stage=stage, block='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    output_X = []
    for oo in range(29):
        output_X.append(
            Dense(classes, name='fc' + str(classes) + str(oo), kernel_initializer=glorot_uniform(seed=0))(X))

    model = Model(inputs=[X_input], outputs=output_X, name="ResNet0")

    return model


def ResNetI7_Good_cen(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    output_X = []
    for oo in range(23):
        output_X.append(Dense(classes, name='cfc' + str(classes)+ str(oo), kernel_initializer = glorot_uniform(seed=0))(X))

    model = Model(inputs = [X_input], outputs = output_X, name = "ResNetCen")

    return model

def ResNetI7_Good_out(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='oa_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='ob_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='oc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    output_X = []
    for oo in range(6):
        output_X.append(Dense(classes, name='ofc' + str(classes)+ str(oo), kernel_initializer = glorot_uniform(seed=0))(X))

    model = Model(inputs = [X_input], outputs = output_X, name = "ResNetOut")

    return model


def ResNetI7_Good_cen_5(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block(X, 5, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    output_X = []
    for oo in range(23):
        output_X.append(Dense(classes, name='cfc' + str(classes)+ str(oo), kernel_initializer = glorot_uniform(seed=0))(X))

    model = Model(inputs = [X_input], outputs = output_X, name = "ResNetCen")

    return model

def ResNetI7_Good_out_5(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='oa_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='ob_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='oc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    output_X = []
    for oo in range(6):
        output_X.append(Dense(classes, name='ofc' + str(classes)+ str(oo), kernel_initializer = glorot_uniform(seed=0))(X))

    model = Model(inputs = [X_input], outputs = output_X, name = "ResNetOut")

    return model

def ResNetI7_Good_depl(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model


def ResNetI7_Good_depl_MD(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_se(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(3000, name='cfc' + str(classes) + str(3), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(1000, name='cfc' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model


def ResNetI7_Good_depl_pre(input_shape = (19, 19, 8), pre_input_shape = (19, 19, 16), pre_value_shape = (1,),  classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_lk(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)

    X_input_p = Input(pre_input_shape)
    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X_p = convolution_block_se_lk(X_input_p, f = 3, filters = stage_filters, stage = stage, block='pca_{}'.format(i), s = 1)
        X_p = identity_block_se_lk(X_p, 3, stage_filters, stage=stage, block ='pcb_{}'.format(i))
        X_p = identity_block_se_lk(X_p, 3, stage_filters, stage=stage, block='pcc_{}'.format(i))
        stage = stage + 1

    X_p = Flatten()(X_p)

    X_input_v = Input(pre_value_shape)

    X = Add()([X, X_p, X_input_v])

    X = Dense(92416//416, name='cfc' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input, X_input_p, X_input_v], outputs = X, name = "ResNetCen")

    return model


def ResNetI7_Good_depl_time(input_shape = (19, 19, 9),  classes = 6):

    encoder_input = Input(input_shape, name='encoder_input')
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_lk(encoder_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)

    X = Dense(92416//416, name='cfc' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(X)
    encoder_output = Dense(512, name="encoder_output", kernel_initializer=glorot_uniform(seed=0))(X)

    encoder_output_predict = Dense(classes, name="encoder_output_predict", kernel_initializer=glorot_uniform(seed=0))(encoder_output)
    predictor_model = Model(inputs = [encoder_input], outputs = [encoder_output_predict], name = "ResNetCen")
    predictor_model.compile(optimizer=Adam(lr=1e-3), loss="mse")
    predictor_model.summary()

    initial_state, decoder_input, decoder_output, connect_decoder = get_decoder(encoder_output, 29*8, 512)
    model_train = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])
    model_train.compile(optimizer=Adam(lr=1e-3), loss="mse")
    model_train.summary()

    return predictor_model, model_train



def ResNetI7_Good_depl_time_r(input_shape = (19, 19, 9),  classes = 6):

    encoder_input = Input(input_shape, name='encoder_input')
    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_lk(encoder_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)

    state_size = 1280

    encoder_output = Dense(state_size, name="encoder_output", kernel_initializer=glorot_uniform(seed=0))(X)

    initial_state, decoder_input, decoder_output, connect_decoder = get_decoder(encoder_output, 29, state_size)
    ps_model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

    xs_initial_state_real, xs_decoder_input_real, xs_decoder_output_real, xs_connect_decoder_real = get_decoder_xs_real(encoder_output,  29, 29*8, state_size)
    xs_model_real = Model(inputs=[encoder_input, xs_decoder_input_real], outputs=[xs_decoder_output_real])

    r_initial_state_real, r_decoder_input_real, r_decoder_output_real, r_connect_decoder_real = get_decoder_r_real(encoder_output, 29*8, 1, state_size)
    bs_model_real = Model(inputs=[encoder_input, r_decoder_input_real], outputs=[r_decoder_output_real])

    ps_model.compile(optimizer=Adam(lr=1e-3), loss="mse")
    ps_model.summary()

    xs_model_real.compile(optimizer=Adam(lr=1e-3), loss="mse")
    xs_model_real.summary()

    bs_model_real.compile(optimizer=Adam(lr=1e-3), loss="mse")
    bs_model_real.summary()

    return ps_model, xs_model_real,  xs_model_real, bs_model_real, bs_model_real

    """
    
    xs_initial_state, xs_decoder_input, xs_decoder_output, xs_connect_decoder = get_decoder_xs(encoder_output, decoder_output,  29*8, state_size)
    xs_model = Model(inputs=[encoder_input, decoder_input], outputs=[xs_decoder_output])

    
    r_initial_state, r_decoder_input, r_decoder_output, r_connect_decoder = get_decoder_r(encoder_output, xs_decoder_output,  1, state_size)
    bs_model = Model(inputs=[encoder_input, decoder_input], outputs=[r_decoder_output])

    xs_model.compile(optimizer=Adam(lr=2.5e-4), loss="mse")
    xs_model.summary()

    bs_model.compile(optimizer=Adam(lr=2.5e-4), loss="mse")
    bs_model.summary()
    """


def ResNetI7_Good_depl_time_xs(input_shape = (19, 19, 9),  classes = 6):

    encoder_input = Input(input_shape, name='encoder_input')
    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_lk(encoder_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)

    state_size = 1800

    encoder_output = Dense(state_size, name="encoder_output", kernel_initializer=glorot_uniform(seed=0))(X)

    r_initial_state_real, r_decoder_input_real, r_decoder_output_real, r_connect_decoder_real = get_decoder_r_real(encoder_output, 29*8, 1, state_size)
    bs_model_real = Model(inputs=[encoder_input, r_decoder_input_real], outputs=[r_decoder_output_real])

    bs_model_real.compile(optimizer=Adam(lr=2.5e-4), loss="mse")
    bs_model_real.summary()

    return bs_model_real

def ResNetI7_Good_depl_(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_lk(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(3000, name='cfc' + str(classes) + str(3), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(1000, name='cfc' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model


def ResNetI7_Good_depl_1D(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_se(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model

def ResNetI7_Good_depl_MD_L(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_l(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se_l(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se_l(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(3000, name='cfc' + str(classes) + str(3), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(1000, name='cfc' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model

def ResNetI7_Good_Comp(input_shape = (64, 64, 3), classes = 6):

    X_First = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_se(X_First, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block_se(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block_se(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X_Second = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X_S = convolution_block_se(X_Second, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X_S = identity_block_se(X_S, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X_S = identity_block_se(X_S, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X_S = Flatten()(X_S)

    X = Add()([X, X_S])
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_First, X_Second], outputs = X, name = "ResNetCen")

    return model

def ResNetI7_Actor(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(2):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='aa_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='ab_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='ac_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, activation='relu', name='afc' + str(classes) + str(1))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    return X_input, model, X

def ResNetI7_Critic(ob_shape = (64, 64, 3),action_shape = (64, 64, 3), inclass = 6):

    ob_input = Input(ob_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(ob_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(inclass, name='ccfc' + str(inclass) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    ac_input = Input(action_shape)
    X2 = Dense(inclass, name='acfc' + str(inclass) + str(1), kernel_initializer=glorot_uniform(seed=0))(ac_input)

    merged = Add()([X, X2])

    merged_h1 = Dense(24, activation='relu')(merged)
    output = Dense(1, activation='linear')(merged_h1)

    model = Model(inputs = [ob_input, ac_input], outputs = output, name = "ResNetCen")

    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return ob_input, ac_input, model, X


def shuffle_all_data():
    dictionary = {}
    creal = 0
    call = 0
    for _ in range(0, 1):

        for j in range(3, 5):
            for i in range(0, 14):

                to_string = '/media/youndukn/lastra/depl_data_{}/data_{}_s'.format(j, i)

                file = open(to_string, 'wb')
                train_sets = []
                try:

                    a_string = '/media/youndukn/lastra/depl_data_{}/data_{}'.format(j, i)
                    print('Processing File ./full_core_data_{}/data_{}'.format(j, i))


                    if os.path.isfile(a_string):

                        file_read = open(a_string, 'rb')

                        while True:
                            depl_set = pickle.load(file_read)
                            train_set = depl_set[0]
                            state = train_set[1]
                            value = hash(state.tostring())

                            if not (value in dictionary):
                                train_sets.append(depl_set)
                                creal += 1
                                call += 1
                                dictionary[value] = 1
                            else:
                                dictionary[value] += 1
                                call += 1
                except (AttributeError, EOFError, ImportError, IndexError) as e:
                    print(e)
                    pass

                numpy.random.shuffle(train_sets)
                for trainset in train_sets:
                    pickle.dump(trainset, file, protocol=pickle.HIGHEST_PROTOCOL)

                file.close()




def shuffle_data_depl_order(include = ["y3"], exclude = ["y310, _s"], folders=["/media/youndukn/lastra/plants_data1/"]):
    dictionary = {}
    creal = 0
    call = 0
    for _ in range(0, 1):
        files = get_files_with(include, exclude, folders = folders)
        for file in files:
            train_sets = []
            try:
                while True:

                    depl_set = pickle.load(file)
                    if depl_set[0][-1] == 0:
                        pre_set = depl_set
                    else:

                        train_set = depl_set[0]
                        state = train_set[1]

                        pre_train_set = pre_set[0]
                        pre_state = pre_train_set[1]
                        value = hash(state.tostring()+pre_state.tostring())

                        now_depl = [depl_set, pre_set]

                        if not (value in dictionary):
                            train_sets.append(now_depl)
                            creal += 1
                            call += 1
                            dictionary[value] = 1
                        else:
                            dictionary[value] += 1
                            call += 1

                        pre_set = depl_set

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

            file_name = file.name
            file_name = file_name.replace('.inp', '_s.inp')
            file_dump = open(file_name, 'wb')

            numpy.random.shuffle(train_sets)
            for trainset in train_sets:
                pickle.dump(trainset, file_dump, protocol=pickle.HIGHEST_PROTOCOL)

            file_dump.close()


def shuffle_data_depl(include = ["y3"], exclude = ["y310, _s"], folders=["/media/youndukn/lastra/plants_data1/"]):
    dictionary = {}
    creal = 0
    call = 0
    for _ in range(0, 1):
        files = get_files_with(include, exclude, folders = folders)
        for file in files:
            train_sets = []
            try:
                while True:
                    depl_set = pickle.load(file)
                    train_set = depl_set[0]
                    state = train_set[1]
                    value = hash(state.tostring())

                    if not (value in dictionary):
                        train_sets.append(depl_set)
                        creal += 1
                        call += 1
                        dictionary[value] = 1
                    else:
                        dictionary[value] += 1
                        call += 1
            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

            file_name = file.name
            file_name = file_name.replace('.inp', '_s.inp')
            file_dump = open(file_name, 'wb')

            numpy.random.shuffle(train_sets)
            for trainset in train_sets:
                pickle.dump(trainset, file_dump, protocol=pickle.HIGHEST_PROTOCOL)

            file_dump.close()


def train_maximum(iter_numb, model):
    files = []
    error_plot = []
    for x_man in range(1, 2):

        for j in range(1, 2):
            for i in range(0, 14):
                to_string = './full_core_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        print("Stage{}".format(x_man))

        for _ in range(1000):
            file_read = files[1]
            train_set = pickle.load(file_read)

            # Clear gradients
            s_batch_init = train_set[0]
            y_batch_init = numpy.amax(train_set[1])

            s_batch_init = numpy.array(s_batch_init)
            s_batch_init = np.expand_dims(s_batch_init, axis=0)

            readout_t0 = model.predict(s_batch_init)[0]
            # print(y_batch_init,numpy.average(abs(y_batch_init - readout_t0)))
            print(abs(100 * (readout_t0[0] - y_batch_init) / y_batch_init), readout_t0[0], y_batch_init)
            error_plot.append(abs((readout_t0[0] - y_batch_init) / y_batch_init * 100))

        # random_range = numpy.random.shuffle((range(0, 12)))
        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                s_batch_init_temp = []

                for num_batch in range(0, 20):
                    train_set = pickle.load(file_read)
                    counter += 1
                    sys.stdout.write("\rD%i" % counter)
                    sys.stdout.flush()

                    if train_set and len(train_set) > 0:

                        s_batch_init = train_set[0]
                        y_batch_init = numpy.amax(train_set[1])
                        if train_set[0].any() and train_set[1].any():
                            y_batch_init_temp.append(y_batch_init)
                            s_batch_init_temp.append(s_batch_init)

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                y_batch_init_temp = numpy.array(y_batch_init_temp)

                model.train_on_batch(s_batch_init_temp, y_batch_init_temp)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-RES-{}-{}.hdf5".format(iter_numb, x_man))


def train_specific(iter_numb, model):
    files = []
    for x_man in range(1, 2):

        for j in range(1, 2):
            for i in range(0, 14):
                to_string = './full_core_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        print("Stage{}".format(x_man))

        # random_range = numpy.random.shuffle((range(0, 12)))
        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                s_batch_init_temp = []
                xs_batch_init_temp = []

                for num_batch in range(0, 5):
                    train_set = pickle.load(file_read)
                    counter += 1
                    sys.stdout.write("\rD%i" % counter)
                    sys.stdout.flush()

                    if train_set and len(train_set) > 0:
                        s_batch_init = train_set[0]
                        s_batch_init = numpy.array(s_batch_init)

                        for row in range(9, 19):
                            for column in range(9, 19):
                                index = (row - 9) * 10 + (column - 9)
                                values = s_batch_init[row][column]
                                values = numpy.array(values)
                                if train_set[1][index] > 0:
                                    y_batch_init_temp.append(train_set[1][index])
                                    s_batch_init_temp.append(s_batch_init)
                                    xs_batch_init_temp.append(values)

                if counter % 2000 == 0:
                    train_set = pickle.load(file_read)

                    # Clear gradients
                    s_batch_init = train_set[0]

                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    print()
                    for row in range(9, 19):
                        a_string = ""
                        for column in range(9, 19):
                            index = (row - 9) * 10 + (column - 9)
                            values = s_batch_init[0][row][column]
                            values = np.expand_dims(values, axis=0)

                            y_batch_init = train_set[1][index]

                            readout_t0 = model.predict([s_batch_init, values])[0]
                            if y_batch_init != 0:
                                value = min(99, int(abs(100 * (readout_t0[0] - y_batch_init) / y_batch_init)))
                                a_string += "{:2d} ".format(value)
                            else:
                                a_string += "{:2d} ".format(99)

                        print(a_string)

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                xs_batch_init_temp = numpy.array(xs_batch_init_temp, dtype=np.float16)

                y_batch_init_temp = numpy.array(y_batch_init_temp)
                model.train_on_batch(x=[s_batch_init_temp, xs_batch_init_temp], y=y_batch_init_temp)

            except (AttributeError, EOFError, ImportError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-RES-{}-{}.hdf5".format(iter_numb, x_man))


def train_all(iter_numb, model):
    files = []
    for x_man in range(1, 2):

        for j in range(1, 3):
            for i in range(0, 14):
                to_string = '/media/youndukn/lastra/depl_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        print("Stage{}".format(x_man))

        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                s_batch_init_temp = []

                for _ in range(29):
                    y_batch_init_temp.append([])

                for num_batch in range(0, 2):
                    depl_set = pickle.load(file_read)
                    counter += 1
                    sys.stdout.write("\rD%i" % counter)
                    sys.stdout.flush()
                    for burnup in depl_set:
                        if len(burnup) > 0:
                            s_batch_init = burnup[1]
                            s_batch_init = numpy.array(s_batch_init)

                            for index_temp in range(29):
                                y_batch_init_temp[index_temp].append(burnup[2][ad[index_temp][0]])

                            s_batch_init_temp.append(s_batch_init)

                for dummy_index in range(29):
                    y_batch_init_temp[dummy_index] = np.array(y_batch_init_temp[dummy_index])

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

                model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)

            except (AttributeError, EOFError, ImportError) as e:
                print(e, file_read.name)
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-DEPL-{}-{}.hdf5".format(iter_numb, x_man))


def train_all_sep(iter_numb, model , model2):
    files = []
    for x_man in range(1, 2):

        for j in range(1, 4):
            for i in range(0, 14):
                to_string = '/media/youndukn/lastra/depl_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        print("Stage{}".format(x_man))

        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                y_batch_init_temp_cen = []
                y_batch_init_temp_out = []
                s_batch_init_temp = []

                for _ in range(29):
                    y_batch_init_temp.append([])

                for num_batch in range(0, 2):
                    depl_set = pickle.load(file_read)
                    counter += 1
                    sys.stdout.write("\rD%i" % counter)
                    sys.stdout.flush()
                    for burnup in depl_set:
                        if len(burnup) > 0:
                            s_batch_init = burnup[1]
                            s_batch_init = numpy.array(s_batch_init)

                            for index_temp in range(29):
                                y_batch_init_temp[index_temp].append(burnup[2][ad[index_temp][0]])

                            s_batch_init_temp.append(s_batch_init)

                for dummy_index in range(29):
                    if not dummy_index in sep_out:
                        y_batch_init_temp_cen.append(np.array(y_batch_init_temp[dummy_index]))
                    else:
                        y_batch_init_temp_out.append(np.array(y_batch_init_temp[dummy_index]))


                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

                model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp_cen)
                model2.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp_out)

            except (AttributeError, EOFError, ImportError) as e:
                print(e)
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-SEP-DEPL-1-{}-{}.hdf5".format(iter_numb, x_man))
        model2.save_weights("weights-SEP-DEPL-2-{}-{}.hdf5".format(iter_numb, x_man))



def train_all_depl(iter_numb, model, include = ["y3"], exclude = ["y314"], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    for x_man in range(1, 2):

        #files = get_files(["y310", "u411", "u405"], "_s")

        files = get_files_with(include, exclude, "_s", folders)
        #files = get_files_with(["u4"],["u412", "u411", "u410"], "_s")
        for file in files:
            print(file.name)
        print("Files {}".format(len(files)))

        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                s_batch_init_temp = []


                for num_batch in range(0, 32):
                    depl_set = pickle.load(file_read)
                    counter += 1

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init_temp.append(s_batch_init)
                    y_batch_init = burnup_eoc[0][0]
                    y_batch_init_temp.append(y_batch_init)

                sys.stdout.write("\rD%i" % counter)
                sys.stdout.flush()

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

                model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)

            except (AttributeError, EOFError, ImportError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-SEP-DEPL-1-{}-{}.hdf5".format(iter_numb, x_man))


def train_all_depl_pre(iter_numb, model, include=["y3"], exclude=["y314"], folders=["/media/youndukn/lastra/plants_data/"],
                   den=False):
    for x_man in range(1, 2):

        # files = get_files(["y310", "u411", "u405"], "_s")

        files = get_files_with(include, exclude, "_s", folders)
        # files = get_files_with(["u4"],["u412", "u411", "u410"], "_s")
        for file in files:
            print(file.name)
        print("Files {}".format(len(files)))

        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                y_batch_init_pre_temp = []

                s_batch_init_temp = []
                s_batch_init_pre_temp = []

                for num_batch in range(0, 20):
                    now_pre = pickle.load(file_read)

                    counter += 1
                    depl_set = now_pre[0]
                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init_temp.append(s_batch_init)
                    y_batch_init = burnup_eoc[0][0]
                    y_batch_init_temp.append(y_batch_init)

                    depl_set = now_pre[1]
                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init_pre = burnup_boc[1]
                    s_batch_init_pre = numpy.array(s_batch_init_pre)
                    s_batch_init_pre = numpy.concatenate((s_batch_init_pre, burnup_eoc[1]), axis=2)
                    s_batch_init_pre_temp.append(s_batch_init_pre)
                    y_batch_init_pre = burnup_eoc[0][0]
                    y_batch_init_pre_temp.append(y_batch_init_pre)

                sys.stdout.write("\rD%i" % counter)
                sys.stdout.flush()

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
                s_batch_init_pre_temp = numpy.array(s_batch_init_pre_temp, dtype=np.float16)
                y_batch_init_pre_temp = numpy.array(y_batch_init_pre_temp, dtype=np.float16)

                model.train_on_batch(x=[s_batch_init_temp, s_batch_init_pre_temp, y_batch_init_pre_temp], y=y_batch_init_temp)

            except (AttributeError, EOFError, ImportError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-SEP-DEPL-1-{}-{}.hdf5".format(iter_numb, x_man))



def train_all_depl_time(iter_numb, pre_model, train_model, include=["y3"], exclude=["y314"], folders=["/media/youndukn/lastra/plants_data/"],
                   den=False):

    file_write = open("/media/youndukn/lastra/shuffled/shuffled.data", 'wb')

    file_write_boc = open("/media/youndukn/lastra/shuffled/boc.data", 'wb')

    file_write_den = open("/media/youndukn/lastra/shuffled/shuffled_den.data", 'wb')

    files = get_files_with(include, exclude, "_s", folders)

    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    while len(files) > 0:

        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]


            all_batch = []

            for num_batch in range(0, 100):


                s_batch = []
                s_batch_den = []

                xs_batch = []
                bs_batch = []
                depl_set = pickle.load(file_read)

                counter += 1
                burnup_eoc = depl_set[-1]
                y_batch_init = burnup_eoc[0][0]

                for burnup in depl_set:

                    s_batch_init = burnup[1]
                    s_batch_init = numpy.array(s_batch_init)

                    xs_output = []
                    for xs_types in range(8):
                        for mapping in ad:
                            xs_output.append(s_batch_init[mapping[1]][mapping[2]][xs_types])

                    xs_output = numpy.array(xs_output)
                    xs_batch.append(xs_output)
                    bs_batch.append(burnup[0][0])

                    b_batch_init = numpy.zeros((19,19,1))
                    b_batch_init.fill(burnup[0][0])
                    s_batch_init = numpy.concatenate((s_batch_init, b_batch_init), axis = 2)

                    if den == True:
                        s_batch_init_den = burnup[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init_den = numpy.concatenate((s_batch_init_den, s_batch_init), axis=2)

                    s_batch.append(s_batch_init)
                    s_batch_den.append(s_batch_init_den)

                for s_index, state in enumerate(s_batch):
                    my_burnup = bs_batch[s_index]
                    bs_s = bs_batch[s_index:]
                    xs_s = xs_batch[s_index:]
                    xs_left = 20-len(xs_s)
                    if xs_left >= 0:
                        bs_l = bs_s[-1]
                        xs_l = xs_s[-1]
                        for _ in range(xs_left):
                            bs_s.append(bs_l)
                            xs_s.append(xs_l)

                        all_batch.append((state, y_batch_init-my_burnup, bs_s, xs_s, s_batch_den[s_index]))

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

            numpy.random.shuffle(all_batch)

            for i in range(len(all_batch)//20-1):

                s_batch_init_temp = []
                y_batch_init_temp = []
                b_batch_input_temp = []
                b_batch_output_temp = []
                s_batch_init_den_temp = []

                for batch in all_batch[i * 20:(i + 1) * 20]:
                    s_batch_init_temp.append(batch[0])
                    y_batch_init_temp.append(batch[1])
                    b_batch_input_temp.append(batch[2])
                    b_batch_output_temp.append(batch[3])
                    s_batch_init_den_temp.append(batch[4])

                if len(s_batch_init_temp) == 20:

                    s_batch_init_temp = numpy.array(s_batch_init_temp)
                    y_batch_init_temp = numpy.array(y_batch_init_temp)
                    b_batch_input_temp = numpy.array(b_batch_input_temp)
                    b_batch_input_temp = b_batch_input_temp.reshape(20, 20, 1)
                    b_batch_output_temp = numpy.array(b_batch_output_temp)
                    s_batch_init_den_temp = numpy.array(s_batch_init_den_temp)

                    pickle.dump((s_batch_init_temp, y_batch_init_temp, b_batch_input_temp, b_batch_output_temp), file_write,
                            protocol=pickle.HIGHEST_PROTOCOL)

                    pickle.dump((s_batch_init_den_temp, y_batch_init_temp, b_batch_input_temp, b_batch_output_temp),
                                file_write_den,
                                protocol=pickle.HIGHEST_PROTOCOL)

                    for lol_index, s_batch_init in enumerate(s_batch_init_temp):
                        if s_batch_init[0][0][8]==0:
                            pickle.dump((s_batch_init, y_batch_init_temp[lol_index]),
                                        file_write_boc,
                                        protocol=pickle.HIGHEST_PROTOCOL)


                    pre_x_data = \
                        {
                            'encoder_input': s_batch_init_den_temp
                        }

                    pre_y_data = \
                        {
                            'encoder_output_predict': y_batch_init_temp
                        }

                    train_x_data = \
                        {
                            'encoder_input': s_batch_init_den_temp,
                            'decoder_input': b_batch_input_temp
                        }

                    train_y_data = \
                        {
                            'decoder_output': b_batch_output_temp
                        }

                    pre_model.train_on_batch(x=pre_x_data, y= pre_y_data)
                    train_model.train_on_batch(x=train_x_data, y=train_y_data)

        except (EOFError) as e:
            files.remove(file_read)
            file_read.close()
            print(e)
            pass

    file_write.close()
    file_write_den.close()
    file_write_boc.close()

def prepare_all_depl_time_r(iter_numb,  ps_model,xs_model,bs_model, include=["y3"], exclude=["y314"], folders=["/media/youndukn/lastra/plants_data/"],
                   den=False):
    batch_size = 32
    files = get_files_with(include, exclude, "_s", folders)

    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    while len(files) > 0:

        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 300):
                ps_batch = []
                bs_batch = []
                dp_batch = []
                depl_set = pickle.load(file_read)

                counter += 1

                pre_burnup_step = 0
                for bs_index, burnup in enumerate(depl_set):

                    if (burnup[0][0] - pre_burnup_step) > 2200:
                        ps_output = []
                        for mapping in ad:
                            ps_output.append(burnup[2][mapping[0]])

                        ps_output = numpy.array(ps_output)

                        dp_output = []
                        for xs_types in range(8):
                            for mapping in ad:
                                dp_output.append(burnup[1][mapping[1]][mapping[2]][xs_types]*dp_norm[xs_types])

                        dp_output = numpy.array(dp_output)

                        bs_batch.append([burnup[0][0], ])
                        ps_batch.append(ps_output)
                        dp_batch.append(dp_output)

                    if bs_index > 3:
                        pre_burnup_step = burnup[0][0]

                burnup_eoc = depl_set[-1]
                y_batch_init = burnup_eoc[0][0]

                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[1]

                if den == True:
                    s_batch_init_den = burnup_boc[4]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)

                if (len(dp_batch)>=6):
                    xs_batch_input_temp_q[6].append(numpy.array(s_batch_init))
                    bs_batch_input_temp_q[6].append(numpy.array(bs_batch[:6]))
                    ps_batch_output_temp_q[6].append(numpy.array(ps_batch[:6]))
                    dp_batch_output_temp_q[6].append(numpy.array(dp_batch[:6]))
                    bs_batch_output_temp_q[6].append(numpy.array([y_batch_init, ]))
                else:
                    print(file_read.name, bs_batch[:6], y_batch_init)
            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

        except (EOFError) as e:
            files.remove(file_read)
            file_read.close()
            print(e)
            pass


def train_all_depl_time_r(ps_model,xs_model_real,xs_model,bs_model_real,bs_model):
    indexes = range(len(xs_batch_input_temp_q))

    indexes = numpy.array(indexes)
    numpy.random.shuffle(indexes)

    for queue_index in indexes:
        if len(xs_batch_input_temp_q[queue_index])> 100:
            print("Queue : ",queue_index, len(xs_batch_input_temp_q[queue_index]))

            ps_x_data = \
                {
                    'encoder_input': numpy.array(xs_batch_input_temp_q[queue_index]),
                    'decoder_input': numpy.array(bs_batch_input_temp_q[queue_index])
                }

            ps_y_data = \
                {
                    'decoder_output': numpy.array(ps_batch_output_temp_q[queue_index])
                }

            xs_x_data_real = \
                {
                    'encoder_input': numpy.array(xs_batch_input_temp_q[queue_index]),
                    'xs_decoder_input_real': numpy.array(ps_batch_output_temp_q[queue_index])
                }

            xs_y_data_real = \
                {
                    'xs_decoder_output_real': numpy.array(dp_batch_output_temp_q[queue_index])
                }

            bs_x_data_real = \
                {
                    'encoder_input': numpy.array(xs_batch_input_temp_q[queue_index]),
                    'r_decoder_input_real': numpy.array(dp_batch_output_temp_q[queue_index])
                }

            bs_y_data_real = \
                {
                    'r_decoder_output_real': numpy.array(bs_batch_output_temp_q[queue_index])
                }

            xs_x_data = \
                {
                    'encoder_input': numpy.array(xs_batch_input_temp_q[queue_index]),
                    'decoder_input': numpy.array(bs_batch_input_temp_q[queue_index])
                }

            xs_y_data = \
                {
                    'xs_decoder_output': numpy.array(dp_batch_output_temp_q[queue_index])
                }

            bs_x_data = \
                {
                    'encoder_input': numpy.array(xs_batch_input_temp_q[queue_index]),
                    'decoder_input': numpy.array(bs_batch_input_temp_q[queue_index])
                }

            bs_y_data = \
                {
                    'r_decoder_output': numpy.array(bs_batch_output_temp_q[queue_index])
                }

            validation_split = len(xs_batch_input_temp_q[queue_index])*0.02


            ps_model.fit(x=ps_x_data,
                            y=ps_y_data,
                            batch_size=32,
                            epochs=1,
                            validation_split=validation_split)

            xs_model_real.fit(x=xs_x_data_real,
                         y=xs_y_data_real,
                         batch_size=32,
                         epochs=1,
                         validation_split=validation_split)

            bs_model_real.fit(x=bs_x_data_real,
                         y=bs_y_data_real,
                         batch_size=32,
                         epochs=1,
                         validation_split=validation_split)
            """
            xs_model.fit(x=xs_x_data,
                            y=xs_y_data,
                            batch_size=10,
                            epochs=1,
                            validation_split=validation_split)

            bs_model.fit(x=bs_x_data,
                            y=bs_y_data,
                            batch_size=10,
                            epochs=1,
                            validation_split=validation_split)
            """



def train_all_depl_time_r_xs(bs_model_real):
    indexes = range(len(xs_batch_input_temp_q))

    indexes = numpy.array(indexes)
    numpy.random.shuffle(indexes)

    for queue_index in indexes:
        if len(xs_batch_input_temp_q[queue_index])> 100:
            print("Queue : ",queue_index, len(xs_batch_input_temp_q[queue_index]))


            bs_x_data_real = \
                {
                    'encoder_input': numpy.array(xs_batch_input_temp_q[queue_index]),
                    'r_decoder_input_real': numpy.array(dp_batch_output_temp_q[queue_index])
                }

            bs_y_data_real = \
                {
                    'r_decoder_output_real': numpy.array(bs_batch_output_temp_q[queue_index])
                }


            validation_split = len(xs_batch_input_temp_q[queue_index])*0.02


            bs_model_real.fit(x=bs_x_data_real,
                         y=bs_y_data_real,
                         batch_size=32,
                         epochs=1,
                         validation_split=validation_split)


def train_all_depl_time_pre_shuf(iter_numb, pre_model, train_model, include=["y3"], exclude=["y314"], folders=["/media/youndukn/lastra/plants_data/"],
                   den=False):

    file_write = open("/media/youndukn/lastra/shuffled/shuffled_den.data", 'rb')
    counter = 0
    the_end = True
    while the_end:
        try:
            counter += 1
            s_batch_init_temp, y_batch_init_temp, b_batch_input_temp, b_batch_output_temp = pickle.load(file_write)
            sys.stdout.write("\r3D%i" % counter)
            sys.stdout.flush()

            pre_x_data = \
                {
                    'encoder_input': s_batch_init_temp
                }

            pre_y_data = \
                {
                    'encoder_output_predict': y_batch_init_temp
                }

            train_x_data = \
                {
                    'encoder_input': s_batch_init_temp,
                    'decoder_input': b_batch_input_temp
                }

            train_y_data = \
                {
                    'decoder_output': b_batch_output_temp
                }

            pre_model.train_on_batch(x=pre_x_data, y= pre_y_data)
            train_model.train_on_batch(x=train_x_data, y=train_y_data)
        except (AttributeError, EOFError, ImportError) as e:
            the_end = False
            print("End Of File")
            pass

    file_write.close()



def train_all_depl_time_pre_shuf_den(iter_numb, pre_model, train_model, include=["y3"], exclude=["y314"], folders=["/media/youndukn/lastra/plants_data/"],
                   den=False):

    file_write = open("/media/youndukn/lastra/shuffled/shuffled_den.data", 'rb')
    counter = 0
    the_end = True
    while the_end:
        try:
            counter += 1
            s_batch_init_temp, y_batch_init_temp, b_batch_input_temp, b_batch_output_temp = pickle.load(file_write)
            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

            pre_x_data = \
                {
                    'encoder_input': s_batch_init_temp
                }

            pre_y_data = \
                {
                    'encoder_output_predict': y_batch_init_temp
                }

            train_x_data = \
                {
                    'encoder_input': s_batch_init_temp,
                    'decoder_input': b_batch_input_temp
                }

            train_y_data = \
                {
                    'decoder_output': b_batch_output_temp
                }

            pre_model.train_on_batch(x=pre_x_data, y= pre_y_data)
            #train_model.train_on_batch(x=train_x_data, y=train_y_data)
        except (AttributeError, EOFError, ImportError) as e:
            the_end = False
            print("End Of File")
            pass

    file_write.close()

def train_all_depl_random(iter_numb, model, include = ["y3"], exclude = ["y314"], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    for x_man in range(1, 2):

        #files = get_files(["y310", "u411", "u405"], "_s")

        files = get_files_with(include, exclude, "_s", folders)
        #files = get_files_with(["u4"],["u412", "u411", "u410"], "_s")
        for file in files:
            print(file.name)
        print("Files {}".format(len(files)))

        sy_batch = []
        counter = 0
        batch_counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                for num_batch in range(0, 32):
                    depl_set = pickle.load(file_read)
                    counter += 1

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    y_batch_init = burnup_eoc[0][0]
                    sy_batch.append([s_batch_init, y_batch_init])

                batch_counter += 1
                sys.stdout.write("\rD%i" % counter)
                sys.stdout.flush()

                if batch_counter%10 == 0:
                    sy_batch_np = numpy.array(sy_batch)
                    numpy.random.shuffle(sy_batch_np)

                    s_batch_init_temp = []
                    y_batch_init_temp = []

                    for sy in sy_batch_np:

                        s_batch_init_temp.append(sy[0])
                        y_batch_init_temp.append(sy[1])

                        if len(s_batch_init_temp)%32 == 0:
                            s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                            y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

                            model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)

                            s_batch_init_temp = []
                            y_batch_init_temp = []

                    sy_batch = []


            except (AttributeError, EOFError, ImportError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-SEP-DEPL-1-{}-{}.hdf5".format(iter_numb, x_man))

def train_all_depl_comp(iter_numb, model, include = ["y3"], exclude = ["y314"], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)
        for file in files:
            print(file.name)
        print("Files {}".format(len(files)))

        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                sys.stdout.write("\rD%i" % counter)
                sys.stdout.flush()

                y_batch_init_temp = []
                s_batch_init_temp = []

                for num_batch in range(0, 20):
                    depl_set = pickle.load(file_read)
                    counter += 1

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init_temp.append(s_batch_init)
                    y_batch_init = burnup_eoc[0][0]
                    y_batch_init_temp.append(y_batch_init)

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

                y_batch_init_temp_1 = []
                s_batch_init_temp_1 = []

                for num_batch in range(0, 20):
                    depl_set = pickle.load(file_read)
                    counter += 1

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init_temp_1.append(s_batch_init)
                    y_batch_init = burnup_eoc[0][0]
                    y_batch_init_temp_1.append(y_batch_init)

                s_batch_init_temp_1 = numpy.array(s_batch_init_temp_1, dtype=np.float16)
                y_batch_init_temp_1 = numpy.array(y_batch_init_temp_1, dtype=np.float16)

                if len(s_batch_init_temp_1) == 20 and len(s_batch_init_temp) == 20:
                    y_batch_init_temp = y_batch_init_temp-y_batch_init_temp_1

                    model.train_on_batch(x=[s_batch_init_temp,s_batch_init_temp_1], y=y_batch_init_temp)

            except (AttributeError, EOFError, ImportError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-SEP-DEPL-1-{}-{}.hdf5".format(iter_numb, x_man))


def train_all_depl_dual(iter_numb, ref_model, train_model, max_numb,  den=False):

    for x_man in range(1, 2):

        files = get_files_with(["u405"])

        counter = 0
        try:

            file_index = random.randrange(len(files))
            file_read = files[0]

            y_batch_init_temp = []
            s_batch_init_temp = []

            for num_batch in range(0, max_numb):
                depl_set = pickle.load(file_read)
                counter += 1

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = burnup_boc[1]
                s_batch_init = numpy.array(s_batch_init)
                if den == True:
                    s_batch_init_den = burnup_boc[4]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init_temp.append(s_batch_init)

                s_batch_init_1 = np.expand_dims(s_batch_init, axis=0)
                readout_t0 = ref_model.predict(s_batch_init_1)[0]

                y_batch_init = burnup_eoc[0][0]
                y_batch_init_temp.append(readout_t0[0] - y_batch_init)


            s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
            y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

            train_model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)
            break
        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass



def train_all_fxy(iter_numb, model, den=False):
    files = []
    for x_man in range(1, 2):

        files = get_files(["y310", "u411", "u405"], "_s")

        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                s_batch_init_temp = []


                for num_batch in range(0, 20):
                    depl_set = pickle.load(file_read)
                    counter += 1

                    burnup_boc = depl_set[0]

                    #find max fxy
                    y_batch_init = 0

                    for burnup_point in depl_set:
                        if y_batch_init<burnup_point[0][5]:
                            y_batch_init = burnup_point[0][5]

                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init_temp.append(s_batch_init)

                    y_batch_init_temp.append(y_batch_init)

                sys.stdout.write("\rD%i" % counter)
                sys.stdout.flush()

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

                model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)

            except (AttributeError, EOFError, ImportError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-SEP-DEPL-1-{}-{}.hdf5".format(iter_numb, x_man))


def train_all_sep_1_65(iter_numb, model , model2):
    files = []
    for x_man in range(1, 10):

        to_string = '/media/youndukn/lastra/depl_data_fxy_1.65_train'

        file_read = open(to_string, 'rb')
        files.append(file_read)


        print("Stage{}".format(x_man))

        counter = 0
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                y_batch_init_temp_cen = []
                y_batch_init_temp_out = []
                s_batch_init_temp = []

                for _ in range(29):
                    y_batch_init_temp.append([])

                for num_batch in range(0, 2):
                    depl_set = pickle.load(file_read)
                    counter += 1
                    sys.stdout.write("\rD%i" % counter)
                    sys.stdout.flush()
                    for burnup in depl_set:
                        if len(burnup) > 0:
                            s_batch_init = burnup[1]
                            s_batch_init = numpy.array(s_batch_init)

                            for index_temp in range(29):
                                y_batch_init_temp[index_temp].append(burnup[2][ad[index_temp][0]])

                            s_batch_init_temp.append(s_batch_init)

                for dummy_index in range(29):
                    if not dummy_index in sep_out:
                        y_batch_init_temp_cen.append(np.array(y_batch_init_temp[dummy_index]))
                    else:
                        y_batch_init_temp_out.append(np.array(y_batch_init_temp[dummy_index]))


                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

                model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp_cen)
                model2.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp_out)

            except (AttributeError, EOFError, ImportError) as e:
                print(e)
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-SEP-DEPL-1-{}-{}.hdf5".format(iter_numb, x_man))
        model2.save_weights("weights-SEP-DEPL-2-{}-{}.hdf5".format(iter_numb, x_man))


def print_weights(index, model):
    table = []
    files = []
    for x_man in range(1, 2):

        for j in range(1, 7):
            for i in range(0, 14):
                to_string = './full_core_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        try:
            model.load_weights("./weights-RES-{}-{}.hdf5".format(index, x_man))
            average = 0
            for _ in range(100):
                file_read = files[1]
                train_set = pickle.load(file_read)

                # Clear gradients
                s_batch_init = train_set[0]
                y_batch_init = numpy.amax(train_set[1])

                s_batch_init = numpy.array(s_batch_init)
                s_batch_init = np.expand_dims(s_batch_init, axis=0)

                readout_t0 = model.predict(s_batch_init)[0]
                # print(y_batch_init,numpy.average(abs(y_batch_init - readout_t0)))
                # print(abs(100 * (readout_t0[0] - y_batch_init) / y_batch_init))
                print(abs(100 * (readout_t0[0] - y_batch_init) / y_batch_init))
                average += abs(100 * (readout_t0[0] - y_batch_init) / y_batch_init)
            print("Model {} Stage {}  Average : {}".format(index, x_man, average / 100))
        except:
            pass


def print_all_weights(index, model):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    files = []
    table = []

    for x_man in range(1, 2):

        for j in range(1, 2):
            for i in range(0, 14):
                to_string = '/media/youndukn/lastra/depl_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        try:
            average = 0

            avg_index = np.zeros(29)
            counter_strike = 0
            for indexed_image in range(200):
                file_read = files[1]
                depl_set = pickle.load(file_read)

                # Clear gradients
                for burnup in depl_set:
                    counter_strike += 1
                    s_batch_init = burnup[1]

                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)

                    readout_t0 = model.predict([s_batch_init])

                    matrix_ = []
                    matrix_e = []

                    max_value = 0
                    max_index = 0

                    avg_error = 0

                    min_value = 100
                    min_index = 0

                    max_value_e = 0
                    max_index_e = 0

                    for row in range(9, 17):
                        array_ = []
                        array_e = []
                        for column in range(9, 17):
                            index = (row - 9) * 10 + (column - 9)
                            output_index = -1
                            for index_wow, value in enumerate(ad):
                                if value[0] == index:
                                    output_index = index_wow
                                    break

                            if output_index >= 0:
                                error = 100 * (burnup[2][index] - readout_t0[output_index][0][0]) / burnup[2][index]
                                if abs(error) > max_value_e:
                                    max_value_e = abs(error)
                                    max_index_e = output_index

                                pow_v = burnup[2][index]

                                if abs(pow_v) < min_value:
                                    min_value = abs(pow_v)
                                    min_index = output_index

                                if abs(pow_v) > max_value:
                                    max_value = abs(pow_v)
                                    max_index = output_index

                                array_e.append(abs(error))
                                avg_error += abs(error)
                                avg_index[output_index] = avg_index[output_index] + abs(error)
                                array_.append(pow_v)
                            else:
                                array_.append(0)
                                array_e.append(0)
                        matrix_.append(array_)
                        matrix_e.append(array_e)

                    distance = math.sqrt(math.pow(abs(ad[max_index][1] - ad[min_index][1]), 2) +
                                         math.pow(abs(ad[max_index][2] - ad[min_index][2]), 2))

                    avg_error /= 29

                    """
                    dist_org = np.array(matrix_)
                    error_mat = np.array(matrix_e)
    
                    fig = plt.figure(figsize=(10,10))
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)
    
                    df = pd.DataFrame(dist_org)
                    dfe = pd.DataFrame(error_mat)
                    sns.heatmap(df, annot=True, annot_kws={"size": 7}, ax=ax1)
                    sns.heatmap(dfe, annot=True, annot_kws={"size": 7}, ax=ax2)
    
                    fig.savefig('dist_img/{}_dist.png'.format(indexed_image))
                    plt.close()
                    """
                    Distribution.create(maxError=max_value_e,
                                        maxErrorIndex=max_index_e,
                                        maxPower=max_value,
                                        maxPowerIndex=max_index,
                                        avgError=avg_error,
                                        minPower=min_value,
                                        minPowerIndex=min_index,
                                        minMaxDistance=distance,
                                        file_path='dist_img/{}_dist.png'.format(indexed_image))
                    sys.stdout.write("\rIndexed%i" % indexed_image)
                    sys.stdout.flush()
            avg_index = avg_index / counter_strike
            print(avg_index)

        except (AttributeError, EOFError, ImportError, IndexError) as e:
            print(e)
            pass


def print_all_weights_sep(index, model, model2):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    files = []
    table = []
    for x_man in range(1, 2):

        for j in range(1, 2):
            for i in range(0, 14):
                to_string = '/media/youndukn/lastra/depl_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        try:
            average = 0

            avg_index = np.zeros(29)

            counter_strike = 0
            for indexed_image in range(100):
                file_index = random.randrange(len(files))

                file_read = files[1]
                depl_set = pickle.load(file_read)

                # Clear gradients
                for burnup in depl_set:
                    counter_strike += 1
                    s_batch_init = burnup[1]

                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)

                    readout_t0 = model.predict([s_batch_init])
                    readout_t1 = model2.predict([s_batch_init])

                    matrix_ = []
                    matrix_e = []

                    max_value = 0
                    max_index = 0

                    avg_error = 0

                    min_value = 100
                    min_index = 0

                    max_value_e = 0
                    max_index_e = 0

                    for row in range(9, 17):
                        array_ = []
                        array_e = []
                        for column in range(9, 17):
                            index = (row - 9) * 10 + (column - 9)
                            output_index = -1
                            for index_wow, value in enumerate(ad):
                                if value[0] == index:
                                    output_index = index_wow
                                    break

                            if output_index >= 0:
                                if not output_index in sep_out:
                                    error = 100 * (burnup[2][index] - readout_t0[sep_cen.index(output_index)][0][0]) / burnup[2][index]
                                else:
                                    error = 100 * (burnup[2][index] - readout_t1[sep_out.index(output_index)][0][0]) / \
                                            burnup[2][index]

                                if abs(error) > max_value_e:
                                    max_value_e = abs(error)
                                    max_index_e = output_index

                                pow_v = burnup[2][index]

                                if abs(pow_v) < min_value:
                                    min_value = abs(pow_v)
                                    min_index = output_index

                                if abs(pow_v) > max_value:
                                    max_value = abs(pow_v)
                                    max_index = output_index

                                array_e.append(abs(error))
                                avg_error += abs(error)
                                avg_index[output_index] = avg_index[output_index] + abs(error)
                                array_.append(pow_v)
                            else:
                                array_.append(0)
                                array_e.append(0)
                        matrix_.append(array_)
                        matrix_e.append(array_e)

                    distance = math.sqrt(math.pow(abs(ad[max_index][1] - ad[min_index][1]), 2) +
                                         math.pow(abs(ad[max_index][2] - ad[min_index][2]), 2))

                    avg_error /= 29


                    dist_org = np.array(matrix_)
                    error_mat = np.array(matrix_e)



                    fig = plt.figure(figsize=(10,10))
                    fig.suptitle("Power Dist/Error BS{}, FXY{}".format(burnup[0][1], burnup[0][5]), fontsize=20)

                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                    df = pd.DataFrame(dist_org)
                    dfe = pd.DataFrame(error_mat)
                    sns.heatmap(df, annot=True, annot_kws={"size": 7}, ax=ax1, vmax=2.0)
                    sns.heatmap(dfe, annot=True, annot_kws={"size": 7}, ax=ax2, vmax=2.0)

                    fig.savefig('dist_img/{}_dist_{}_{}.png'.format(counter_strike, burnup[0][1], burnup[0][5]))
                    plt.close()

                    Distribution.create(maxError=max_value_e,
                                        maxErrorIndex=max_index_e,
                                        maxPower=max_value,
                                        maxPowerIndex=max_index,
                                        avgError=avg_error,
                                        minPower=min_value,
                                        minPowerIndex=min_index,
                                        minMaxDistance=distance,
                                        file_path='dist_img/{}_dist_{}_{}.png'.format(counter_strike, burnup[0][1], burnup[0][5]),
                                        burnup_step = burnup[0][1],
                                        fxy = burnup[0][5])
                    sys.stdout.write("\rIndexed%i" % counter_strike)
                    sys.stdout.flush()

            avg_index = avg_index / counter_strike
            print(avg_index)
            print(counter_strike)

        except (AttributeError, EOFError, ImportError, IndexError) as e:
            print(e)
            pass

def print_all_weights_sep_1_65(index, model, model2):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    table = []
    files = []
    for x_man in range(1, 2):

        to_string = '/media/youndukn/lastra/depl_data_fxy_1.65_test'

        file_read = open(to_string, 'rb')
        files.append(file_read)

        try:
            average = 0

            avg_index = np.zeros(29)

            counter_strike = 0
            for indexed_image in range(10):

                file_read = files[0]
                depl_set = pickle.load(file_read)

                # Clear gradients
                for burnup in depl_set:
                    counter_strike += 1
                    s_batch_init = burnup[1]

                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)

                    readout_t0 = model.predict([s_batch_init])
                    readout_t1 = model2.predict([s_batch_init])

                    matrix_ = []
                    matrix_e = []

                    max_value = 0
                    max_index = 0

                    avg_error = 0

                    min_value = 100
                    min_index = 0

                    max_value_e = 0
                    max_index_e = 0

                    for row in range(9, 17):
                        array_ = []
                        array_e = []
                        for column in range(9, 17):
                            index = (row - 9) * 10 + (column - 9)
                            output_index = -1
                            for index_wow, value in enumerate(ad):
                                if value[0] == index:
                                    output_index = index_wow
                                    break

                            if output_index >= 0:
                                if not output_index in sep_out:
                                    error = 100 * (burnup[2][index] - readout_t0[sep_cen.index(output_index)][0][0]) / burnup[2][index]
                                else:
                                    error = 100 * (burnup[2][index] - readout_t1[sep_out.index(output_index)][0][0]) / \
                                            burnup[2][index]

                                if abs(error) > max_value_e:
                                    max_value_e = abs(error)
                                    max_index_e = output_index

                                pow_v = burnup[2][index]

                                if abs(pow_v) < min_value:
                                    min_value = abs(pow_v)
                                    min_index = output_index

                                if abs(pow_v) > max_value:
                                    max_value = abs(pow_v)
                                    max_index = output_index

                                array_e.append(abs(error))
                                avg_error += abs(error)
                                avg_index[output_index] = avg_index[output_index] + abs(error)
                                array_.append(pow_v)
                            else:
                                array_.append(0)
                                array_e.append(0)
                        matrix_.append(array_)
                        matrix_e.append(array_e)

                    distance = math.sqrt(math.pow(abs(ad[max_index][1] - ad[min_index][1]), 2) +
                                         math.pow(abs(ad[max_index][2] - ad[min_index][2]), 2))

                    avg_error /= 29
                    """

                    dist_org = np.array(matrix_)
                    error_mat = np.array(matrix_e)



                    fig = plt.figure(figsize=(10,10))
                    fig.suptitle("Power Dist/Error BS{}, FXY{}".format(burnup[0][1], burnup[0][5]), fontsize=20)

                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                    df = pd.DataFrame(dist_org)
                    dfe = pd.DataFrame(error_mat)
                    sns.heatmap(df, annot=True, annot_kws={"size": 7}, ax=ax1, vmax=2.0)
                    sns.heatmap(dfe, annot=True, annot_kws={"size": 7}, ax=ax2, vmax=2.0)

                    fig.savefig('dist_img/{}_dist_{}_{}.png'.format(counter_strike, burnup[0][1], burnup[0][5]))
                    plt.close()
                    """
                    Distribution.create(maxError=max_value_e,
                                        maxErrorIndex=max_index_e,
                                        maxPower=max_value,
                                        maxPowerIndex=max_index,
                                        avgError=avg_error,
                                        minPower=min_value,
                                        minPowerIndex=min_index,
                                        minMaxDistance=distance,
                                        file_path='dist_img/{}_dist_{}_{}.png'.format(counter_strike, burnup[0][1], burnup[0][5]),
                                        burnup_step = burnup[0][1],
                                        fxy = burnup[0][5])
                    sys.stdout.write("\rIndexed%i" % counter_strike)
                    sys.stdout.flush()

            avg_index = avg_index / counter_strike
            print(avg_index)
            print(counter_strike)

        except (AttributeError, EOFError, ImportError, IndexError) as e:
            print(e)
            pass

def shuffle_all_weights_sep():
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    files = []
    table = []
    to_string = '/media/youndukn/lastra/depl_data_fxy_1.65'
    dump_file = open(to_string, 'wb')
    for x_man in range(1, 2):

        for j in range(1, 5):
            for i in range(0, 14):
                to_string = '/media/youndukn/lastra/depl_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        counter=0
        counter_strike = 0
        while len(files) > 0:
            try:
                file_index = random.randrange(len(files))

                file_read = files[file_index]
                depl_set = pickle.load(file_read)
                max_fxy = 0.0
                # Clear gradients
                for burnup in depl_set:

                    if burnup[0][5] >max_fxy:
                        max_fxy = burnup[0][5]

                counter_strike +=1
                if max_fxy < 1.65:
                    pickle.dump(depl_set, dump_file, protocol=pickle.HIGHEST_PROTOCOL)
                    counter+=1

                sys.stdout.write("\rIndexed%i/ %i" % (counter, counter_strike))
                sys.stdout.flush()

            except (AttributeError, EOFError, ImportError) as e:
                #print(e)
                files.remove(file_read)
                file_read.close()
                pass


def print_all_weights_depl(index, model, include = ["y314"], exclude = [], folders = ["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(200):

                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)[0]
                    y_batch_init = burnup_eoc[0][0]
                    print(y_batch_init, readout_t0[0],(readout_t0[0] - y_batch_init), 100 * (readout_t0[0] - y_batch_init) / y_batch_init)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_all_weights_depl_pre(index, model, include=["y314"], exclude=[], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(200):

                    now_pre = pickle.load(file_read)

                    depl_set = now_pre[0]
                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    y_batch_init = burnup_eoc[0][0]

                    depl_set = now_pre[1]
                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init_pre = burnup_boc[1]
                    s_batch_init_pre = numpy.array(s_batch_init_pre)
                    s_batch_init_pre = numpy.concatenate((s_batch_init_pre, burnup_eoc[1]), axis=2)
                    y_batch_init_pre = [burnup_eoc[0][0],]

                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    s_batch_init_pre = np.expand_dims(s_batch_init_pre, axis=0)
                    y_batch_init_pre = np.expand_dims(y_batch_init_pre, axis=0)

                    readout_t0 = model.predict([s_batch_init, s_batch_init_pre, y_batch_init_pre])[0]

                    print(y_batch_init, readout_t0[0], (readout_t0[0] - y_batch_init),
                          100 * (readout_t0[0] - y_batch_init) / y_batch_init)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_all_weights_depl_time(index, model, include=["y314"], exclude=[], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(200):

                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    y_batch_init = burnup_eoc[0][0]

                    b_batch_init = numpy.zeros((19, 19, 1))
                    b_batch_init.fill(burnup_boc[0][0])
                    s_batch_init = numpy.concatenate((s_batch_init, b_batch_init), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)

                    readout_t0 = model.predict([s_batch_init])[0]

                    print("| Normal {:5.0f}".format(y_batch_init), "| Pred {:5.0f}".format(readout_t0[0]),
                          "| Diff {:3.0f}".format((readout_t0[0] - y_batch_init)),
                          "| Perc {:3.2f}".format(100 * (readout_t0[0] - y_batch_init) / y_batch_init))

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_all_weights_depl_time_r(index, ps_model, xs_model, bs_model, include=["y314"], exclude=[], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(20):
                    is_predictable = True
                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    xs_batch_init = burnup_boc[1]
                    bs_batch_init = burnup_eoc[0][0]

                    bs_batch = []
                    ps_batch = []
                    dp_batch = []

                    pre_burnup_step = 0

                    for bs_index, burnup in enumerate(depl_set):
                        if (burnup[0][0] - pre_burnup_step) > 2200:
                            ps_output = []
                            for mapping in ad:
                                if ps_output == None:
                                    is_predictable = False
                                ps_output.append(burnup[2][mapping[0]])

                            dp_output = []
                            for xs_types in range(8):
                                for mapping in ad:
                                    dp_output.append(burnup[1][mapping[1]][mapping[2]][xs_types])

                            dp_output = numpy.array(dp_output)

                            ps_output = numpy.array(ps_output)
                            if burnup[0][0]== None:
                                is_predictable = False
                            bs_batch.append(burnup[0][0])
                            ps_batch.append(ps_output)
                            dp_batch.append(dp_output)

                        if bs_index > 3:
                            pre_burnup_step = burnup[0][0]
                    bs_batch_expect = bs_batch
                    bs_batch_expect.append(bs_batch_init)

                    xs_batch_init = numpy.array(xs_batch_init)
                    xs_batch_init = np.expand_dims(xs_batch_init, axis=0)
                    bs_batch = numpy.array(bs_batch)
                    bs_length = len(bs_batch)
                    bs_batch = bs_batch.reshape(1, bs_length, 1)

                    ps_batch = numpy.array(ps_batch)
                    ps_length = len(ps_batch)
                    ps_batch = ps_batch.reshape(1, ps_length, 29)

                    burnup_eoc = depl_set[-1]

                    dp_output = []
                    for xs_types in range(8):
                        for mapping in ad:
                            dp_output.append(burnup_eoc[1][mapping[1]][mapping[2]][xs_types])

                    dp_batch.append(numpy.array(dp_output))

                    dp_batch = numpy.array(dp_batch)
                    dp_length = len(dp_batch)
                    dp_batch = dp_batch.reshape(1, dp_length, 29*8)

                    if is_predictable :
                        ps_batches = ps_model.predict([xs_batch_init, bs_batch])[0]
                        xs_batches = xs_model.predict([xs_batch_init, bs_batch])[0]
                        bs_batches = bs_model.predict([xs_batch_init, bs_batch])[0]

                        print(ps_batches[0][:5])
                        print(ps_batch[0][0][:5])

                        print(xs_batches[0][:5])
                        print(dp_batch[0][1][:5])

                        print("| Normal {:5.0f}".format(bs_batch_expect[-1]),
                              "| Prep {:5.0f}".format(bs_batches[0])
                              )

                        print()

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_all_weights_depl_time_r_real(index, ps_model, xs_model_real, bs_model_real, include=["y314"], exclude=[], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(20):
                    is_predictable = True
                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    xs_batch_init = burnup_boc[1]
                    bs_batch_init = burnup_eoc[0][0]

                    bs_batch = []
                    ps_batch = []
                    dp_batch = []

                    pre_burnup_step = 0

                    for bs_index, burnup in enumerate(depl_set):
                        if (burnup[0][0] - pre_burnup_step) > 2200:
                            ps_output = []
                            for mapping in ad:
                                if ps_output == None:
                                    is_predictable = False
                                ps_output.append(burnup[2][mapping[0]])

                            dp_output = []
                            for xs_types in range(8):
                                for mapping in ad:
                                    dp_output.append(burnup[1][mapping[1]][mapping[2]][xs_types]*dp_norm[xs_types])

                            dp_output = numpy.array(dp_output)

                            ps_output = numpy.array(ps_output)
                            if burnup[0][0]== None:
                                is_predictable = False
                            bs_batch.append(burnup[0][0])
                            ps_batch.append(ps_output)
                            dp_batch.append(dp_output)

                        if bs_index > 3:
                            pre_burnup_step = burnup[0][0]
                    bs_batch_expect = bs_batch
                    bs_batch_expect.append(bs_batch_init)

                    xs_batch_init = numpy.array(xs_batch_init)

                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        xs_batch_init = numpy.concatenate((xs_batch_init, s_batch_init_den), axis=2)

                    xs_batch_init = np.expand_dims(xs_batch_init, axis=0)

                    if len(bs_batch)> 6:
                        bs_batch = numpy.array(bs_batch[:6])
                        bs_batch = bs_batch.reshape(1, 6, 1)

                        ps_batch = numpy.array(ps_batch[:6])
                        ps_batch = ps_batch.reshape(1, 6, 29)

                        dp_batch = numpy.array(dp_batch[:6])
                        dp_batch = dp_batch.reshape(1, 6, 29*8)

                        if is_predictable :
                            ps_batches = ps_model.predict([xs_batch_init, bs_batch])
                            xs_batches = xs_model_real.predict([xs_batch_init, ps_batches])
                            bs_batches = bs_model_real.predict([xs_batch_init, xs_batches])

                            xs_batches_1 = xs_model_real.predict([xs_batch_init, ps_batch])
                            bs_batches_1 = bs_model_real.predict([xs_batch_init, dp_batch])

                            numpy.set_printoptions(precision=4, suppress=True)

                            print(ps_batches[0][0][:5])
                            print(ps_batch[0][0][:5])

                            print(xs_batches[0][0][:5])
                            print(xs_batches_1[0][0][:5])
                            print(dp_batch[0][1][:5])

                            print("| Normal {:5.0f}".format(bs_batch_init),
                                  "| Prep {:5.0f}".format(bs_batches[0][0]),
                                  "| Prep {:5.0f}".format(bs_batches_1[0][0])
                                  )

                            print()

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass



def print_all_weights_depl_time_r_real_xs(index, bs_model_real, include=["y314"], exclude=[], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()

    numpy.set_printoptions(precision=4, suppress=True)

    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(200):
                    is_predictable = True
                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    xs_batch_init = burnup_boc[1]
                    bs_batch_init = burnup_eoc[0][0]

                    bs_batch = []
                    ps_batch = []
                    dp_batch = []

                    pre_burnup_step = 0

                    for bs_index, burnup in enumerate(depl_set):
                        if (burnup[0][0] - pre_burnup_step) > 2200:
                            ps_output = []
                            for mapping in ad:
                                if ps_output == None:
                                    is_predictable = False
                                ps_output.append(burnup[2][mapping[0]])

                            dp_output = []
                            for xs_types in range(8):
                                for mapping in ad:
                                    dp_output.append(burnup[1][mapping[1]][mapping[2]][xs_types]*dp_norm[xs_types])

                            dp_output = numpy.array(dp_output)

                            ps_output = numpy.array(ps_output)
                            if burnup[0][0]== None:
                                is_predictable = False
                            bs_batch.append(burnup[0][0])
                            ps_batch.append(ps_output)
                            dp_batch.append(dp_output)

                        if bs_index > 3:
                            pre_burnup_step = burnup[0][0]
                    bs_batch_expect = bs_batch
                    bs_batch_expect.append(bs_batch_init)

                    xs_batch_init = numpy.array(xs_batch_init)

                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        xs_batch_init = numpy.concatenate((xs_batch_init, s_batch_init_den), axis=2)

                    xs_batch_init = np.expand_dims(xs_batch_init, axis=0)
                    bs_batch = numpy.array(bs_batch)
                    bs_length = len(bs_batch)
                    bs_batch = bs_batch.reshape(1, bs_length, 1)

                    ps_batch = numpy.array(ps_batch)
                    ps_length = len(ps_batch)
                    ps_batch = ps_batch.reshape(1, ps_length, 29)

                    burnup_eoc = depl_set[-1]

                    dp_output = []

                    dp_batch = numpy.array(dp_batch[:6])
                    dp_length = len(dp_batch)
                    dp_batch = dp_batch.reshape(1, dp_length, 29*8)

                    if is_predictable :
                        bs_batches_1 = bs_model_real.predict([xs_batch_init, dp_batch])

                        print("| Normal {:5.0f}".format(bs_batch_expect[-1]),
                              "| Prep {:5.0f}".format(bs_batches_1[0][0])
                              )

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_all_weights_depl_time_den(index, model, include=["y314"], exclude=[], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(200):

                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    y_batch_init = burnup_eoc[0][0]

                    b_batch_init = numpy.zeros((19, 19, 1))
                    b_batch_init.fill(burnup_boc[0][0])
                    s_batch_init = numpy.concatenate((s_batch_init, b_batch_init), axis=2)
                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init_den, s_batch_init), axis=2)

                    s_batch_init = np.expand_dims(s_batch_init, axis=0)



                    readout_t0 = model.predict([s_batch_init])[0]

                    print("| Normal {:5.0f}".format(y_batch_init), "| Pred {:5.0f}".format(readout_t0[0]),
                          "| Diff {:3.0f}".format((readout_t0[0] - y_batch_init)),
                          "| Perc {:3.2f}".format(100 * (readout_t0[0] - y_batch_init) / y_batch_init))

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_all_weights_comp(index, model, include = ["y314"], exclude = [], folders = ["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                for indexed_image in range(200):

                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    y_batch_init = burnup_eoc[0][0]

                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init_1 = burnup_boc[1]
                    s_batch_init_1 = numpy.array(s_batch_init_1)
                    if den == True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init_1 = numpy.concatenate((s_batch_init_1, s_batch_init_den), axis=2)
                    s_batch_init_1 = np.expand_dims(s_batch_init_1, axis=0)
                    y_batch_init_1 = burnup_eoc[0][0]

                    readout_t0 = model.predict([s_batch_init, s_batch_init_1])[0]
                    print(y_batch_init-y_batch_init_1, readout_t0[0])

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_all_weights_depl_full(index, model, include = [], exclude = [], folders = ["/media/youndukn/lastra/plants_data/"], den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):
        averages = []
        files = get_files_with(include, exclude, "_s", folders)
        files.sort(key=lambda x: x.name)
        for file_read in files:
            try:
                average = 0
                dev_values = []
                for indexed_image in range(200):
                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[1]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)[0]
                    y_batch_init = burnup_eoc[0][0]
                    #print(y_batch_init, readout_t0[0],(readout_t0[0] - y_batch_init), 100 * (readout_t0[0] - y_batch_init) / y_batch_init)
                    average += (y_batch_init - readout_t0[0])
                    dev_values.append((y_batch_init - readout_t0[0]))

                average /= 200
                standard = 0
                for value in dev_values:
                    standard += (value-average)**2
                standard = (standard/199)**.5
                print(file_read.name, average, standard)
                averages.append((file_read.name,average, standard))
            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_all_weights_depl_dual(index, ref_model, train_model, den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(["u411"],[])

        try:
            average = 0

            avg_index = np.zeros(29)
            counter_strike = 0
            for indexed_image in range(100):
                file_read = files[1]
                depl_set = pickle.load(file_read)

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = burnup_boc[1]
                s_batch_init = numpy.array(s_batch_init)
                if den==True:
                    s_batch_init_den = burnup_boc[4]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init = np.expand_dims(s_batch_init, axis=0)
                readout_t0 = ref_model.predict(s_batch_init)[0]
                readout_t1 = train_model.predict(s_batch_init)[0]
                y_batch_init = burnup_eoc[0][0]
                print(y_batch_init, readout_t0[0], readout_t1[0])

        except (AttributeError, EOFError, ImportError, IndexError) as e:
            print(e)
            pass

def print_all_weights_fxy(index, model, den=False):

    files = []
    table = []
    print()
    for x_man in range(1, 2):

        files = get_files_with(["u411"])

        try:
            average = 0

            avg_index = np.zeros(29)
            counter_strike = 0
            for indexed_image in range(200):
                file_read = files[1]
                depl_set = pickle.load(file_read)

                burnup_boc = depl_set[0]

                # find max fxy
                y_batch_init = 0

                for burnup_point in depl_set:
                    if y_batch_init < burnup_point[0][5]:
                        y_batch_init = burnup_point[0][5]

                s_batch_init = burnup_boc[1]
                s_batch_init = numpy.array(s_batch_init)
                if den==True:
                    s_batch_init_den = burnup_boc[4]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init = np.expand_dims(s_batch_init, axis=0)
                readout_t0 = model.predict(s_batch_init)[0]

                print(readout_t0[0],(readout_t0[0] - y_batch_init), 100 * (readout_t0[0] - y_batch_init) / y_batch_init)

        except (AttributeError, EOFError, ImportError, IndexError) as e:
            print(e)
            pass

def sep_depl():

    files = []
    table = []
    to_string = '/media/youndukn/lastra/depl_data_fxy_1.65_train'
    dump_file_train = open(to_string, 'wb')
    to_string = '/media/youndukn/lastra/depl_data_fxy_1.65_test'
    dump_file_test = open(to_string, 'wb')
    for x_man in range(1, 2):

        to_string = '/media/youndukn/lastra/depl_data_fxy_1.65'

        file_read = open(to_string, 'rb')
        files.append(file_read)

        counter_strike = 0
        while len(files) > 0:
            try:
                file_index = random.randrange(len(files))

                file_read = files[file_index]
                depl_set = pickle.load(file_read)

                counter_strike +=1
                if counter_strike < 100:
                    pickle.dump(depl_set, dump_file_test, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(depl_set, dump_file_train, protocol=pickle.HIGHEST_PROTOCOL)

                sys.stdout.write("\rIndexed%i" % ( counter_strike))
                sys.stdout.flush()

            except (AttributeError, EOFError, ImportError) as e:
                #print(e)
                files.remove(file_read)
                file_read.close()
                pass


if __name__ == "__main__":

    """
    for i in range(5):
        if i == 0:
            model = ResNetI1((19, 19, 8), classes = 1)
        elif i == 1:
            model = ResNetI3((19, 19, 8), classes = 1)
        elif i == 2:
            model = ResNetI7((19, 19, 8), classes = 1)
        elif i == 3:
            model = ResNetI14((19, 19, 8), classes = 1)
        elif i == 4:
            model = ResNetI40((19, 19, 8), classes = 1)
    
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
    
        #model.load_weights("weights-RES50.hdf5")
    
        error_plot = []
    
        train_set = []
    
        files = []
    
        #shuffle_all_data()
    
        #train_maximum(i)
    
        print_weights(i)
    
        model.save_weights("weights-RES.hdf5")
    """
    """
    for i in range(4):
        if i == 0:
            model = FullyConnected_Small_3((19, 19, 8), classes = 1)
        elif i == 1:
            model = FullyConnected_Small_6((19, 19, 8), classes = 1)
        elif i == 2:
            model = FullyConnected_Small_9((19, 19, 8), classes = 1)
        elif i == 3:
            model = FullyConnected_Small_3_3((19, 19, 8), classes = 1)
    
    
        model.compile(loss='mse', optimizer=Adam(lr=0.1))
    
        #model.load_weights("weights-RES50.hdf5")
    
        error_plot = []
    
        train_set = []
    
        files = []
    
        #shuffle_all_data()
    
        train_maximum(i)
    
        print_weights(i)
    
        model.save_weights("weights-RES.hdf5")
    """

    """
    model2 = ResNetI7_Good_out((19, 19, 8), classes = 1)
    
    model = ResNetI7_Good((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.004))
    #model.load_weights("weights-DEP_train_3_2-8_1.hdf5")
    error_plot = []
    
    train_set = []
    
    files = []
    
    train_all(4, model)
    print_all_weights(0, model)
    
    model.save_weights("weights-DEP_train_3_2-8_1004.hdf5")
    
    
    model2 = ResNetI7_Good_out((19, 19, 8), classes = 1)
    
    model = ResNetI7_Good((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.load_weights("weights-DEP_train_3_2-8_1.hdf5")
    error_plot = []
    
    train_set = []
    
    files = []
    
    train_all(4, model)
    print_all_weights(0, model)
    train_all(5, model)
    print_all_weights(0, model)
    
    model.save_weights("weights-DEP_train_3_2-8_1.hdf5")
    
    
    model = ResNetI7_Good_5((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.load_weights("weights-DEP_train_5_2-8_1.hdf5")
    error_plot = []
    
    train_set = []
    
    files = []
    
    train_all(10, model)
    print_all_weights(0, model)
    train_all(11, model)
    print_all_weights(0, model)
    
    model.save_weights("weights-DEP_train_5_2-8_1.hdf5")
    """
    """
    model = ResNetI7_Good_cen((19, 19, 8), classes = 1)
    model2 = ResNetI7_Good_out((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model2.compile(loss='mse', optimizer=Adam(lr=0.001))
    #shuffle_all_data()
    #shuffle_all_data()
    
    model.load_weights("weights-SEP-DEPL-1-3-1.hdf5")
    
    model2.load_weights("weights-SEP-DEPL-2-3-1.hdf5")
    
    error_plot = []
    
    train_set = []
    
    files = []
    
    print_all_weights_sep(0, model, model2)
    
    
    
    
    model = ResNetI7_Good_cen_5((19, 19, 8), classes = 1)
    model2 = ResNetI7_Good_out_5((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model2.compile(loss='mse', optimizer=Adam(lr=0.001))
    #shuffle_all_data()
    #shuffle_all_data()
    
    model.load_weights("weights-SEP-DEPL-1-7-1.hdf5")
    model2.load_weights("weights-SEP-DEPL-2-7-1.hdf5")
    
    error_plot = []
    
    train_set = []
    
    files = []
    
    print_all_weights_sep(0, model, model2)
    """

    """
    model = ResNetI7_Good_cen((19, 19, 8), classes = 1)
    model2 = ResNetI7_Good_out((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model2.compile(loss='mse', optimizer=Adam(lr=0.001))
    #shuffle_all_data()
    #shuffle_all_data()
    
    model.load_weights("weights-DEP_train_sep_3_2-8_1.hdf5")
    model2.load_weights("weights-DEP_train_sep_3_2-8_2.hdf5")
    
    
    error_plot = []
    
    train_set = []
    
    files = []
    
    train_all_sep(20, model, model2)
    train_all_sep(21, model, model2)
    train_all_sep(22, model, model2)
    train_all_sep(23, model, model2)
    
    try:
        print_all_weights_sep(0, model, model2)
    except:
        pass
    model.save_weights("weights-DEP_train_sep_3_2-8_1.hdf5")
    model2.save_weights("weights-DEP_train_sep_3_2-8_2.hdf5")
    
    """
    """
    model = ResNetI7_Good_cen_5((19, 19, 8), classes = 1)
    model2 = ResNetI7_Good_out_5((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model2.compile(loss='mse', optimizer=Adam(lr=0.001))
    #shuffle_all_data()
    #shuffle_all_data()
    
    #model.load_weights("0_weights-RES-3-2-1.hdf5")
    model.load_weights("weights-SEP-DEPL-1-27-1.hdf5")
    model2.load_weights("weights-SEP-DEPL-2-27-1.hdf5")
    
    error_plot = []
    
    train_set = []
    
    files = []
    
    train_all_sep(24, model, model2)
    train_all_sep(25, model, model2)
    train_all_sep(26, model, model2)
    train_all_sep(27, model, model2)
    
    try:
        print_all_weights_sep(5, model, model2)
    except:
        pass
    
    model.save_weights("weights-DEP_train_sep_5_2-8_1.hdf5")
    model2.save_weights("weights-DEP_train_sep_5_2-8_2.hdf5")
    """
    #shuffle_all_data()
    #shuffle_all_weights_sep()
    #sep_depl()
    """
    model = ResNetI7_Good_cen_5((19, 19, 8), classes = 1)
    model2 = ResNetI7_Good_out_5((19, 19, 8), classes = 1)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model2.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.load_weights("weights-DEP_train_sep_5_2-8_1_1_65.hdf5")
    model2.load_weights("weights-DEP_train_sep_5_2-8_2_1_65.hdf5")
    
    error_plot = []
    
    train_set = []
    
    files = []
    
    train_all_sep_1_65(32, model, model2)
    print_all_weights_sep_1_65(5, model, model2)
    train_all_sep_1_65(33, model, model2)
    print_all_weights_sep_1_65(5, model, model2)
    train_all_sep_1_65(34, model, model2)
    print_all_weights_sep_1_65(5, model, model2)
    train_all_sep_1_65(35, model, model2)
    print_all_weights_sep_1_65(5, model, model2)
    
    
    model.save_weights("weights-DEP_train_sep_5_2-8_1_1_65.hdf5")
    model2.save_weights("weights-DEP_train_sep_5_2-8_2_1_65.hdf5")
    """
    #depl cycle length calculation
    """
    pre_weights = False
    pre_shuffle = False
    bs_model_real = ResNetI7_Good_depl_time_xs((19, 19, 28), classes=1)

    if pre_weights:
        bs_model_real.load_weights("01_u4_y3_y311_ny314_dp_real_only_den.hdf5")

    if pre_shuffle:
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data3/"], )
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data4/"], )
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data5/"], )

    print_all_weights_depl_time_r_real_xs(0, bs_model_real,
                                          ["y08"],
                                          ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                                          ["/media/youndukn/lastra/plants_data3/"],
                                          True)
    prepare_all_depl_time_r(0, bs_model_real, bs_model_real, bs_model_real,
                            [],
                            ["y307"],
                            ["/media/youndukn/lastra/plants_data3/",
                             "/media/youndukn/lastra/plants_data4/",
                             "/media/youndukn/lastra/plants_data5/"],
                            True)

    for ix in range(132, 300):
        train_all_depl_time_r_xs(bs_model_real)

        bs_model_real.save_weights("01_u4_y3_y311_ny314_dp_real_only_den.hdf5")

        print_all_weights_depl_time_r_real_xs(0, bs_model_real,
                                              ["y305", "y306", "y308"],
                                              ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                                              ["/media/youndukn/lastra/plants_data3/"],
                                              True)
        print_all_weights_depl_time_r_real_xs(0, bs_model_real,
                                              ["y310", "yg311"],
                                              ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6", "data_7",
                                               "data_8", "data_9", "data_10", "data_11", "data_12"],
                                              ["/media/youndukn/lastra/plants_data4/"],
                                              True)
    """


    """
    pre_weights = False
    pre_shuffle = False
    bs_model_real = ResNetI7_Good_depl_time_xs((19, 19, 8), classes=1)

    if pre_weights:
        bs_model_real.load_weights("01_u4_y3_y311_ny314_dp_real_only.hdf5")

    if pre_shuffle:
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data3/"], )
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data4/"], )
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data5/"], )

    print_all_weights_depl_time_r_real_xs(0, bs_model_real,
                                       ["y307"],
                                       ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                                       ["/media/youndukn/lastra/plants_data3/"],
                                       True)
    prepare_all_depl_time_r(0, bs_model_real, bs_model_real, bs_model_real,
                            [],
                            [],
                            ["/media/youndukn/lastra/plants_data3/",
                             "/media/youndukn/lastra/plants_data4/",
                             "/media/youndukn/lastra/plants_data5/"],
                            True)

    for ix in range(132, 300):
        train_all_depl_time_r_xs(bs_model_real)

        bs_model_real.save_weights("01_u4_y3_y311_ny314_dp_real_only.hdf5")

        print_all_weights_depl_time_r_real_xs(0, bs_model_real,
                                           ["y307"],
                                           ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                                           ["/media/youndukn/lastra/plants_data3/"],
                                           True)
        print_all_weights_depl_time_r_real_xs(0, bs_model_real,
                                           ["y310"],
                                           ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6", "data_7",
                                            "data_8", "data_9", "data_10", "data_11", "data_12"],
                                           ["/media/youndukn/lastra/plants_data4/"],
                                           True)
    """

    pre_weights = False
    pre_shuffle = False
    ps_model, xs_model_real, xs_model, bs_model_real, bs_model  = ResNetI7_Good_depl_time_r((19, 19, 28), classes=1)


    if pre_weights:
        ps_model.load_weights("01_u4_y3_y311_ny314_ps_big.hdf5")
        xs_model_real.load_weights("01_u4_y3_y311_ny314_xs_real_big.hdf5")
        bs_model_real.load_weights("01_u4_y3_y311_ny314_dp_real_big.hdf5")
        xs_model.load_weights("01_u4_y3_y311_ny314_xs_big.hdf5")
        bs_model.load_weights("01_u4_y3_y311_ny314_dp_big.hdf5")

    if pre_shuffle:
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data3/"], )
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data4/"], )
        shuffle_data_depl([],
                          ["_s"],
                          ["/media/youndukn/lastra/plants_data5/"], )


    print_all_weights_depl_time_r_real(0, ps_model,xs_model_real,bs_model_real,
                                    ["y308"],
                                    ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                                    ["/media/youndukn/lastra/plants_data3/"],
                                    True)
    prepare_all_depl_time_r(0, ps_model,xs_model,bs_model,
                                         [],
                                         [],
                                         ["/media/youndukn/lastra/plants_data3/",
                                          "/media/youndukn/lastra/plants_data4/",
                                          "/media/youndukn/lastra/plants_data5/"],
                                         True)


    for ix in range(132, 300):
        train_all_depl_time_r(ps_model, xs_model_real, xs_model, bs_model_real, bs_model)

        ps_model.save_weights("01_u4_y3_y311_ny314_ps_big.hdf5")
        xs_model_real.save_weights("01_u4_y3_y311_ny314_xs_real_big.hdf5")
        bs_model_real.save_weights("01_u4_y3_y311_ny314_dp_real_big.hdf5")
        xs_model.save_weights("01_u4_y3_y311_ny314_xs_big.hdf5")
        bs_model.save_weights("01_u4_y3_y311_ny314_dp_big.hdf5")

        print_all_weights_depl_time_r_real(0, ps_model,xs_model_real,bs_model_real,
                                        ["y308"],
                                        ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                                        ["/media/youndukn/lastra/plants_data3/"],
                                        True)
        print_all_weights_depl_time_r_real(0, ps_model,xs_model_real,bs_model_real,
                                        ["y310"],
                                        ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6", "data_7",
                                         "data_8", "data_9", "data_10", "data_11", "data_12"],
                                        ["/media/youndukn/lastra/plants_data4/"],
                                        True)



    """
    pre_weights = False
    pre_shuffle = False
    pre_data = True
    pre_model, train_model = ResNetI7_Good_depl_time((19, 19, 29), classes=1)

    if pre_weights:
        pre_model.load_weights("01_u4_y3_y311_ny314_pre_den.hdf5")
        train_model.load_weights("01_u4_y3_y311_ny314_rnn_den.hdf5")

    if pre_shuffle:
        shuffle_data_depl([],
                                  ["y308", "_s"],
                                  ["/media/youndukn/lastra/plants_data3/"],)
        shuffle_data_depl([],
                                  ["_s"],
                                  ["/media/youndukn/lastra/plants_data4/"],)

    if pre_data:
        train_all_depl_time(0, pre_model, train_model,
                              [],
                              ["y308"],
                              ["/media/youndukn/lastra/plants_data3/",
                               "/media/youndukn/lastra/plants_data4/"],
                              True)

    print_all_weights_depl_time_den(0, pre_model,
                               ["y307"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data3/"],
                               True)

    for ix in range(132, 180):
        train_all_depl_time_pre_shuf_den(ix, pre_model, train_model,
                              [],
                              ["y308"],
                              ["/media/youndukn/lastra/plants_data3/",
                               "/media/youndukn/lastra/plants_data4/"],
                              True)
        print_all_weights_depl_time_den(0, pre_model,
                               ["y307"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data3/"],
                               True)
        print_all_weights_depl_time_den(0, pre_model,
                               ["y310"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6","data_7", "data_8", "data_9", "data_10", "data_11", "data_12"],
                               ["/media/youndukn/lastra/plants_data4/"],
                               True)

        pre_model.save_weights("01_u4_y3_y311_ny314_pre_den.hdf5")
        train_model.save_weights("01_u4_y3_y311_ny314_rnn_den.hdf5")
        
        
    """
    """
    pre_weights = False
    pre_shuffle = False
    pre_data = False
    pre_model, train_model = ResNetI7_Good_depl_time((19, 19, 9), classes=1)

    if pre_weights:
        pre_model.load_weights("01_u4_y3_y311_ny314_pre.hdf5")
        train_model.load_weights("01_u4_y3_y311_ny314_rnn.hdf5")

    if pre_shuffle:
        shuffle_data_depl([],
                                  ["y308", "_s"],
                                  ["/media/youndukn/lastra/plants_data3/"],)
        shuffle_data_depl([],
                                  ["_s"],
                                  ["/media/youndukn/lastra/plants_data4/"],)

    print_all_weights_depl_time(0, pre_model,
                               ["y307"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data3/"],
                               True)
    if pre_data:
        train_all_depl_time(0, pre_model, train_model,
                              [],
                              ["y308"],
                              ["/media/youndukn/lastra/plants_data3/",
                               "/media/youndukn/lastra/plants_data4/"],
                              True)

    print_all_weights_depl_time(0, pre_model,
                               ["y307"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data3/"],
                               True)

    for ix in range(132, 145):
        train_all_depl_time_pre_shuf(ix, pre_model, train_model,
                              [],
                              ["y308"],
                              ["/media/youndukn/lastra/plants_data3/",
                               "/media/youndukn/lastra/plants_data4/"],
                              True)
        print_all_weights_depl_time(0, pre_model,
                               ["y307"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data3/"],
                               True)
        print_all_weights_depl_time(0, pre_model,
                               ["y310"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6","data_7", "data_8", "data_9", "data_10", "data_11", "data_12"],
                               ["/media/youndukn/lastra/plants_data4/"],
                               True)

        pre_model.save_weights("01_u4_y3_y311_ny314_pre.hdf5")
        train_model.save_weights("01_u4_y3_y311_ny314_rnn.hdf5")
    """
    """
    model = ResNetI7_Good_depl_MD_lk((19, 19, 28), classes=1)
    # shuffle_data_depl()

    model.compile(loss='mse', optimizer=Adam(lr=0.00025))
    
    for ix in range(132, 153):
        train_all_depl(ix, model,
                              [],
                              ["y314"],
                              ["/media/youndukn/lastra/depl_1/",
                               "/media/youndukn/lastra/depl_2/",
                               "/media/youndukn/lastra/depl_3/",
                               "/media/youndukn/lastra/depl_4/",
                               "/media/youndukn/lastra/plants_data/"],
                              True)
        print_all_weights_depl(0, model,
                               ["y310", "y314"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data/"],
                               True)

    model.save_weights("01_u4_y3_y311_ny314_se_MD_lk.hdf5")
    """
    """
    model = ResNetI7_Good_depl_MD_lk((19, 19, 28), classes = 1)
    #shuffle_data_depl()

    model.compile(loss='mse', optimizer=Adam(lr=0.00025))
    
    model.load_weights("01_u4_y3_y311_ny314_se_MD_lk_ra.hdf5")

    for ix in range(132,143):
        train_all_depl_random(ix, model,
                       [],
                       ["y314"],
                       ["/media/youndukn/lastra/depl_1/",
                        "/media/youndukn/lastra/depl_2/",
                        "/media/youndukn/lastra/depl_3/",
                        "/media/youndukn/lastra/depl_4/",
                        "/media/youndukn/lastra/plants_data/"],
                       True)
        print_all_weights_depl(0, model,
                               ["y310", "y314"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data/"],
                               True)

    model.save_weights("01_u4_y3_y311_ny314_se_MD_lk_ra_30.hdf5")
    """
    """
    model = ResNetI7_Good_depl_MD_L((19, 19, 28), classes = 1)
    #shuffle_data_depl()

    model.compile(loss='mse', optimizer=Adam(lr=0.00025))


    for ix in range(132,153):
        train_all_depl(ix, model,
                       [],
                       ["y314"],
                       ["/media/youndukn/lastra/depl_1/",
                    "/media/youndukn/lastra/depl_2/",
                    "/media/youndukn/lastra/depl_3/",
                    "/media/youndukn/lastra/depl_4/",
                        "/media/youndukn/lastra/plants_data/"],
                       True)
        print_all_weights_depl(0, model,
                               ["y310", "y314"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data/"],
                               True)

    model.save_weights("01_u4_y3_y311_ny314_se_md_l.hdf5")
    """
    """
    #depl cycle length calculation
    with tf.device('/device:GPU:0'):

        model = ResNetI7_Good_Comp((19, 19, 28), classes = 1)
        #shuffle_data_depl()

        model.compile(loss='mse', optimizer=Adam(lr=0.001))


        for ix in range(132,143):
            train_all_depl_comp(ix, model,
                           [],
                           ["y314"],
                           [
                            "/media/youndukn/lastra/plants_data/"],
                           True)
            print_all_weights_comp(0, model,
                                   ["y310", "y314"],
                                   ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                                   ["/media/youndukn/lastra/plants_data/"],
                                   True)

        model.save_weights("01_u4_y3_y311_ny314_comp.hdf5")
    """
    """

    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    #shuffle_data_depl()

    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    for ix in range(132,143):
        train_all_depl(ix, model,
                       [],
                       ["y314"],
                       ["/media/youndukn/lastra/depl_1/",
                        "/media/youndukn/lastra/depl_2/",
                        "/media/youndukn/lastra/depl_3/",
                        "/media/youndukn/lastra/depl_4/",
                        "/media/youndukn/lastra/plants_data/"],
                       True)
        print_all_weights_depl(0, model,
                               ["y310", "y314"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data/"],
                               True)

    model.save_weights("01_u4_y3_y311_ny314.hdf5")
    """
    """
    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    #shuffle_data_depl()

    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    for ix in range(132,143):
        train_all_depl(ix, model,
                       [],
                       ["y314"],
                       [
                        "/media/youndukn/lastra/plants_data/"],
                       True)
        print_all_weights_depl(0, model,
                               ["y310", "y314"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data/"],
                               True)

    model.save_weights("01_u4_y3_ny314.hdf5")


    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    #shuffle_data_depl()

    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    for ix in range(132,137):
        train_all_depl(ix, model,
                       ["y3"],
                       ["y314"],
                       [
                        "/media/youndukn/lastra/plants_data/"],
                       True)
        print_all_weights_depl(0, model,
                               ["y310", "y314"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data/"],
                               True)

    model.save_weights("01_y3_y311.hdf5")
    """

    """
    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    #shuffle_data_depl()
    
    #model.load_weights("depl_ref_model_only_y311_a_lot.hdf5")
    #model.load_weights("depl_ref_model_u4_y3_y311.hdf5")
    model.load_weights("01_u4_y3_y311_ny314.hdf5")
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    #["y304", "y306", "y307", "y308", "y309", "y310", "y311", "y312", "y313", "y314"]
    for ix in range(132,133):
        print_all_weights_depl(0, model,
                               ["y314"],
                               ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                               ["/media/youndukn/lastra/plants_data/"],
                               True)
    
    #depl cycle
    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    
    model.load_weights("depl_ref_model_only_y311_a_lot.hdf5")
    #model.load_weights("depl_ref_model_u4_y3_y311.hdf5")
    #model.load_weights("depl_ref_model_y3_not_y314.hdf5")
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    
    print_all_weights_depl_full(0, model, ["y3"], ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"], den = True)
    """

    """
    #shuffle_data_depl()
    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    
    model.load_weights("depl_weights-DEP_train_3_2-8_1004_yes_cl.hdf5")
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    print_all_weights_depl(0, model, True)
    for ix in range(144,155):
        train_all_depl(ix, model, True)
        print_all_weights_depl(0, model, True)
    
    model.save_weights("depl_weights-DEP_train_3_2-8_1004_yes_cl.hdf5")
    """
    """
    #depl cycle length calculation
    model = ResNetI7_Good_depl((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.load_weights("depl_weights-DEP_train_3_2-8_1004_no_cl.hdf5")
    print_all_weights_depl(0, model, False)
    for ix in range(132,143):
        train_all_depl(ix, model, False)
        print_all_weights_depl(0, model, False)
    
    model.save_weights("depl_weights-DEP_train_3_2-8_1004_no_cl.hdf5")
    
    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.load_weights("depl_weights-DEP_train_3_2-8_1004_yes_cl.hdf5")
    print_all_weights_depl(0, model, True)
    for ix in range(144,155):
        train_all_depl(ix, model, True)
        print_all_weights_depl(0, model, True)
    
    model.save_weights("depl_weights-DEP_train_3_2-8_1004_yes_cl.hdf5")
    """


    #fxy calculation
    """
    model = ResNetI7_Good_depl((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    print_all_weights_fxy(0, model, False)
    for ix in range(132,143):
        train_all_fxy(ix, model, False)
        print_all_weights_fxy(0, model, False)
    
    model.save_weights("depl_weights-DEP_train_3_2-8_1004_no.hdf5")
    """
    """
    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    
    model.load_weights("depl_weights-DEP_train_3_2-8_1004_yes.hdf5")
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    print_all_weights_fxy(0, model, True)
    for ix in range(144,155):
        train_all_fxy(ix, model, True)
        print_all_weights_fxy(0, model, True)
    
    model.save_weights("depl_weights-DEP_train_3_2-8_1004_yes.hdf5")
    """



    """
    #loading fxy model to retrain
    model = ResNetI7_Good_depl((19, 19, 8), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.load_weights("depl_weights-DEP_train_3_2-8_1004_no.hdf5")
    print_all_weights_fxy(0, model, False)
    for ix in range(156,160):
        train_all_fxy(ix, model, False)
        print_all_weights_fxy(0, model, False)
    
    model.save_weights("depl_weights-DEP_train_3_2-8_1004_no_fxy.hdf5")
    
    model = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.load_weights("depl_weights-DEP_train_3_2-8_1004_yes.hdf5")
    print_all_weights_fxy(0, model, True)
    for ix in range(161,165):
        train_all_fxy(ix, model, True)
        print_all_weights_fxy(0, model, True)
    
    model.save_weights("depl_weights-DEP_train_3_2-8_1004_yes_fxy.hdf5")
    
    """


    #Dual Prediction
    """
    model_1 = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    model_2 = ResNetI7_Good_depl((19, 19, 28), classes = 1)
    
    
    model_1.compile(loss='mse', optimizer=Adam(lr=0.00025))
    model_2.compile(loss='mse', optimizer=Adam(lr=0.00025))
    model_1.load_weights("depl_weights-DEP_train_3_2-8_1004_yes_cl.hdf5")
    
    print_all_weights_depl(0, model_1, model_2, True)
    
    for jx in range(200, 50, -10):
        print("Stage JX {}".format(jx))
        model_2.load_weights("depl_weights-DEP_train_3_2-8_1004_yes_cl.hdf5")
        for ix in range(0,1000):
            sys.stdout.write("\rD%i" % ix)
            sys.stdout.flush()
            train_all_depl(ix, model_1, model_2, jx, True)
    
        print_all_weights_depl(0, model_1, model_2, True)
    
    model_2.save_weights("depl_weights_train_model.hdf5")
    """
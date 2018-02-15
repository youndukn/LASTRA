import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
#import pydot
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
#from matplotlib.pyplot import imshow
from keras.optimizers import Adam
import os
import pickle
import random

import keras.backend as K
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys


import tensorflow as tf

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' +str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) +  block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

"""
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f=2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out ="+ str(out[0][1][1][0]))
"""

def convolution_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block +'_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X

def convolution_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block +'_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X
def convolution_block_no(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block +'_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)

    #X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis=3, name = bn_name_base + '1')(X_shortcut)

    X = Activation("relu")(X)

    return X


def ResNet50(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    #X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis= 3, name = 'bn_conv1')(X)
    #X = Activation('relu')(X)

    stage_filters = [64, 64, 256]
    X = convolution_block(X_input, f = 3, filters = stage_filters, stage = 1, block='a', s = 1)
    X = identity_block(X, 3, stage_filters, stage=1, block ='b')
    X = identity_block(X, 3, stage_filters, stage=1, block='c')

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f = 3, filters = stage_filters, stage = 2, block='a', s = 2)
    X = identity_block(X, 3, stage_filters, stage=2, block ='b')
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
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def FullyConnected(input_shape = (64, 64, 3), classes = 6):
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

def FullyConnected(input_shape = (64, 64, 3), classes = 6):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(256*2, name='fc1' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(256*3, name='fc2' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(256*3, name='fc3' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(256*2, name='fc4' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = Dense(classes, name='fc6' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name="FullyConnected")

    return model

def ResNet10(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f = 3, filters = stage_filters, stage = stage, block='a', s = 2)
    X = identity_block(X, 3, stage_filters, stage=stage, block ='b')
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
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)


    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNet5(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f = 3, filters = stage_filters, stage = stage, block='a', s = 2)
    X = identity_block(X, 3, stage_filters, stage=stage, block ='b')
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
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)


    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNet2(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(2):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    stage_filters = [128, 128, 512]
    X = convolution_block(X, f = 3, filters = stage_filters, stage = stage, block='a', s = 2)
    X = identity_block(X, 3, stage_filters, stage=stage, block ='b')
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
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)


    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNet20(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI1(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(1):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI3(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(3):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI7(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(7):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI14(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(14):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI40(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(40):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetC1(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(1):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetC3(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(3):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetC7(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(7):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetC14(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(14):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetC40(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(40):
        stage_filters = [64, 64, 256]
        X = convolution_block_no(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_64(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 64]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_128(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 128]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_256(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_512(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 512]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_1024(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 1024]
        X = convolution_block(X_input, f = 3, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 3, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 3, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_f2(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 2, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 2, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 2, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_f4(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 4, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 4, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 4, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def ResNetI5_f5(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block(X_input, f = 5, filters = stage_filters, stage = stage, block='a_{}'.format(i), s = 1)
        X = identity_block(X, 5, stage_filters, stage=stage, block ='b_{}'.format(i))
        X = identity_block(X, 5, stage_filters, stage=stage, block='c_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet0")

    return model

def shuffle_all_data():

    dictionary = {}
    creal = 0
    call = 0
    for _ in range(0, 1):
        for j in range(1, 9):
            for i in range(0, 14):
                a_string = './full_core_data_{}/data_{}'.format(j, i)
                print('Processing File ./full_core_data_{}/data_{}'.format(j, i))

                train_sets = []

                try:
                    if os.path.isfile(a_string):

                        file_read = open(a_string, 'rb')

                        while True:
                            train_set = pickle.load(file_read)
                            state = train_set[0]
                            value = hash(state.tostring())

                            if not (value in dictionary):
                                train_sets.append(train_set)
                                creal += 1
                                call += 1
                                dictionary[value] = 1
                            else:
                                dictionary[value] += 1
                                call += 1



                except:
                    pass

                to_string = './full_core_data_{}/data_{}_s'.format(j, i)

                file = open(to_string, 'wb')

                numpy.random.shuffle(train_sets)

                for trainset in train_sets:
                    pickle.dump(trainset, file, protocol=pickle.HIGHEST_PROTOCOL)

                file.close()
                file_read = open(to_string, 'rb')

def train_center():
    for _ in range(0, 2):

        for j in range(1, 9):
            for i in range(0, 14):

                to_string = './full_core_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        #random_range = numpy.random.shuffle((range(0, 12)))
        while len(files) > 0:
            try:

                file_index = random.randrange(len(files))
                file_read = files[file_index]

                y_batch_init_temp = []
                s_batch_init_temp = []

                for num_batch in range(0, 20):
                    train_set = pickle.load(file_read)
                    if train_set and len(train_set)>0:

                        s_batch_init = train_set[0]
                        y_batch_init = train_set[1][0]
                        if train_set[0].any() and train_set[1].any():
                            if y_batch_init > 0.7 and y_batch_init < 1.0:
                                y_batch_init_temp.append(y_batch_init)
                                s_batch_init_temp.append(s_batch_init)

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                y_batch_init_temp = numpy.array(y_batch_init_temp)

                model.train_on_batch(s_batch_init_temp, y_batch_init_temp)

                train_set = pickle.load(file_read)

                # Clear gradients
                s_batch_init = train_set[0]
                y_batch_init = train_set[1]

                s_batch_init = numpy.array(s_batch_init)
                s_batch_init = np.expand_dims(s_batch_init, axis=0)

                if y_batch_init[0] > 0.7 and y_batch_init[0] < 1.0:
                    readout_t0 = model.predict(s_batch_init)[0]
                    # print(y_batch_init,numpy.average(abs(y_batch_init - readout_t0)))
                    print(abs(100*(readout_t0[0] - y_batch_init[0]) / y_batch_init[0]), readout_t0[0], y_batch_init[0])
                    error_plot.append(abs((readout_t0[0] - y_batch_init[0]) / y_batch_init[0] * 100))

            except (AttributeError,  EOFError, ImportError, IndexError) as e:
                files.remove(file_read)
                file_read.close()
                pass

def train_maximum(iter_numb):
    for x_man in range(1, 2):

        for j in range(1, 7):
            for i in range(0, 14):

                to_string = './full_core_data_{}/data_{}_s'.format(j, i)

                file_read = open(to_string, 'rb')
                files.append(file_read)

        print("Stage{}".format( x_man ))

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

        #random_range = numpy.random.shuffle((range(0, 12)))
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
                    sys.stdout.write("\rDoing thing %i" % counter)
                    sys.stdout.flush()

                    if train_set and len(train_set)>0:

                        s_batch_init = train_set[0]
                        y_batch_init = numpy.amax(train_set[1])
                        if train_set[0].any() and train_set[1].any():
                            y_batch_init_temp.append(y_batch_init)
                            s_batch_init_temp.append(s_batch_init)

                s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
                y_batch_init_temp = numpy.array(y_batch_init_temp)

                model.train_on_batch(s_batch_init_temp, y_batch_init_temp)

            except (AttributeError,  EOFError, ImportError, IndexError) as e:
                files.remove(file_read)
                file_read.close()
                pass

        model.save_weights("weights-RES-{}-{}.hdf5".format(iter_numb, x_man))

def print_weights(index):
    table = []
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
                #print(abs(100 * (readout_t0[0] - y_batch_init) / y_batch_init))
                average += abs(100 * (readout_t0[0] - y_batch_init) / y_batch_init)
            print("Model {} Stage {}  Average : {}".format(index, x_man, average/100))
        except:
            pass


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

    """
    if i == 0:
        model = ResNetI5_f2((19, 19, 8), classes = 1)
    elif i == 1:
        model = ResNetI5_f4((19, 19, 8), classes = 1)
    elif i == 2:
        model = ResNetI5_f5((19, 19, 8), classes = 1)
    """
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    #model.load_weights("weights-RES50.hdf5")

    error_plot = []

    train_set = []

    files = []

    #shuffle_all_data()

    #train_maximum(i)

    print_weights(i)

    model.save_weights("weights-RES.hdf5")
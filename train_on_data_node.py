import numpy as np
from keras.layers import Input, Add, Dense, Activation, Concatenate, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,GlobalAveragePooling2D, \
    Conv3D, AveragePooling2D, MaxPooling2D, Reshape, Multiply, LeakyReLU, LSTM, GRU, Lambda,GlobalAveragePooling3D, MaxPooling1D, MaxPooling3D,ZeroPadding3D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, load_model, Sequential
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
import pickle
import random

import keras.backend as K
import numpy
import sys
import glob
import os
from astra_io.astra_ff_reader import AstraFFReader as FFL
from astra_rd import AstraRD
import matplotlib.pyplot as pl
import matplotlib.cm as cm

from keras.engine.topology import Layer
from keras.engine import InputSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
"""
AstraRD("/home/youndukn/Plants/1.4.0/ucn5/c16/depl/01_astra_u516_nep_depl.inp",
        main_directory="/home/youndukn/Plants/1.4.0/ucn5/c16/depl/", thread_id=-1)
"""
cl = 0
fxy = 9
fr = 10
both = 3
pd = 2
cbc = -1

astra = 1
karma = 4
karma_node = 6

cl_ra, cl_rb = 10000,10000
fxy_ra, fxy_rb = 1, 1

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


ab_si = [
    [0, 9, 9],
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
    [11, 10, 10],
    [22, 11, 11],
    [33, 12, 12],
    [44, 13, 13],
    [55, 14, 14],
]
ab_in = [
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [45, 13, 14],
    [46, 13, 15],
]

sep_cen = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26]
sep_out = [7, 14, 20, 24, 27, 28]
import tensorflow as tf
b_norm = 1000
dp_norm = (1000, 100, 1000, 100, 1, 1, 100, 1)


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):

    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):

    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = numpy.ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

"""

layer = model.get_layer(name="res0ca_0_branch2b")
weights = layer.get_weights()[0]

# Visualize weights
W = model.get_layer(name="res0ca_0_branch2b").get_weights()[0]
print("W shape : ", W.shape)

W = numpy.swapaxes(W[0,:,:,:], 0, 1)
W = numpy.swapaxes(W, 0, 2)
print("W shape : ", W.shape)

pl.figure(figsize=(20, 20))
pl.title('conv1 weights')
nice_imshow(pl.gca(), make_mosaic(W, 32, 2), cmap=cm.binary)
pl.show()
"""

def slicer(x_loc, y_loc):
    def func(x):
        return x[:,x_loc,y_loc,:]
    return Lambda(func)


def gSlicer():
    def func(x):
        return x[:,:8,:8,:]
    return Lambda(func)

def gSlicer3D():
    def func(x):
        return x[:,:8,:8,:, :]
    return Lambda(func)


def nodeRotate():
    def func(x):
        x[:, 1:, 0:1, :] = x[:, 1:2,  1:, :]
        x[:, 0:1, 1:, :] = x[:, 1:, 1:2, :]
        x[:, 0:1,  0:1, :] = x[:, 1:2,   1:2, :]
        return x
    return Lambda(func)


def nodeColPermute():
    def func(x):
        x = K.permute_dimensions(x[:, 0:1, :, :], (0, 2, 1, 3))
        return x
    return Lambda(func)

def nodeRowPermute():
    def func(x):
        x = K.permute_dimensions(x[:, :, 0:1, :], (0, 2, 1, 3))
        return x
    return Lambda(func)

def nodeCen():
    def func(x):
        return x[:,0:1, 0:1, :]
    return Lambda(func)

def assColPermute():
    def func(x):
        x = K.permute_dimensions(x[:, 1:2, :, :], (0, 2, 1, 3))
        return x
    return Lambda(func)

def assRowPermute():
    def func(x):
        x = K.permute_dimensions(x[:, :, 1:2, :], (0, 2, 1, 3))
        return x
    return Lambda(func)

def assCen():
    def func(x):
        return x[:,1:2, 1:2, :]
    return Lambda(func)




def nodeColPermute3D():
    def func(x):
        x = K.permute_dimensions(x[:, 0:1, :, :, :], (0, 2, 1, 3, 4))
        return x
    return Lambda(func)

def nodeRowPermute3D():
    def func(x):
        x = K.permute_dimensions(x[:, :, 0:1, :, :], (0, 2, 1, 3, 4))
        return x
    return Lambda(func)

def nodeCen3D():
    def func(x):
        return x[:,0:1, 0:1, :, :]
    return Lambda(func)

def assColPermute3D():
    def func(x):
        x = K.permute_dimensions(x[:, 1:2, :, :, :], (0, 2, 1, 3, 4))
        return x
    return Lambda(func)

def assRowPermute3D():
    def func(x):
        x = K.permute_dimensions(x[:, :, 1:2, :, :], (0, 2, 1, 3, 4))
        return x
    return Lambda(func)


def assCen3D():
    def func(x):
        return x[:,1:2, 1:2, :, :]
    return Lambda(func)

def assemblyRotate():
    def func(x):
        x[:, 1:, 0:1, :] = x[:, 2:3,  1:, :]
        x[:, 0:1, 1:, :] = x[:, 1:,  2:3, :]
        x[:, 0:1,  0:1, :] = x[:, 2:3,   2:3, :]
        return x
    return Lambda(func)

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


class SymmetricPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'SYMMETRIC')


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def convolution_block_se_lk_nb(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X = Add()([X_shortcut, X])
    X = LeakyReLU()(X)

    return X


def convolution_block_se_lk_bn(X, f, filters, stage, block, s=2):
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
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = LeakyReLU()(X)

    return X


def identity_block_bn(X, f, filters, stage, block):
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


def convolution_block_se_nb(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn(X, f, filters, stage, block, s=2):
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
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_jin(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2d')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*2, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_jin_3d(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(f, 1, 1), strides=(1, 1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, f, 1), strides=(1, 1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, 1, f), strides=(1, 1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)

    se = GlobalAveragePooling3D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*2, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_jin_3d_iquad(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = ZeroPadding3D(padding=((0, 1), (0, 1), (1, 1)))(X)
    X_Col = nodeColPermute3D()(X)
    X_Row = nodeRowPermute3D()(X)
    X_Cen = nodeCen3D()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])

    X = Conv3D(filters=F2, kernel_size=(f, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, f, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, 1, f), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)

    se = GlobalAveragePooling3D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*2, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X

def convolution_block_se_bn_jin_2d_iquad(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D(padding=((0, 1), (0, 1)))(X)
    X_Col = nodeColPermute()(X)
    X_Row = nodeRowPermute()(X)
    X_Cen = nodeCen()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*2, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)

    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X




def convolution_block_se_bn_jin_se(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2d')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_jin_skip(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2d')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*2, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_jin_nbn(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*2, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_jin_sig(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('sigmoid')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('sigmoid')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('sigmoid')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2d')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*2, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('sigmoid')(X)

    return X


def convolution_block_resize(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3= filters

    X = Conv2D(filters=F1, kernel_size=(2, 2), strides=(2, 2), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='valid', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = gSlicer()(X)
    X = Activation('relu')(X)



    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_3d(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2= filters

    X = Conv3D(filters=F1, kernel_size=(2, 2, 3), strides=(2, 2, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(f, f, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_3d_pin(X, X_input,f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3= filters

    X = Conv3D(filters=F1, kernel_size=(2, 2, 3), strides=(2, 2, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(f, f, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = gSlicer()(X)
    X = Multiply()([X, X_input])
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_3d_wopin(X,f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3= filters

    X = Conv3D(filters=F1, kernel_size=(2, 2, 3), strides=(2, 2, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(f, f, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = gSlicer()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_3d_wopin_iquad(X,f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3= filters

    X = Conv3D(filters=F1, kernel_size=(2, 2, 3), strides=(2, 2, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_2d_wopin_iquad(X,f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3= filters

    X = ZeroPadding2D(padding=((0, 1), (0, 1)))(X)
    X_Col = nodeColPermute()(X)
    X_Row = nodeRowPermute()(X)
    X_Cen = nodeCen()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv2D(filters=F1, kernel_size=(2, 2), strides=(2, 2), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = gSlicer()(X)

    X = ZeroPadding2D(padding=((0, 1), (0, 1)))(X)
    X_Col = assColPermute()(X)
    X_Row = assRowPermute()(X)
    X_Cen = assCen()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_3d_wopin_iquad(X,f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3= filters

    X = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 0)))(X)
    X_Col = nodeColPermute3D()(X)
    X_Row = nodeRowPermute3D()(X)
    X_Cen = nodeCen3D()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv3D(filters=F1, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = gSlicer3D()(X)

    X = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 0)))(X)
    X_Col = assColPermute3D()(X)
    X_Row = assRowPermute3D()(X)
    X_Cen = assCen3D()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv3D(filters=F2, kernel_size=(f, f, f), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_3d_wopin_iquad_all(X,f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3= filters

    X = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 0)))(X)
    X_Col = nodeColPermute3D()(X)
    X_Row = nodeRowPermute3D()(X)
    X_Cen = nodeCen3D()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv3D(filters=F1, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = gSlicer3D()(X)

    X = ZeroPadding3D(padding=((0, 1), (0, 1), (1, 1)))(X)
    X_Col = assColPermute3D()(X)
    X_Row = assRowPermute3D()(X)
    X_Cen = assCen3D()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv3D(filters=F2, kernel_size=(f, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, f, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, 1, f), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_resize_3d_wopin_iquad_all_global(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 0)))(X)
    X_Col = nodeColPermute3D()(X)
    X_Row = nodeRowPermute3D()(X)
    X_Cen = nodeCen3D()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv3D(filters=F1, kernel_size=(2, 2, 1), strides=(2, 2, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = gSlicer3D()(X)

    X = ZeroPadding3D(padding=((0, 1), (0, 1), (1, 1)))(X)
    X_Col = assColPermute3D()(X)
    X_Row = assRowPermute3D()(X)
    X_Cen = assCen3D()(X)
    X_Row = Concatenate(axis=2)([X_Cen, X_Row])
    X = Concatenate(axis=2)([X_Col, X])
    X = Concatenate(axis=1)([X_Row, X])
    X = Conv3D(filters=F2, kernel_size=(f, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, f, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F2, kernel_size=(1, 1, 16), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X

def convolution_block_resize_only(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2= filters

    X = Conv2D(filters=F1, kernel_size=(2, 2), strides=(2, 2), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    return X


def convolution_block_squash_only(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2= filters

    X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='valid', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_jin_id(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_incep(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X
    X_55 = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)


    X_55 = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '5a',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5a')(X_55)
    X_55 = Activation('relu')(X_55)

    X_55 = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '5b',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5b')(X_55)
    X_55 = Activation('relu')(X_55)

    X_55 = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '5c',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5c')(X_55)
    X_55 = Activation('relu')(X_55)


    X_55 = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '5d',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5d')(X_55)
    X_55 = Activation('relu')(X_55)

    X_55 = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '5e',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5e')(X_55)
    X_55 = Activation('relu')(X_55)

    X = Add()([X, X_55])

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2d')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_incep_1(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X
    X_55 = X

    X = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)


    X_55 = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '5b',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5b')(X_55)
    X_55 = Activation('relu')(X_55)

    X_55 = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '5c',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5c')(X_55)
    X_55 = Activation('relu')(X_55)


    X_55 = Conv2D(filters=F2, kernel_size=(1, f), strides=(1, 1), padding='same', name=conv_name_base + '5d',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5d')(X_55)
    X_55 = Activation('relu')(X_55)

    X_55 = Conv2D(filters=F2, kernel_size=(f, 1), strides=(1, 1), padding='same', name=conv_name_base + '5e',
               kernel_initializer=glorot_uniform(seed=0))(X_55)
    X_55 = BatchNormalization(axis=3, name=bn_name_base + '5e')(X_55)
    X_55 = Activation('relu')(X_55)

    X = Add()([X, X_55])

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2d',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2d')(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_se_bn_re(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3*4, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    X_shortcut = Activation('relu')(X_shortcut)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolution_block_bn(X, f, filters, stage, block, s=2):
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
    X = Activation('relu')(X)

    return X


"""
def ResNetI7_MD_BN(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_se_bn(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_se_bn(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    featureX = Dense(2000, name='cfc' + str(classes) + str(3), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(3000, name='cfc' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(featureX)
    X = Dense(1000, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, name='cfc' + str(classes) + str(0), kernel_initializer=glorot_uniform(seed=0))(X)

    F_input = Input((2000,))
    XF = Dense(3000, name='cff' + str(classes) + str(3), kernel_initializer=glorot_uniform(seed=0))(F_input)
    XF = Dense(1000, name='cff' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(XF)
    XF = Dense(classes, name='cff' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(XF)

    feature_model = Model(inputs = [X_input], outputs = featureX, name = "ResNetFeature")
    output_model = Model(inputs=[F_input], outputs=XF, name="ResNetOutput")
    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model, feature_model, output_model
"""


def ResNetI7_MD_BN(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(10):
        stage_filters = [256, 256, 256]
        X = convolution_block_se_bn_jin(X, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='cfc' + str(classes) + str(0), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model


def ResNetI7_MD_BN_aux(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        stage = stage + 1

    AUX_ouput = Flatten()(X)
    AUX_ouput = Dense(classes, name='cfb' + str(classes) + str(0), kernel_initializer=glorot_uniform(seed=0))(AUX_ouput)

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='cfc' + str(classes) + str(0), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = [X, AUX_ouput], name = "ResNetCen")

    return model


def ResNetI7_MD_BN_BU(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    stage_filters = [64, int(64/8), int(64/16)]
    X = convolution_block_se_bn_jin(X, f=1, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)

    stage = 0
    X = Flatten()(X)
    for i in range(24):
        X_pd = X
        X_pd = Dense(classes, name='cfc{}'.format(stage) + str(classes) + str(0), kernel_initializer=glorot_uniform(seed=0))(X_pd)
        stage = stage + 1
        outputs.append(X_pd)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []

    for i in range(24):
        X_pd = X

        stage_filters = [64, int(64 / 16), int(64 / 64)]
        X_pd = convolution_block_se_bn_jin(X_pd, f=1, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)

        each_pd = []

        for si_i, si in enumerate(ab_si):
            si11 = si[1]
            si12 = si[2]
            si21 = (si12 - 9) + 9
            si22 = -(si11 - 9) + 9
            si31 = (si22 - 9) + 9
            si32 = -(si21 - 9) + 9
            si41 = (si32 - 9) + 9
            si42 = -(si31 - 9) + 9

            s_pd1 = slicer(si11, si12)(X_pd)
            s_pd2 = slicer(si21, si22)(X_pd)
            s_pd3 = slicer(si31, si32)(X_pd)
            s_pd4 = slicer(si41, si42)(X_pd)

            s_pd = Concatenate()([s_pd1, s_pd2, s_pd3, s_pd4])

            s_pd = Dense(1, name='cfc' + str(classes) + str(stage)+ str(i) + str(si[0]),
                         kernel_initializer=glorot_uniform(seed=0))(s_pd)
            each_pd.append(s_pd)

        for in_val_i, in_val in enumerate(ab_in):
            si11 = in_val[1]
            si12 = in_val[2]
            si21 = (si12 - 9) + 9
            si22 = -(si11 - 9) + 9
            si31 = (si22 - 9) + 9
            si32 = -(si21 - 9) + 9
            si41 = (si32 - 9) + 9
            si42 = -(si31 - 9) + 9

            s_pd1 = slicer(si11, si12)(X_pd)
            s_pd2 = slicer(si21, si22)(X_pd)
            s_pd3 = slicer(si31, si32)(X_pd)
            s_pd4 = slicer(si41, si42)(X_pd)

            si11 = in_val[2]
            si12 = in_val[1]
            si21 = (si12 - 9) + 9
            si22 = -(si11 - 9) + 9
            si31 = (si22 - 9) + 9
            si32 = -(si21 - 9) + 9
            si41 = (si32 - 9) + 9
            si42 = -(si31 - 9) + 9

            s_pd5 = slicer(si11, si12)(X_pd)
            s_pd6 = slicer(si21, si22)(X_pd)
            s_pd7 = slicer(si31, si32)(X_pd)
            s_pd8 = slicer(si41, si42)(X_pd)

            s_pd = Concatenate()([s_pd1, s_pd2, s_pd3, s_pd4, s_pd5, s_pd6, s_pd7, s_pd8])

            s_pd = Dense(1, name='cfb' + str(classes) + str(stage)+str(i) + str(in_val[0]),
                         kernel_initializer=glorot_uniform(seed=0))(s_pd)
            each_pd.append(s_pd)

        final_pd = Concatenate()(each_pd)
        stage = stage + 1
        outputs.append(final_pd)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    stage_filters = [int(256/2), 24]
    X = convolution_block_resize(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
    X = Flatten()(X)

    model = Model(inputs=[X_input], outputs=X, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize_3d(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize_3d(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_quad(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_max(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = MaxPooling3D(pool_size=(8, 8, 1))(pd_X)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_2d_dense_max(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = MaxPooling2D(pool_size=(8, 8))(pd_X)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_max_iquad(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = MaxPooling3D(pool_size=(8, 8, 1))(pd_X)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_MD_BN_BU_AT_MATRIX_24_2d_dense_max_iquad(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_2d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_2d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_2d_wopin_iquad(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = MaxPooling2D(pool_size=(8, 8))(pd_X)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        #pd_X = MaxPooling2D(pool_size=(8, 8))(pd_X)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad_all(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad_all_global(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_GAN(input_shape=(17, 17, 5), input_shape2=(17, 17, 1), input_shape3=(17, 17, 1), classes=6):

    X_xs = Input(input_shape)

    stage = 0
    X_x = X_xs

    for i in range(15):
        stage_filters = [64, 64, 256]
        X_x = convolution_block_se_bn_jin_3d_iquad(X_x, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X_x = Concatenate()([X_x, X_xs])
        stage = stage + 1

    X_fxy = Input(input_shape2)

    X_xs = Input(input_shape3)

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad_all(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        #pd_X = MaxPooling2D(pool_size=(8, 8))(pd_X)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_local_max(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1
        stage_filters = [int(256 / 2), 64, 1]
        i_X = convolution_block_resize_3d_wopin_iquad(pd_X, f=3, filters=stage_filters, stage=stage,
                                                       block='ca_{}'.format(i), s=1)
        i_X = Flatten()(i_X)

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten()(pd_X)

        outputs.append(pd_X)
        outputs.append(i_X)


    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_MD_BN_BU_AT_MATRIX_24_2d_dense_max_iquad_cbc(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_2d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_2d_iquad(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(15):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_2d_wopin_iquad(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Conv2D(filters=1, kernel_size=(8, 8), strides=(1, 1))(pd_X)
        pd_X = Flatten()(pd_X)
        #pd_X = Dense(1)(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_max_local(input_shape=(64, 64, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [32, 32, 64]
        pd_X = convolution_block_se_bn_jin_3d(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Concatenate()([pd_X, X_input])
        stage = stage + 1

        stage_filters = [32, 16, 1]
        local_X = pd_X
        local_X = convolution_block_resize_3d_wopin(local_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i),
                                                 s=1)
        local_X = Flatten()(local_X)
        stage_filters = [32, 16, 1]
        pd_X = convolution_block_resize_3d_wopin(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = MaxPooling3D(pool_size=(8, 8, 1))(pd_X)
        pd_X = Flatten()(pd_X)

        outputs.append(local_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_MD_BN_BU_AT_MATRIX_1_3d_dense_pin(input_shape=(34, 34, 3, 3), input_shape2=(8, 8, 64), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    X_input2 = Input(input_shape2)
    outputs = []

    for i in range(1):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_pin(pd_X, X_input2, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input, X_input2], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_1_3d_dense(input_shape=(34, 34, 3, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []

    for i in range(1):
        pd_X = X
        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=X_input, outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_Merged(input_shape=(64, 64, 3), input_shape_2=(64,64,3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    stage_filters = [256, 24]
    X = convolution_block_resize_only(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(stage), s=1)
    stage = stage+1

    X_input_2 = Input(input_shape_2)
    X = Concatenate()([X, X_input_2])

    for i in range(2):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), int(256/2)]
        pd_X = convolution_block_squash_only(pd_X, f=3, filters=stage_filters, stage=stage, block='firstsq_{}'.format(i), s=1)
        stage = stage + 1
        stage_filters = [1, 1]
        pd_X = convolution_block_squash_only(pd_X, f=1, filters=stage_filters, stage=stage, block='secondsq_{}'.format(i), s=1)
        stage = stage + 1
        pd_X = Flatten(name=str(i))(pd_X)
        pd_X = Concatenate()([pd_X, pd_X])
        outputs.append(pd_X)

    model = Model(inputs=[X_input, X_input_2], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_Merged_1(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    stage_filters = [256, 24]
    X = convolution_block_resize_only(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(stage), s=1)
    stage = stage+1

    for i in range(2):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X
        stage_filters = [int(256/2), int(256/2)]
        pd_X = convolution_block_squash_only(pd_X, f=3, filters=stage_filters, stage=stage, block='firstsq_{}'.format(i), s=1)
        stage = stage + 1
        stage_filters = [1, 1]
        pd_X = convolution_block_squash_only(pd_X, f=1, filters=stage_filters, stage=stage, block='secondsq_{}'.format(i), s=1)
        stage = stage + 1
        pd_X = Flatten(name=str(i))(pd_X)
        pd_X = Concatenate()([pd_X, pd_X])
        outputs.append(pd_X)

    model = Model(inputs=X_input, outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_1(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(1):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_10(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_5(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_SE(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_se(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_se(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_SKIP(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_skip(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_skip(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_NBN(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_nbn(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_nbn(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_SIG(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_sig(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    for i in range(15):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_sig(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        stage = stage + 1

    outputs = []
    for i in range(24):
        pd_X = X

        stage_filters = [int(256/2), 1]
        pd_X = convolution_block_resize(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        pd_X = Flatten(name=str(i))(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7(input_shape = (64, 64, 3), classes = 6):

    X_input = Input(input_shape)
    stage = 0;

    for i in range(10):
        stage_filters = [64, 64, 256]
        X = convolution_block_bn(X_input, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        X = identity_block_bn(X, 3, stage_filters, stage=stage, block ='cb_{}'.format(i))
        X = identity_block_bn(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(2000, name='cfc' + str(classes) + str(3), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(1000, name='cfc' + str(classes) + str(2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, name='cfc' + str(classes) + str(1), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    return model


def shuffle_data(include = ["y3"], exclude = ["y310, _s"], folders=["/media/youndukn/lastra/plants_data1/"]):
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


def train_all_depl_fxy(iter_numb, model, include = ["y3"], exclude = ["y314"], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = get_files_with(include, exclude, "_s", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            y_batch_init_temp = []
            y1_batch_init_temp = []
            s_batch_init_temp = []


            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = burnup_boc[1][1:-1,1:-1,:]
                s_batch_init = numpy.array(s_batch_init)
                if den == True:
                    s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init_temp.append(s_batch_init)
                y_batch_init = burnup_eoc[0][0]

                max_fxy = 0.0
                # Clear gradients
                for burnup in depl_set:
                    if burnup[0][5] >max_fxy:
                        max_fxy = burnup[0][5]

                y_batch_init_temp.append(y_batch_init/18000.0)
                y1_batch_init_temp.append(max_fxy/2.0)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

            s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
            y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
            y1_batch_init_temp = numpy.array(y1_batch_init_temp, dtype=np.float16)

            model.train_on_batch(x=s_batch_init_temp, y=[y_batch_init_temp, y1_batch_init_temp])

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    model.save_weights("weights-SEP-DEPL-1-{}.hdf5".format(iter_numb))



def train_all_depl(iter_numb, model, include = ["y3"], exclude = ["y314"], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = get_files_with(include, exclude, "_s", folders)
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


            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = burnup_boc[1][1:-1,1:-1,:]
                s_batch_init = numpy.array(s_batch_init)
                if den == True:
                    s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init_temp.append(s_batch_init)
                y_batch_init = burnup_eoc[0][0]

                y_batch_init_temp.append(y_batch_init/18000.0)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

            s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
            y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

            model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    model.save_weights("weights-SEP-DEPL-1-{}.hdf5".format(iter_numb))


def train_all_fxy(iter_numb, model, include = ["y3"], exclude = ["y314"], folders=["/media/youndukn/lastra/plants_data/"], den=False):

    files = get_files_with(include, exclude, "_s", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            y1_batch_init_temp = []
            s_batch_init_temp = []


            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = burnup_boc[1][1:-1,1:-1,:]
                s_batch_init = numpy.array(s_batch_init)
                if den == True:
                    s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init_temp.append(s_batch_init)
                y_batch_init = burnup_eoc[0][0]

                max_fxy = 0.0
                # Clear gradients
                for burnup in depl_set:
                    if burnup[0][5] >max_fxy:
                        max_fxy = burnup[0][5]

                y1_batch_init_temp.append(max_fxy/2.0)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

            s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
            y1_batch_init_temp = numpy.array(y1_batch_init_temp, dtype=np.float16)

            model.train_on_batch(x=s_batch_init_temp, y=y1_batch_init_temp)

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    model.save_weights("{}.hdf5".format(iter_numb))


def train_ixs(optimize_id,
              input_id,
              model,
              include = ["y3"],
              exclude = ["y314"],
              folders=["/media/youndukn/lastra/plants_data/"],
              den=False):

    files = get_files_with(include, exclude, "_s", folders)
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


            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = np.delete(burnup_boc[input_id], 9, 1)
                s_batch_init = np.insert(s_batch_init, 18, 0.0, 1)
                s_batch_init = s_batch_init[1:-1, 1:-1, :]
                s_batch_init = numpy.array(s_batch_init)
                if den == True:
                    s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init_temp.append(s_batch_init)

                if optimize_id == cl:
                    y_batch_init =(burnup_eoc[0][0]-cl_rb)/cl_ra
                    y_batch_init_temp.append(y_batch_init)
                elif optimize_id == fxy:
                    max_fxy = 0.0

                    for burnup in depl_set:
                        if burnup[0][5] >max_fxy:
                            max_fxy = burnup[0][5]
                    max_fxy = (max_fxy - fxy_rb) / fxy_ra
                    y_batch_init_temp.append(max_fxy)

                elif optimize_id == both:
                    cycle_length = (burnup_eoc[0][0]-cl_rb)/cl_ra

                    max_fxy = 0.0
                    for burnup in depl_set:
                        if burnup[0][5] >max_fxy:
                            max_fxy = burnup[0][5]
                    max_fxy = (max_fxy-fxy_rb)/fxy_ra

                    y_batch_init_temp.append([cycle_length, max_fxy])

                elif optimize_id == pd:
                    y_batch_temp = []
                    for si in ab_si:
                        y_batch_temp.append(burnup_boc[2][si[0]])
                    for in_tmp in ab_in:
                        y_batch_temp.append(burnup_boc[2][in_tmp[0]])
                    y_batch_init_temp.append(y_batch_temp)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

            s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
            y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

            model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

def prepare_ixs(optimize_id,
              input_id,
              model,
              include = ["y3"],
              exclude = ["y314"],
              folders=["/media/youndukn/lastra/plants_data/"],
              den=False):

    files = get_files_with(include, exclude, "_s", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]



            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = np.delete(burnup_boc[input_id], 9, 1)
                s_batch_init = np.insert(s_batch_init, 18, 0.0, 1)
                s_batch_init = s_batch_init[1:-1, 1:-1, :]
                s_batch_init = numpy.array(s_batch_init)
                if den == True:
                    s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                    s_batch_init_den = numpy.array(s_batch_init_den)
                    s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                s_batch_init_temp.append(s_batch_init)

                if optimize_id == cl:
                    y_batch_init =(burnup_eoc[0][0]-cl_rb)/cl_ra
                    y_batch_init_temp.append(y_batch_init)
                elif optimize_id == fxy:
                    max_fxy = 0.0

                    for burnup in depl_set:
                        if burnup[0][5] >max_fxy:
                            max_fxy = burnup[0][5]
                    max_fxy = (max_fxy - fxy_rb) / fxy_ra
                    y_batch_init_temp.append(max_fxy)

                elif optimize_id == both:
                    cycle_length = (burnup_eoc[0][0]-cl_rb)/cl_ra

                    max_fxy = 0.0
                    for burnup in depl_set:
                        if burnup[0][5] >max_fxy:
                            max_fxy = burnup[0][5]
                    max_fxy = (max_fxy-fxy_rb)/fxy_ra

                    y_batch_init_temp.append([cycle_length, max_fxy])

                elif optimize_id == pd:
                    y_batch_temp = []
                    for si in ab_si:
                        y_batch_temp.append(burnup_boc[2][si[0]])
                    for in_tmp in ab_in:
                        y_batch_temp.append(burnup_boc[2][in_tmp[0]])
                    y_batch_init_temp.append(y_batch_temp)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

    return s_batch_init_temp, y_batch_init_temp


def prepare_ixs1(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"],
                den=False):
    files = get_files_with(include, exclude, "_s", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                for burnup1 in depl_set:
                    s_batch_init = np.delete(burnup1[input_id], 9, 1)
                    s_batch_init = np.insert(s_batch_init, 18, 0.0, 1)
                    s_batch_init = s_batch_init[1:-1, 1:-1, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init_temp.append(s_batch_init)

                    y_batch_temp = []
                    for si in ab_si:
                        y_batch_temp.append(burnup1[2][si[0]])
                    for in_tmp in ab_in:
                        y_batch_temp.append(burnup1[2][in_tmp[0]])
                    y_batch_init_temp.append(y_batch_temp)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

    return s_batch_init_temp, y_batch_init_temp

def prepare_ixs_aux(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"],
                den=False):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                for burnup1 in depl_set:
                    s_batch_init = np.delete(burnup1[input_id], 9, 1)
                    s_batch_init = np.insert(s_batch_init, 18, 0.0, 1)
                    s_batch_init = s_batch_init[1:-1, 1:-1, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init_temp.append(s_batch_init)

                    y_batch_temp = []
                    y_batch_temp2 = []
                    for si in ab_si:
                        y_batch_temp.append(burnup1[2][si[0]])
                        y_batch_temp2.append(burnup1[2][si[0]])
                    for in_tmp in ab_in:
                        y_batch_temp.append(burnup1[2][in_tmp[0]])
                        y_batch_temp2.append(burnup1[2][in_tmp[0]])
                    y_batch_init_temp.append([y_batch_temp, y_batch_temp2])

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

    return s_batch_init_temp, y_batch_init_temp

def prepare_ixs_aux_node(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"],
                den=False):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    y_batch_init_temp1 = []
    s_batch_init_temp = []

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                for burnup1 in depl_set:
                    s_batch_init = burnup1[input_id]
                    s_batch_init = s_batch_init[2:-2, 2:-2, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init_temp.append(s_batch_init)

                    y_batch_temp = []
                    y_batch_temp1 = []
                    for si in ab_si:
                        y_batch_temp.append(burnup1[2][si[0]])
                        y_batch_temp1.append(burnup1[2][si[0]])
                    for in_tmp in ab_in:
                        y_batch_temp.append(burnup1[2][in_tmp[0]])
                        y_batch_temp1.append(burnup1[2][in_tmp[0]])
                    y_batch_init_temp.append(y_batch_temp)
                    y_batch_init_temp1.append(y_batch_temp1)
            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    y_batch_init_temp1 = numpy.array(y_batch_init_temp1, dtype=np.float16)

    return s_batch_init_temp, y_batch_init_temp, y_batch_init_temp1

def prepare_ixs_bu_node(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(24):
        y_batch_init_temp.append([])

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[input_id]
                s_batch_init = s_batch_init[2:-2, 2:-2, :]
                s_batch_init = numpy.array(s_batch_init)
                s_batch_init_temp.append(s_batch_init)


                for bu, burnup1 in enumerate(depl_set):
                    y_batch_temp = []
                    for si in ab_si:
                        y_batch_temp.append(burnup1[2][si[0]])
                    for in_tmp in ab_in:
                        y_batch_temp.append(burnup1[2][in_tmp[0]])
                    y_batch_temp = numpy.array(y_batch_temp, dtype=np.float16)
                    y_batch_init_temp[bu].append(y_batch_temp)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    return s_batch_init_temp, y_batch_init_temp


def prepare_ixs_bu_node_matrix(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[input_id]
                s_batch_init = s_batch_init[2:-2, 2:-2, :]
                s_batch_init = numpy.array(s_batch_init)
                s_batch_init_temp.append(s_batch_init)

                values = numpy.zeros((8, 8, 24))
                for bu, burnup1 in enumerate(depl_set):

                    a = burnup1[2].reshape((10, 10))
                    values[:,:,bu] = a[:8,:8]

                second = np.rot90(values)
                first = np.rot90(second)
                third = np.delete(np.rot90(first), 0, 0)
                fourth = np.delete(values, 0, 0)

                second = np.array(second, dtype=float)
                first = np.array(first, dtype=float)
                third = np.array(third, dtype=float)
                fourth = np.array(fourth, dtype=float)

                half1 = np.concatenate((first, third))
                half2 = np.concatenate((second, fourth))

                full_matrix = np.concatenate((half1, half2), axis=1)
                full_matrix = np.delete(full_matrix, [7, ], axis=1)
                full_matrix = full_matrix.flatten()
                y_batch_init_temp.append(full_matrix)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp, y_batch_init_temp

def prepare_ixs_bu_node_matrix_24(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(24):
        y_batch_init_temp.append([])

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[input_id]
                s_batch_init = s_batch_init[2:-2, 2:-2, :]
                s_batch_init = numpy.array(s_batch_init)
                s_batch_init_temp.append(s_batch_init)

                for bu, burnup1 in enumerate(depl_set):
                    values = numpy.zeros((8, 8, 1))
                    a = burnup1[2].reshape((10, 10))
                    values[:,:, 0] = a[:8,:8]
                    values[:,0,0] = values[0,:,0]

                    second = np.rot90(values)
                    first = np.rot90(second)
                    third = np.delete(np.rot90(first), 0, 0)
                    fourth = np.delete(values, 0, 0)

                    second = np.array(second, dtype=float)
                    first = np.array(first, dtype=float)
                    third = np.array(third, dtype=float)
                    fourth = np.array(fourth, dtype=float)

                    half1 = np.concatenate((first, third))
                    half2 = np.concatenate((second, fourth))

                    full_matrix = np.concatenate((half1, half2), axis=1)
                    full_matrix = np.delete(full_matrix, [7, ], axis=1)
                    full_matrix = full_matrix.flatten()
                    y_batch_init_temp[bu].append(full_matrix)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp, y_batch_init_temp


def prepare_ixs_bu_node_matrix_24_fxy(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(24):
        y_batch_init_temp.append([])

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[input_id]
                s_batch_init = s_batch_init[2:-2, 2:-2, :]
                s_batch_init = numpy.array(s_batch_init)
                s_batch_init_temp.append(s_batch_init)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[9][2:-2, 2:-2, :]).flatten()
                    y_batch_init_temp[bu].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp, y_batch_init_temp




def prepare_ixs_bu_node_matrix_24_fr(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(24):
        y_batch_init_temp.append([])

    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[input_id]
                s_batch_init = s_batch_init[2:-2, 2:-2, :]
                s_batch_init = numpy.array(s_batch_init)
                s_batch_init_temp.append(s_batch_init)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[10][2:-2, 2:-2, :]).flatten()
                    y_batch_init_temp[bu].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp, y_batch_init_temp

def prepare_ixs_bu_node_matrix_24_3d(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4), :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2), :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4), :]), axis=-1)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[9][2:-2, 2:-2, :]).flatten()
                    y_batch_init_temp[bu].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp


def prepare_ixs_bu_node_matrix_24_3d_conv(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4):int(axial_numb/4)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2):int(axial_numb/2)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4)+1:int(axial_numb*3/4)+2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[optimize_id][2:-2, 2:-2, :]).flatten()
                    y_batch_init_temp[bu].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def prepare_ixs_bu_node_matrix_24_3d_conv_quad(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4):int(axial_numb/4)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2):int(axial_numb/2)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4)+1:int(axial_numb*3/4)+2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[optimize_id][2:10, 2:10, :]).flatten()
                    y_batch_init_temp[bu].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp


def prepare_ixs_bu_node_matrix_24_3d_conv_max(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4):int(axial_numb/4)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2):int(axial_numb/2)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4)+1:int(axial_numb*3/4)+2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.amax(numpy.array(burnup1[optimize_id][2:10, 2:10, :]).flatten())
                    y_batch_init_temp[bu].append(numpy.array([y_batch_init]))

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp


def prepare_ixs_bu_node_matrix_24_3d_conv_max_iquad(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[-19:-2, -19:-2, int(axial_numb/4):int(axial_numb/4)+1, :],
                     s_batch_init[-19:-2, -19:-2, int(axial_numb/2):int(axial_numb/2)+1, :],
                     s_batch_init[-19:-2, -19:-2, int(axial_numb*3/4)+1:int(axial_numb*3/4)+2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.amax(numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten())
                    y_batch_init_temp[bu].append(numpy.array([y_batch_init]))

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def prepare_ixs_bu_node_matrix_24_2d_conv_max_iquad(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(15):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[6]
                s_batch_init1 = numpy.array(s_batch_init[-19:-2, -19:-2, :])

                s_batch_init_temp.append(s_batch_init1)


                for bu, burnup1 in enumerate(depl_set):
                    if bu < 15:
                        if optimize_id == cbc:
                            y_batch_init_temp[bu].append(numpy.array([burnup1[0][1]/1000]))
                        else:
                            y_batch_init = numpy.amax(numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten())
                            y_batch_init_temp[bu].append(numpy.array([y_batch_init]))
            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp



def prepare_ixs_bu_node_matrix_24_3d_conv_iquad_averaged(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    everything = [9.05,	6.19,	2.86,	12.38,	7.62,	20,	18.1,	18.1,	20,
                  20,	18.1,	18.1,	20,	20,	18.1,	18.1,	20,
                  18.1, 20, 18.1, 20, 7.62, 12.38, 2.86, 6.19, 9.05]
    first = [9.05,	6.19,	2.86,	12.38,	7.62,	20,	18.1,	18.1,	20]
    second = [20,	18.1,	18.1,	20,	20,	18.1,	18.1,	20]
    third = [18.1,	20,	18.1,	20,	7.62,	12.38,	2.86,	6.19,	9.05]

    first_sum = 114.3
    second_sum = 152.4
    third_sum  = 114.3

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]

                first_batch = s_batch_init[-19:-2, -19:-2, 0:1, :]*first[0]
                for xxx in range(1, 9):
                    xxx1 = xxx-0
                    first_batch[:,:,0,:] += s_batch_init[-19:-2, -19:-2, xxx, :]*first[xxx1]
                first_batch /= first_sum

                second_batch = s_batch_init[-19:-2, -19:-2, 9:10, :]*second[0]
                for xxx in range(10, 17):
                    xxx1 = xxx-9
                    second_batch[:,:,0,:] += s_batch_init[-19:-2, -19:-2, xxx, :]*second[xxx1]
                second_batch /= second_sum

                third_batch = s_batch_init[-19:-2, -19:-2, 17:18, :]*third[0]
                for xxx in range(18, 26):
                    xxx1 = xxx-17
                    third_batch[:,:,0,:] += s_batch_init[-19:-2, -19:-2, xxx, :]*third[xxx1]
                third_batch /= third_sum

                s_batch_init1 = numpy.concatenate(
                    (first_batch,
                     second_batch,
                     third_batch), axis=-2)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    if optimize_id == cbc:
                        y_batch_init_temp[bu].append(numpy.array([burnup1[0][1]/1000]))
                    else:
                        y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                        y_batch_init_temp[bu].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def prepare_ixs_bu_node_matrix_24_3d_conv_iquad(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    y_batch_init_temp1 = []

    indexes = [0, 1, 10, 11, 12, 13, 14, 15, 16]

    for i in range(24):
        y_batch_init_temp.append([])

    for i in indexes:
        y_batch_init_temp1.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]

                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[-19:-2, -19:-2, 5:5+1, :],
                     (s_batch_init[-19:-2, -19:-2, 12:12+1, :]+s_batch_init[-19:-2, -19:-2, 13:13+1, :])/2,
                     s_batch_init[-19:-2, -19:-2, 20:21, :]), axis=-2)

                s_batch_init_temp.append(s_batch_init1)
                maximum1 = 0
                bu1 = 0
                for bu, burnup1 in enumerate(depl_set):
                    if optimize_id == cbc:
                        y_batch_init_temp[bu].append(numpy.array([burnup1[0][1]/1000]))
                    else:
                        y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                        a_maximum = numpy.amax(y_batch_init)
                        if maximum1 < a_maximum:
                            maximum1 = a_maximum
                            bu1 = bu
                        y_batch_init_temp[bu].append(y_batch_init)

                maximum2 = 0
                bu2 = 0
                for bu_i, bu in enumerate(indexes):
                    burnup1 = depl_set[bu]
                    y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                    a_maximum = numpy.amax(y_batch_init)
                    if maximum2 < a_maximum:
                        maximum2 = a_maximum
                        bu2 = bu
                    y_batch_init_temp1[bu_i].append(y_batch_init)

                if maximum1 != maximum2:
                    print(" ",bu1, bu2, maximum1, maximum2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp, y_batch_init_temp1



def prepare_ixs_bu_node_matrix_24_3d_conv_iquad(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    y_batch_init_temp1 = []

    indexes = [0, 1, 10, 11, 12, 13, 14, 15, 16]

    for i in range(24):
        y_batch_init_temp.append([])

    for i in indexes:
        y_batch_init_temp1.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]

                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[-19:-2, -19:-2, 5:5+1, :],
                     (s_batch_init[-19:-2, -19:-2, 12:12+1, :]+s_batch_init[-19:-2, -19:-2, 13:13+1, :])/2,
                     s_batch_init[-19:-2, -19:-2, 20:21, :]), axis=-2)

                s_batch_init_temp.append(s_batch_init1)
                maximum1 = 0
                bu1 = 0
                for bu, burnup1 in enumerate(depl_set):
                    if optimize_id == cbc:
                        y_batch_init_temp[bu].append(numpy.array([burnup1[0][1]/1000]))
                    else:
                        y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                        a_maximum = numpy.amax(y_batch_init)
                        if maximum1 < a_maximum:
                            maximum1 = a_maximum
                            bu1 = bu
                        y_batch_init_temp[bu].append(y_batch_init)

                maximum2 = 0
                bu2 = 0
                for bu_i, bu in enumerate(indexes):
                    burnup1 = depl_set[bu]
                    y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                    a_maximum = numpy.amax(y_batch_init)
                    if maximum2 < a_maximum:
                        maximum2 = a_maximum
                        bu2 = bu
                    y_batch_init_temp1[bu_i].append(y_batch_init)

                if maximum1 != maximum2:
                    print(" ",bu1, bu2, maximum1, maximum2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp, y_batch_init_temp1

def prepare_ixs_bu_node_matrix_1_3d_conv_iquad(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]

                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[-19:-2, -19:-2, 5:5+1, :],
                     (s_batch_init[-19:-2, -19:-2, 12:12+1, :]+s_batch_init[-19:-2, -19:-2, 13:13+1, :])/2,
                     s_batch_init[-19:-2, -19:-2, 20:21, :]), axis=-2)

                s_batch_init_temp.append(s_batch_init1)

                y_batch_init_temp1 = []
                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                    y_batch_init_temp1.append(y_batch_init)
                y_batch_init_temp1 = numpy.array(y_batch_init_temp1)
                y_batch_init_temp2 = numpy.argmax(y_batch_init_temp1, axis=0)
                y_batch_init_temp1 = numpy.max(y_batch_init_temp1, axis=0)
                y_batch_init_temp2 = numpy.divide(y_batch_init_temp2, 5).astype(int)
                y_batch_init_temp2[y_batch_init_temp2 > 1] = 1
                y_batch_init_temp[0].append(y_batch_init_temp1)
                #y_batch_init_temp[1].append(y_batch_init_temp2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def prepare_ixs_bu_node_matrix_1_3d_conv_iquad(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]

                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[-19:-2, -19:-2, 5:5+1, :],
                     (s_batch_init[-19:-2, -19:-2, 12:12+1, :]+s_batch_init[-19:-2, -19:-2, 13:13+1, :])/2,
                     s_batch_init[-19:-2, -19:-2, 20:21, :]), axis=-2)

                s_batch_init_temp.append(s_batch_init1)

                y_batch_init_temp1 = []
                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                    y_batch_init_temp1.append(y_batch_init)
                y_batch_init_temp1 = numpy.array(y_batch_init_temp1)
                y_batch_init_temp2 = numpy.argmax(y_batch_init_temp1, axis=0)
                y_batch_init_temp1 = numpy.max(y_batch_init_temp1, axis=0)
                y_batch_init_temp2 = numpy.divide(y_batch_init_temp2, 5).astype(int)
                y_batch_init_temp2[y_batch_init_temp2 > 1] = 1
                y_batch_init_temp[0].append(y_batch_init_temp1)
                #y_batch_init_temp[1].append(y_batch_init_temp2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp


def prepare_ixs_bu_node_matrix_1_3d_conv_iquad_all_global(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]

                s_batch_init_temp.append(s_batch_init[-19:-2, -19:-2, 6:20, :])

                y_batch_init_temp1 = []
                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                    y_batch_init_temp1.append(y_batch_init)
                y_batch_init_temp1 = numpy.array(y_batch_init_temp1)
                y_batch_init_temp2 = numpy.argmax(y_batch_init_temp1, axis=0)
                y_batch_init_temp1 = numpy.max(y_batch_init_temp1, axis=0)
                y_batch_init_temp2 = numpy.divide(y_batch_init_temp2, 5).astype(int)
                y_batch_init_temp2[y_batch_init_temp2 > 1] = 1
                y_batch_init_temp[0].append(y_batch_init_temp1)
                #y_batch_init_temp[1].append(y_batch_init_temp2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp


def prepare_ixs_bu_node_matrix_1_3d_conv_iquad_divide(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]

                s_batch_init = burnup_boc[14]
                for axial_index in range(6, 20):
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[-19:-2, -19:-2, axial_index-1:axial_index-1+1, :],
                         s_batch_init[-19:-2, -19:-2, axial_index:axial_index+1, :],
                         s_batch_init[-19:-2, -19:-2, axial_index+1:axial_index+1+1, :]), axis=-2)

                    s_batch_init_temp.append(s_batch_init1)

                    y_batch_init_temp1 = []
                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[13][-10:-2, -10:-2, axial_index, :]).flatten()
                        y_batch_init_temp1.append(y_batch_init)
                    y_batch_init_temp1 = numpy.array(y_batch_init_temp1)
                    y_batch_init_temp2 = numpy.argmax(y_batch_init_temp1, axis=0)
                    y_batch_init_temp1 = numpy.max(y_batch_init_temp1, axis=0)
                    y_batch_init_temp2 = numpy.divide(y_batch_init_temp2, 5).astype(int)
                    y_batch_init_temp2[y_batch_init_temp2 > 1] = 1
                    y_batch_init_temp[0].append(y_batch_init_temp1)
                    #y_batch_init_temp[1].append(y_batch_init_temp2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp




def prepare_ixs_bu_node_matrix_1_3d_conv_iquad_all(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]

                s_batch_init = burnup_boc[14]

                s_batch_init_temp.append(s_batch_init[-19:-2, -19:-2, 6:20, :])

                y_batch_init_temp1 = []
                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[13][-10:-2, -10:-2, 6:20, :]).flatten()
                    y_batch_init_temp1.append(y_batch_init)
                y_batch_init_temp1 = numpy.array(y_batch_init_temp1)
                y_batch_init_temp1 = numpy.max(y_batch_init_temp1, axis=0)
                y_batch_init_temp[0].append(y_batch_init_temp1)
                #y_batch_init_temp[1].append(y_batch_init_temp2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def prepare_ixs_bu_node_matrix_1_3d_conv_iquad_boc(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []

    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]

                s_batch_init = burnup_boc[14]

                s_batch_init_temp.append(s_batch_init[-19:-2, -19:-2, 6:20, :])
                """
                y_batch_init_temp1 = []
                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[13][-10:-2, -10:-2, 6:20, :]).flatten()
                    y_batch_init_temp1.append(y_batch_init)
                y_batch_init_temp1 = numpy.array(y_batch_init_temp1)
                y_batch_init_temp1 = numpy.max(y_batch_init_temp1, axis=0)
                y_batch_init_temp[0].append(y_batch_init_temp1)
                """
                y_batch_init = numpy.array(burnup_boc[13][-10:-2, -10:-2, 6:20, :]).flatten()
                y_batch_init_temp[0].append(y_batch_init)
                #y_batch_init_temp[1].append(y_batch_init_temp2)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def prepare_ixs_bu_node_matrix_24_2d_conv_max(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []
    if optimize_id == cbc:
        for i in range(15):
            y_batch_init_temp.append([])
    else:
        for i in range(24):
            y_batch_init_temp.append([])
    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[6]
                s_batch_init1 = numpy.array(s_batch_init[2:-2, 2:-2, :])

                s_batch_init_temp.append(s_batch_init1)

                if optimize_id == cbc:
                    for bu, burnup1 in enumerate(depl_set):
                        if bu < 15:
                            y_batch_init_temp[bu].append(numpy.array([burnup1[0][1] / 1000]))
                else:
                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.amax(numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten())
                        y_batch_init_temp[bu].append(numpy.array([y_batch_init]))
            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp


def prepare_ixs_bu_node_matrix_24_3d_conv_max_local(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24*2):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4):int(axial_numb/4)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2):int(axial_numb/2)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4)+1:int(axial_numb*3/4)+2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[optimize_id][2:10, 2:10, :]).flatten()
                    y_batch_init_max = numpy.amax(numpy.array(burnup1[optimize_id][2:10, 2:10, :]).flatten())
                    y_batch_init_temp[bu*2].append(y_batch_init)
                    y_batch_init_temp[bu*2+1].append(numpy.array([y_batch_init_max]))

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def prepare_ixs_bu_node_matrix_1_3d_conv_pin(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):

    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))
    ffl = FFL("HIPER_U56_HANA6.FF", "/home/youndukn/Plants/db/tset/")
    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []


    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4):int(axial_numb/4)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2):int(axial_numb/2)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4)+1:int(axial_numb*3/4)+2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)
                s_batch_init_temp.append(s_batch_init1)
                
                s_batch_init2 = numpy.zeros((8,8,1,64))
                burnup_id = burnup_boc[8][2:10, 2:10, :]
                for row_i, row in enumerate(burnup_id):
                    for col_i, col in enumerate(row):
                        if isInt(col[1]):
                            ahole = ffl.ff["HIPER_X"+str(col[1])+"C"][0,0,:8,:8]
                        elif "c" == str(col[1]).lower():
                            ahole = ffl.ff["HIPER_220C"][0, 0, :8, :8]
                        else:
                            ahole = numpy.zeros((64))
                        ahole = ahole.flatten()
                        s_batch_init2[row_i,col_i,0,:] = ahole[:]

                s_batch_init_temp1.append(s_batch_init2)

                y_batch_init = numpy.array(burnup_boc[optimize_id][2:10, 2:10, :]).flatten()
                y_batch_init_temp[0].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp


def prepare_ixs_bu_node_matrix_24_3d_conv_pin(optimize_id,
                                             input_id,
                                             model,
                                             include=["y3"],
                                             exclude=["y314"],
                                             folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))
    ffl = FFL("HIPER_U56_HANA6.FF", "/home/youndukn/Plants/db/tset/")
    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(1):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)
                s_batch_init_temp.append(s_batch_init1)

                s_batch_init2 = numpy.zeros((8, 8, 1, 64))
                burnup_id = burnup_boc[8][2:10, 2:10, :]
                for row_i, row in enumerate(burnup_id):
                    for col_i, col in enumerate(row):
                        if isInt(col[1]):
                            ahole = ffl.ff["HIPER_X" + str(col[1]) + "C"][0, 0, :8, :8]
                        elif "c" == str(col[1]).lower():
                            ahole = ffl.ff["HIPER_220C"][0, 0, :8, :8]
                        else:
                            ahole = numpy.zeros((64))
                        ahole = ahole.flatten()
                        s_batch_init2[row_i, col_i, 0, :] = ahole[:]

                s_batch_init_temp1.append(s_batch_init2)

                y_batch_init = numpy.array(burnup_boc[optimize_id][2:10, 2:10, :]).flatten()
                y_batch_init_temp[0].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    # y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp


def prepare_ixs_bu_node_matrix_24_3d_conv_pd(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4):int(axial_numb/4)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2):int(axial_numb/2)+1, :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4)+1:int(axial_numb*3/4)+2, :]), axis=-2)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    values = numpy.zeros((8, 8, 1))
                    a = burnup1[2].reshape((10, 10))
                    values[:, :, 0] = a[:8, :8]
                    values[:, 0, 0] = values[0, :, 0]

                    second = np.rot90(values)
                    first = np.rot90(second)
                    third = np.delete(np.rot90(first), 0, 0)
                    fourth = np.delete(values, 0, 0)

                    second = np.array(second, dtype=float)
                    first = np.array(first, dtype=float)
                    third = np.array(third, dtype=float)
                    fourth = np.array(fourth, dtype=float)

                    half1 = np.concatenate((first, third))
                    half2 = np.concatenate((second, fourth))

                    full_matrix = np.concatenate((half1, half2), axis=1)
                    full_matrix = np.delete(full_matrix, [7, ], axis=1)
                    full_matrix = full_matrix.flatten()
                    y_batch_init_temp[bu].append(full_matrix)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp, y_batch_init_temp

def prepare_ixs_bu_node_matrix_24_3d_pd(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])

    axial_numb = 26
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[14]
                s_batch_init1 = numpy.concatenate(
                    (s_batch_init[2:-2, 2:-2, int(axial_numb/4), :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb/2), :],
                     s_batch_init[2:-2, 2:-2, int(axial_numb*3/4), :]), axis=-1)
                s_batch_init1 = numpy.array(s_batch_init1)

                s_batch_init_temp.append(s_batch_init1)

                for bu, burnup1 in enumerate(depl_set):
                    values = numpy.zeros((8, 8, 1))
                    a = burnup1[2].reshape((10, 10))
                    values[:,:,0] = a[:8,:8]
                    values[:,0,0] = values[0,:,0]

                    second = np.rot90(values)
                    first = np.rot90(second)
                    third = np.delete(np.rot90(first), 0, 0)
                    fourth = np.delete(values, 0, 0)

                    second = np.array(second, dtype=float)
                    first = np.array(first, dtype=float)
                    third = np.array(third, dtype=float)
                    fourth = np.array(fourth, dtype=float)

                    half1 = np.concatenate((first, third))
                    half2 = np.concatenate((second, fourth))

                    full_matrix = np.concatenate((half1, half2), axis=1)
                    full_matrix = np.delete(full_matrix, [7, ], axis=1)
                    full_matrix = full_matrix.flatten()
                    y_batch_init_temp[bu].append(full_matrix)


            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  y_batch_init_temp

def prepare_ixs_bu_node_matrix_24_merged(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])
    numb_type = 0
    keys = []
    output_keys = []
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[input_id]
                s_batch_init = s_batch_init[2:-2, 2:-2, :]
                s_batch_init = numpy.array(s_batch_init)
                s_batch_init_temp.append(s_batch_init)

                #burnup
                #s_batch_init_1 = burnup_boc[7]
                """
                if numb_type == 0:
                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0]+col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)
                """
                s_batch_init1 = numpy.zeros((17, 17, numb_type), dtype=numpy.int8)
                """
                burnup_id = burnup_boc[8][1:-1, 1:-1, :]
                for row_i, row in enumerate(burnup_id):
                    for col_i, col in enumerate(row):
                        the_index = keys.index(col[0]+col[1])
                        s_batch_init1[row_i][col_i][the_index] = 1
                """
                s_batch_init_temp1.append(s_batch_init1)
                """
                for bu, burnup1 in enumerate(depl_set):
                    values = numpy.zeros((8, 8, 1))
                    a = burnup1[2].reshape((10, 10))
                    values[:,:, 0] = a[:8,:8]

                    second = np.rot90(values)
                    first = np.rot90(second)
                    third = np.delete(np.rot90(first), 0, 0)
                    fourth = np.delete(values, 0, 0)

                    second = np.array(second, dtype=float)
                    first = np.array(first, dtype=float)
                    third = np.array(third, dtype=float)
                    fourth = np.array(fourth, dtype=float)

                    half1 = np.concatenate((first, third))
                    half2 = np.concatenate((second, fourth))

                    full_matrix = np.concatenate((half1, half2), axis=1)
                    full_matrix = np.delete(full_matrix, [7, ], axis=1)
                    full_matrix = full_matrix.flatten()
                    y_batch_init_temp[bu].append(full_matrix)
                """

                y_batch_init1 = numpy.zeros((15, 15, 1))
                """
                burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                for row_i, row in enumerate(burnup_id):
                    for col_i, col in enumerate(row):
                        the_index = output_keys.index(str(col[0]))
                        multiplier = 0
                        if the_index == 1:
                            multiplier = 0.5
                        elif the_index == 2:
                            multiplier = 0.8
                        elif the_index == 3:
                            multiplier = 1.

                        y_batch_init1[row_i][col_i][0] = multiplier
                """
                y_batch_init1 = y_batch_init1.flatten()

                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[10][2:-2, 2:-2, :]).flatten()
                    y_batch_init = numpy.concatenate((y_batch_init, y_batch_init1))
                    y_batch_init_temp[bu].append(y_batch_init)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    s_batch_init_temp1 = numpy.array(s_batch_init_temp1, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  s_batch_init_temp1, y_batch_init_temp, numb_type

def prepare_ixs_bu_node_matrix_24_merged_pd(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    for i in range(24):
        y_batch_init_temp.append([])
    numb_type = 0
    keys = []
    output_keys = []
    while len(files) > 0:
        try:

            file_index = random.randrange(len(files))
            file_read = files[file_index]

            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1
                burnup_boc = depl_set[0]
                s_batch_init = burnup_boc[input_id]
                s_batch_init = s_batch_init[2:-2, 2:-2, :]
                s_batch_init = numpy.array(s_batch_init)
                s_batch_init_temp.append(s_batch_init)

                #burnup
                #s_batch_init_1 = burnup_boc[7]

                if numb_type == 0:
                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0]+col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                s_batch_init1 = numpy.zeros((17, 17, numb_type), dtype=numpy.int8)
                burnup_id = burnup_boc[8][1:-1, 1:-1, :]
                for row_i, row in enumerate(burnup_id):
                    for col_i, col in enumerate(row):
                        the_index = keys.index(col[0]+col[1])
                        s_batch_init1[row_i][col_i][the_index] = 1

                s_batch_init_temp1.append(s_batch_init1)



                y_batch_init1 = numpy.zeros((15, 15, 1))
                burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                for row_i, row in enumerate(burnup_id):
                    for col_i, col in enumerate(row):
                        the_index = output_keys.index(str(col[0]))
                        multiplier = 0
                        if the_index == 1:
                            multiplier = 0.5
                        elif the_index == 2:
                            multiplier = 0.8
                        elif the_index == 3:
                            multiplier = 1.

                        y_batch_init1[row_i][col_i][0] = multiplier
                y_batch_init1 = y_batch_init1.flatten()

                for bu, burnup1 in enumerate(depl_set):
                    values = numpy.zeros((8, 8, 1))
                    a = burnup1[2].reshape((10, 10))
                    values[:,:, 0] = a[:8,:8]

                    second = np.rot90(values)
                    first = np.rot90(second)
                    third = np.delete(np.rot90(first), 0, 0)
                    fourth = np.delete(values, 0, 0)

                    second = np.array(second, dtype=float)
                    first = np.array(first, dtype=float)
                    third = np.array(third, dtype=float)
                    fourth = np.array(fourth, dtype=float)

                    half1 = np.concatenate((first, third))
                    half2 = np.concatenate((second, fourth))

                    full_matrix = np.concatenate((half1, half2), axis=1)
                    full_matrix = np.delete(full_matrix, [7, ], axis=1)
                    full_matrix = full_matrix.flatten()
                    full_matrix = numpy.concatenate((full_matrix, y_batch_init1))

                    y_batch_init_temp[bu].append(full_matrix)

                """
                for bu, burnup1 in enumerate(depl_set):
                    y_batch_init = numpy.array(burnup1[9][2:-2, 2:-2, :]).flatten()
                    y_batch_init_temp[bu].append(y_batch_init)
                """

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()


        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass

    s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
    s_batch_init_temp1 = numpy.array(s_batch_init_temp1, dtype=np.float16)

    """
    for i in range(24):
        y_batch_init_temp[i] = numpy.array(y_batch_init_temp[i], dtype=np.float16)
    """
    #y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)
    return s_batch_init_temp,  s_batch_init_temp1, y_batch_init_temp, numb_type

def train_ixs_fully(optimize_id,
              input_id,
              model,
              include = ["y3"],
              exclude = ["y314"],
              folders=["/media/youndukn/lastra/plants_data/"],
              den=False):

    files = get_files_with(include, exclude, "_s", folders)
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


            for num_batch in range(0, 20):
                depl_set = pickle.load(file_read)
                counter += 1

                burnup_boc = depl_set[0]
                burnup_eoc = depl_set[-1]
                s_batch_init = burnup_boc[input_id]

                temp_array = []

                for ab in ab_si:
                    temp_array += s_batch_init[ab[1], ab[2]]
                for ab in ab_in:
                    temp_array += s_batch_init[ab[1], ab[2]]

                s_batch_init = numpy.array(temp_array)

                s_batch_init_temp.append(s_batch_init)

                if optimize_id == cl:
                    y_batch_init = burnup_eoc[0][0]
                    y_batch_init_temp.append(y_batch_init)
                elif optimize_id == fxy:
                    max_fxy = 0.0

                    for burnup in depl_set:
                        if burnup[0][5] >max_fxy:
                            max_fxy = burnup[0][5]

                    y_batch_init_temp.append(max_fxy/2.0)

                elif optimize_id == both:
                    cycle_length = (burnup_eoc[0][0]-cl_rb)/cl_ra

                    max_fxy = 0.0
                    for burnup in depl_set:
                        if burnup[0][5] >max_fxy:
                            max_fxy = burnup[0][5]
                    max_fxy = (max_fxy-fxy_rb)/fxy_ra

                    y_batch_init_temp.append([cycle_length, max_fxy])

                elif optimize_id == pd:
                    y_batch_temp = []
                    for si in ab_si:
                        y_batch_temp.append(burnup_boc[2][si[0]])
                    for in_tmp in ab_in:
                        y_batch_temp.append(burnup_boc[2][in_tmp[0]])
                    y_batch_init_temp.append(y_batch_temp)

            sys.stdout.write("\rD%i" % counter)
            sys.stdout.flush()

            s_batch_init_temp = numpy.array(s_batch_init_temp, dtype=np.float16)
            y_batch_init_temp = numpy.array(y_batch_init_temp, dtype=np.float16)

            model.train_on_batch(x=s_batch_init_temp, y=y_batch_init_temp)

        except (AttributeError, EOFError, ImportError) as e:
            files.remove(file_read)
            file_read.close()
            pass


def print_all_weights_depl_fxy(index, model, include = ["y314"], exclude = [], folders = ["/media/youndukn/lastra/plants_data/"], den=False):

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
                    s_batch_init = burnup_boc[1][1:-1,1:-1,:]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    y_batch_init = burnup_eoc[0][0]

                    max_fxy = 0.0
                    # Clear gradients
                    for burnup in depl_set:
                        if burnup[0][5] > max_fxy:
                            max_fxy = burnup[0][5]

                    print(y_batch_init,
                          readout_t0[0][0][0]*18000.0,
                          (readout_t0[0][0][0]*18000.0 - y_batch_init),
                          100 * (readout_t0[0][0][0]*18000.0 - y_batch_init) / y_batch_init,
                          max_fxy, readout_t0[1][0][0]*2.0,
                          (readout_t0[1][0][0]*2.0 - max_fxy),
                          100 * (readout_t0[1][0][0]*2.0 - max_fxy) / max_fxy)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_all_weights_depl_fxy(index, model, include = ["y314"], exclude = [], folders = ["/media/youndukn/lastra/plants_data/"], den=False):

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
                    s_batch_init = burnup_boc[1][1:-1,1:-1,:]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    y_batch_init = burnup_eoc[0][0]

                    max_fxy = 0.0
                    # Clear gradients
                    for burnup in depl_set:
                        if burnup[0][5] > max_fxy:
                            max_fxy = burnup[0][5]

                    print(y_batch_init,
                          readout_t0[0][0][0]*18000.0,
                          (readout_t0[0][0][0]*18000.0 - y_batch_init),
                          100 * (readout_t0[0][0][0]*18000.0 - y_batch_init) / y_batch_init,
                          max_fxy, readout_t0[1][0][0]*2.0,
                          (readout_t0[1][0][0]*2.0 - max_fxy),
                          100 * (readout_t0[1][0][0]*2.0 - max_fxy) / max_fxy)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_node_matrix_24_3d_conv(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[2:-2, 2:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(24, dtype=int)
                    max_value = numpy.zeros(24, dtype=int)
                    max_max = numpy.zeros(24, dtype=int)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)

                    y_batch_init1 = numpy.zeros((15, 15, 1))
                    burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()
                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[9][2:-2, 2:-2, :]).flatten()
                        position = 0
                        for value, fresh in zip(y_batch_init, y_batch_init1):
                            if fresh:
                                diff = abs(readout_t0[bu][0][position] - value)
                                diff = int(diff * 1000)
                                maxv = int(value*10)
                                if max_max[bu] < maxv:
                                    max_max[bu] = maxv

                                if max_diff[bu] < diff:
                                    max_diff[bu] = diff
                                    max_value[bu] = maxv
                                    max_row[bu] = int(position / 15)
                                    max_col[bu] = position % 15
                            position += 1

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("dif", max_diff)
                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_node_matrix_24_3d_conv_quad(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[2:-2, 2:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(24, dtype=int)
                    max_value = numpy.zeros(24, dtype=int)
                    max_max = numpy.zeros(24, dtype=int)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)

                    y_batch_init1 = numpy.zeros((8, 8, 1))
                    burnup_id = burnup_boc[8][2:10, 2:10, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()
                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[optimize_id][2:10, 2:10, :]).flatten()
                        position = 0
                        for value, fresh in zip(y_batch_init, y_batch_init1):
                            if fresh:
                                diff = abs(readout_t0[bu][0][position] - value)
                                diff = int(diff * 1000)
                                maxv = int(value*10)
                                if max_max[bu] < maxv:
                                    max_max[bu] = maxv

                                if max_diff[bu] < diff:
                                    max_diff[bu] = diff
                                    max_value[bu] = maxv
                                    max_row[bu] = int(position / 15)
                                    max_col[bu] = position % 15
                            position += 1

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("dif", max_diff)
                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_node_matrix_24_3d_conv_iquad(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[-19:-2, -19:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[-19:-2, -19:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[-19:-2, -19:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(24, dtype=int)
                    max_value = numpy.zeros(24, dtype=int)
                    max_max = numpy.zeros(24, dtype=int)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)

                    y_batch_init1 = numpy.zeros((8, 8, 1))
                    burnup_id = burnup_boc[8][-10:-2, -10:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()
                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[optimize_id][-10:-2, -10:-2, :]).flatten()
                        position = 0
                        for value, fresh in zip(y_batch_init, y_batch_init1):
                            if fresh:
                                diff = abs(readout_t0[bu][0][position] - value)
                                diff = int(diff * 1000)
                                maxv = int(value*10)
                                if max_max[bu] < maxv:
                                    max_max[bu] = maxv

                                if max_diff[bu] < diff:
                                    max_diff[bu] = diff
                                    max_value[bu] = maxv
                                    max_row[bu] = int(position / 15)
                                    max_col[bu] = position % 15
                            position += 1

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("dif", max_diff)
                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_ixs_node_matrix_24_3d_conv_max_iquad(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[-19:-2, -19:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[-19:-2, -19:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[-19:-2, -19:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(24, dtype=int)
                    max_value = numpy.zeros(24, dtype=int)
                    max_max = numpy.zeros(24, dtype=int)

                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[optimize_id][-10:-2,-10:-2, :]).flatten()
                        y_batch_init_value = numpy.amax(y_batch_init)

                        diff = abs(readout_t0[bu][0][0] - y_batch_init_value)
                        diff = int(diff * 1000)
                        maxv = int(y_batch_init_value * 10)
                        if max_max[bu] < maxv:
                            max_max[bu] = maxv

                        if max_diff[bu] < diff:
                            max_diff[bu] = diff
                            max_value[bu] = y_batch_init_value

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("dif", max_diff)
                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_node_matrix_24_3d_conv_max_iquad_all(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                maximum = 0
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.array(s_batch_init[-19:-2, -19:-2, 6:20, :])
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()

                    y_batch_init_temp1 = []
                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[13][-10:-2, -10:-2, 6:20, :]).flatten()
                        y_batch_init_temp1.append(y_batch_init)
                    y_batch_init_temp1 = numpy.array(y_batch_init_temp1)
                    y_batch_init_temp1 = numpy.max(y_batch_init_temp1, axis=0)
                    maxi_index = numpy.argmax(y_batch_init_temp1)
                    diff = readout_t0[0,:] - y_batch_init_temp1
                    diff2 = abs(readout_t0[0, maxi_index] - y_batch_init_temp1[maxi_index])

                    print("dif {:0.2f} {:0.2f} {:0.2f} {:0.2f} ".format(diff2, y_batch_init_temp1[maxi_index], numpy.amin(diff), numpy.amax(diff)))

                    if maximum < diff2:
                        maximum =diff2
                        print(diff2, y_batch_init_temp1[maxi_index])
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_node_matrix_24_2d_conv_max_iquad(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[6]
                    s_batch_init1 = numpy.array(s_batch_init[-19:-2, -19:-2, :])
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(24, dtype=int)
                    max_value = numpy.zeros(24, dtype=int)
                    max_max = numpy.zeros(24, dtype=int)

                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[optimize_id][-10:-2,-10:-2, :]).flatten()
                        y_batch_init_value = numpy.amax(y_batch_init)

                        diff = abs(readout_t0[bu][0][0] - y_batch_init_value)
                        diff = int(diff * 1000)
                        maxv = int(y_batch_init_value * 10)
                        if max_max[bu] < maxv:
                            max_max[bu] = maxv

                        if max_diff[bu] < diff:
                            max_diff[bu] = diff
                            max_value[bu] = y_batch_init_value

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("dif", max_diff)
                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_ixs_node_matrix_24_3d_conv_max(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[2:-2, 2:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(24, dtype=int)
                    max_value = numpy.zeros(24, dtype=int)
                    max_max = numpy.zeros(24, dtype=int)

                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[optimize_id][2:10, 2:10, :]).flatten()
                        y_batch_init_value = numpy.amax(y_batch_init)

                        diff = abs(readout_t0[bu][0][0] - y_batch_init_value)
                        diff = int(diff * 1000)
                        maxv = int(y_batch_init_value * 10)
                        if max_max[bu] < maxv:
                            max_max[bu] = maxv

                        if max_diff[bu] < diff:
                            max_diff[bu] = diff
                            max_value[bu] = y_batch_init_value

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("dif", max_diff)
                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_ixs_node_matrix_1_3d_conv_pin(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    ffl = FFL("HIPER_U56_HANA6.FF", "/home/youndukn/Plants/db/tset/")
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[2:-2, 2:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)

                    s_batch_init2 = numpy.zeros((8, 8, 1, 64))
                    burnup_id = burnup_boc[8][2:10, 2:10, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            if isInt(col[1]):
                                ahole = ffl.ff["HIPER_X" + str(col[1]) + "C"][0, 0, :8, :8]
                            elif "c" == str(col[1]).lower():
                                ahole = ffl.ff["HIPER_220C"][0, 0, :8, :8]
                            else:
                                ahole = numpy.zeros((64))
                            ahole = ahole.flatten()
                            s_batch_init2[row_i, col_i, 0, :] = ahole[:]

                    s_batch_init2 = np.expand_dims(s_batch_init2, axis=0)
                    readout_t0 = model.predict([s_batch_init, s_batch_init2])

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(1, dtype=int)
                    max_value = numpy.zeros(1, dtype=int)
                    max_value_diff = numpy.zeros(1, dtype=int)
                    max_max = numpy.zeros(1, dtype=int)
                    max_value_diff_pred = numpy.zeros(1, dtype=int)
                    max_max_pred = numpy.zeros(1, dtype=int)
                    max_row = numpy.zeros(1)
                    max_col = numpy.zeros(1)

                    y_batch_init1 = numpy.zeros((15, 15, 1))
                    burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()

                    y_batch_init = numpy.array(burnup_boc[optimize_id][2:10, 2:10, :]).flatten()
                    position = 0
                    for value, fresh in zip(y_batch_init, y_batch_init1):
                        if fresh:
                            diff = abs(readout_t0[0][position] - value)
                            diff = int(diff * 1000)
                            maxv = int(value * 1000)

                            if max_diff[0] < diff:
                                max_diff[0] = diff
                                max_value[0] = maxv
                                max_row[0] = int(position / 15)
                                max_col[0] = position % 15

                        position += 1
                    position = 0
                    for value in y_batch_init:
                        diff = abs(readout_t0[0][position] - value)
                        diff = int(diff * 1000)
                        maxv = int(value * 1000)
                        if max_max[0] < maxv:
                            max_max[0] = maxv
                            max_value_diff[0] = diff

                        maxv_pred = int(readout_t0[0][position] * 1000)
                        if max_max_pred[0] < maxv_pred:
                            max_max_pred[0] = maxv_pred

                        position += 1

                    float_formatter = lambda x : "%.3f" % x
                    numpy.set_printoptions(formatter={'float_kind':float_formatter})

                    print(numpy.array(readout_t0[0]).reshape((8,8)))
                    print(y_batch_init.reshape((8,8)))
                    print("dif", max_diff, max_max, max_value_diff, max_max_pred, max_max-max_max_pred)

                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                    for value_i, value in enumerate(max_value_diff):
                        max_sum[value_i+1] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_node_matrix_1_3d_conv_wopin(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[2:-2, 2:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)

                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_diff = numpy.zeros(1, dtype=int)
                    max_value = numpy.zeros(1, dtype=int)
                    max_value_diff = numpy.zeros(1, dtype=int)
                    max_max = numpy.zeros(1, dtype=int)
                    max_row = numpy.zeros(1)
                    max_col = numpy.zeros(1)

                    y_batch_init1 = numpy.zeros((15, 15, 1))
                    burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()

                    y_batch_init = numpy.array(burnup_boc[optimize_id][2:10, 2:10, :]).flatten()
                    position = 0
                    for value, fresh in zip(y_batch_init, y_batch_init1):
                        if fresh:
                            diff = abs(readout_t0[0][position] - value)
                            diff = int(diff * 1000)
                            maxv = int(value*10)
                            if max_max[0] < maxv:
                                max_max[0] = maxv
                                max_value_diff[0] = diff

                            if max_diff[0] < diff:
                                max_diff[0] = diff
                                max_value[0] = maxv
                                max_row[0] = int(position / 15)
                                max_col[0] = position % 15
                        position += 1

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("dif", max_diff, max_max, max_value_diff)
                    #print("val", max_value)
                    #print("max", max_max)
                    for value_i, value in enumerate(max_diff):
                        max_sum[value_i] += value / 200
                    for value_i, value in enumerate(max_value_diff):
                        max_sum[value_i+1] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_ixs_node_matrix_24_3d_conv_pd(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    axial_numb = 26
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):
                    keys = []
                    output_keys = []
                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[14]
                    s_batch_init1 = numpy.concatenate(
                        (s_batch_init[2:-2, 2:-2, int(axial_numb / 4):int(axial_numb / 4) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb / 2):int(axial_numb / 2) + 1, :],
                         s_batch_init[2:-2, 2:-2, int(axial_numb * 3 / 4) + 1:int(axial_numb * 3 / 4) + 2, :]), axis=-2)
                    s_batch_init1 = numpy.array(s_batch_init1)
                    s_batch_init = np.expand_dims(s_batch_init1, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    type_dictionary = {}
                    output_dictionary = {}
                    for row in burnup_boc[8]:
                        for col in row:
                            type_dictionary[str(col[0] + col[1])] = 1
                            output_dictionary[str(col[0])] = 1
                    for an_object in type_dictionary.keys():
                        keys.append(an_object)
                    for an_object in output_dictionary.keys():
                        output_keys.append(an_object)
                    keys.sort()
                    output_keys.sort()
                    numb_type = len(keys)

                    max_value = numpy.zeros(24, dtype=int)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)

                    y_batch_init1 = numpy.zeros((15, 15, 1))
                    burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()

                    for bu, burnup1 in enumerate(depl_set):
                        values = numpy.zeros((8, 8, 1))
                        a = burnup1[2].reshape((10, 10))
                        values[:, :, 0] = a[:8, :8]

                        second = np.rot90(values)
                        first = np.rot90(second)
                        third = np.delete(np.rot90(first), 0, 0)
                        fourth = np.delete(values, 0, 0)

                        second = np.array(second, dtype=float)
                        first = np.array(first, dtype=float)
                        third = np.array(third, dtype=float)
                        fourth = np.array(fourth, dtype=float)

                        half1 = np.concatenate((first, third))
                        half2 = np.concatenate((second, fourth))

                        full_matrix = np.concatenate((half1, half2), axis=1)
                        full_matrix = np.delete(full_matrix, [7, ], axis=1)
                        full_matrix = full_matrix.flatten()
                        position = 0
                        for value, fresh in zip(full_matrix, y_batch_init1):
                            if fresh:
                                diff = abs(readout_t0[bu][0][position] - value)
                                diff = int(diff * 1000)
                                if max_value[bu] < diff:
                                    max_value[bu] = diff
                                    max_row[bu] = int(position / 15)
                                    max_col[bu] = position % 15
                            position += 1

                    # float_formatter = lambda x : "%.2f" % x
                    # numpy.set_printoptions(threshold=10)
                    print("max", max_value)
                    for value_i, value in enumerate(max_value):
                        max_sum[value_i] += value / 200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs(optimize_id,
              input_id,
              model,
              include = ["y314"],
              exclude = [],
              folders = ["/media/youndukn/lastra/plants_data/"],
              den=False):

    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                average_abs_diff_cl = 0
                average_abs_diff_fxy = 0

                for indexed_image in range(200):

                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = np.delete(burnup_boc[input_id], 9, 1)
                    s_batch_init = np.insert(s_batch_init, 18, 0.0, 1)

                    s_batch_init = s_batch_init[1:-1, 1:-1, :]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    if optimize_id == cl:
                        y_batch_init = burnup_eoc[0][0]
                        predict_cl = readout_t0[0][0] * cl_ra + cl_rb
                        average_abs_diff_cl += abs(100 * (predict_cl - y_batch_init) / y_batch_init)

                        print(("{:1.4f} "*4).format(y_batch_init,
                                                    predict_cl,
                              (predict_cl - y_batch_init),
                              100 * (predict_cl - y_batch_init) / y_batch_init))

                    elif optimize_id == fxy:

                        y_batch_init = 0.0
                        # Clear gradients
                        for burnup in depl_set:
                            if burnup[0][5] > y_batch_init:
                                y_batch_init = burnup[0][5]

                        predict_fxy = readout_t0[0][0]*fxy_ra+fxy_rb

                        average_abs_diff_fxy += abs(100 * (predict_fxy - y_batch_init) / y_batch_init)

                        print(("{:1.4f} "*4).format(y_batch_init,
                                                    predict_fxy,
                              (predict_fxy - y_batch_init),
                                  100 * (predict_fxy - y_batch_init) / y_batch_init))
                    elif optimize_id == both:
                        y_batch_init = burnup_eoc[0][0]

                        y_batch_init_1 = 0.0
                        # Clear gradients
                        for burnup in depl_set:
                            if burnup[0][5] > y_batch_init_1:
                                y_batch_init_1 = burnup[0][5]

                        predict_cl =readout_t0[0][0]*cl_ra+cl_rb
                        predict_fxy = readout_t0[0][1]*fxy_ra+fxy_rb

                        average_abs_diff_cl += abs(100 * (predict_cl - y_batch_init) / y_batch_init)
                        average_abs_diff_fxy += abs(100 * (predict_fxy - y_batch_init_1) / y_batch_init_1)

                        print(("{:1.4f} "*4).format(y_batch_init,
                                                    predict_cl,
                              (predict_cl - y_batch_init),
                              100 * (predict_cl - y_batch_init) / y_batch_init),
                              ("{:1.4f} " * 4).format(y_batch_init_1,
                                                      predict_fxy,
                                                      (predict_fxy - y_batch_init_1),
                                                      100 * (predict_fxy - y_batch_init_1) / y_batch_init_1)
                              )

                    elif optimize_id == pd:
                        max_p = 0
                        max_i = 0
                        average = 0
                        diff_abs_avg = 0
                        diff_avg = 0
                        added_number = 0
                        max_p_diff = 0
                        y_batch_temp = []
                        for si in ab_si:
                            y_batch_temp.append(burnup_boc[2][si[0]])
                        for in_tmp in ab_in:
                            y_batch_temp.append(burnup_boc[2][in_tmp[0]])
                        for i, p in enumerate(y_batch_temp):
                            if p != 0:
                                added_number+=1
                                diff_abs_avg += abs(p - readout_t0[0][0][i])
                                diff_avg +=  abs(p - readout_t0[0][0][i])/p*100

                                if max_p_diff < abs(p - readout_t0[0][0][i]):
                                    max_p_diff = abs(p - readout_t0[0][0][i])

                            average = average+p
                            if p > max_p:
                                max_p = p
                                max_i = i

                        average /= 100
                        diff_avg /= added_number
                        diff_abs_avg /= added_number

                        y_batch_init = max_p

                        print(("{:1.4f} "*7).format(y_batch_init,
                              readout_t0[0][0][max_i],
                              (readout_t0[0][0][max_i] - y_batch_init),
                              100 * (readout_t0[0][0][max_i]- y_batch_init) / y_batch_init, diff_abs_avg, diff_avg, max_p_diff))



            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass
def print_ixs_node(optimize_id,
              input_id,
              model,
              include = ["y314"],
              exclude = [],
              folders = ["/media/youndukn/lastra/plants_data/"],
              den=False):

    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)

                while True:

                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[input_id]
                    s_batch_init = s_batch_init[2:-2, 2:-2, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init,axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    max_p = 0
                    max_i = 0
                    average = 0
                    diff_abs_avg = 0
                    diff_avg = 0
                    added_number = 0
                    max_p_diff = 0
                    max_p_diff_percent = 0
                    y_batch_temp = []
                    y_batch_init_temp = []
                    for burnup1 in depl_set:

                        y_batch_temp = []
                        for si in ab_si:
                            y_batch_temp.append(burnup1[2][si[0]])
                        for in_tmp in ab_in:
                            y_batch_temp.append(burnup1[2][in_tmp[0]])
                    y_batch_init_temp.append(y_batch_temp)

                    for i ,p in enumerate(y_batch_init_temp):
                        added_number += 1
                        print(i, p)
                    for i, p in enumerate(readout_t0):
                        added_number += 1
                        print(i, p)
                        """
                        if p != 0:
                            added_number+=1
                            diff_abs_avg += abs(p[i] - readout_t0[0][i])
                            diff_avg +=  abs(p[i] - readout_t0[0][i])/p[i]*100

                            if max_p_diff < abs(p[i] - readout_t0[0][i]):
                                    max_p_diff = abs(p[i] - readout_t0[0][i])
                                    max_p_diff_percent = abs(p[i] - readout_t0[0][i])/p[i]*100

                        average = average+p
                        if p > max_p:
                            max_p = p
                            max_i = i

                            #print(("{:1.4f} "*4).format(i,p,readout_t0[0][i],p-readout_t0[0][i]))

                        average /= 100
                        diff_avg /= added_number
                        diff_abs_avg /= added_number#

                        y_batch_init = max_p
                        if max_p < 1.4:
                            print(("{:1.4f} "*5).format(y_batch_init,
                                  readout_t0[0][max_i],
                                  abs(readout_t0[0][max_i] - y_batch_init)/readout_t0[0][max_i]*100,
                                  max_p_diff,max_p_diff_percent))
"""

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                #print(e)
                pass


def print_ixs_node_matrix(optimize_id,
              input_id,
              model,
              include = ["y314"],
              exclude = [],
              folders = ["/media/youndukn/lastra/plants_data/"],
              den=False):

    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)

                for _ in range(200):

                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[input_id]
                    s_batch_init = s_batch_init[2:-2, 2:-2, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init,axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    y_batch_init_temp = []

                    values = numpy.zeros((8, 8, 24))
                    for bu, burnup1 in enumerate(depl_set):
                        a = burnup1[2].reshape((10, 10))
                        values[:, :, bu] = a[:8, :8]

                    second = np.rot90(values)
                    first = np.rot90(second)
                    third = np.delete(np.rot90(first), 0, 0)
                    fourth = np.delete(values, 0, 0)

                    second = np.array(second, dtype=float)
                    first = np.array(first, dtype=float)
                    third = np.array(third, dtype=float)
                    fourth = np.array(fourth, dtype=float)

                    half1 = np.concatenate((first, third))
                    half2 = np.concatenate((second, fourth))

                    full_matrix = np.concatenate((half1, half2), axis=1)
                    full_matrix = np.delete(full_matrix, [7, ], axis=1)
                    full_matrix = full_matrix.flatten()

                    max_value = numpy.zeros(24)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)
                    for i, value in enumerate(full_matrix):
                        position = i%225
                        depl_point = int(i/225)
                        diff = abs(readout_t0[0][i]- value)
                        if max_value[depl_point]< diff:
                            max_value[depl_point] = diff
                            max_row[depl_point] = int(position/15)
                            max_col[depl_point] = position%15

                    float_formatter = lambda x : "%.3f" % x
                    numpy.set_printoptions(formatter={'float_kind': float_formatter})
                    print(max_value)


            except (AttributeError, EOFError, ImportError, IndexError) as e:
                #print(e)
                pass

def print_ixs_node_matrix_24(optimize_id,
              input_id,
              model,
              include = ["y314"],
              exclude = [],
              folders = ["/media/youndukn/lastra/plants_data/"],
              den=False):

    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        for file_read in files:
            try:
                print(file_read.name)

                for _ in range(200):

                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[input_id]
                    s_batch_init = s_batch_init[2:-2, 2:-2, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init,axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    y_batch_init_temp = []
                    max_value = numpy.zeros(24, dtype=int)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)

                    for bu, burnup1 in enumerate(depl_set):
                        values = numpy.zeros((8, 8, 1))
                        a = burnup1[2].reshape((10, 10))
                        values[:, :, 0] = a[:8, :8]

                        second = np.rot90(values)
                        first = np.rot90(second)
                        third = np.delete(np.rot90(first), 0, 0)
                        fourth = np.delete(values, 0, 0)

                        second = np.array(second, dtype=float)
                        first = np.array(first, dtype=float)
                        third = np.array(third, dtype=float)
                        fourth = np.array(fourth, dtype=float)

                        half1 = np.concatenate((first, third))
                        half2 = np.concatenate((second, fourth))

                        full_matrix = np.concatenate((half1, half2), axis=1)
                        full_matrix = np.delete(full_matrix, [7, ], axis=1)
                        full_matrix = full_matrix.flatten()
                        for position, value in enumerate(full_matrix):
                            diff = abs(readout_t0[bu][0][position] - value)
                            diff = int(diff*1000)
                            if max_value[bu]< diff:
                                max_value[bu] = diff
                                max_row[bu] = int(position/15)
                                max_col[bu] = position%15

                    #float_formatter = lambda x : "%.2f" % x
                    #numpy.set_printoptions(threshold=10)
                    print(max_value)


            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def print_ixs_node_matrix_24_merged(optimize_id,
              input_id,
              model,
              include = ["y314"],
              exclude = [],
              folders = ["/media/youndukn/lastra/plants_data/"],
              den=False):

    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        keys = []
        output_keys = []
        numb_type = 0
        for file_read in files:
            try:
                print(file_read.name)
                max_sum = [0]*24
                for _ in range(200):

                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[input_id]
                    s_batch_init = s_batch_init[2:-2, 2:-2, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init,axis=0)

                    if numb_type == 0:
                        type_dictionary = {}
                        output_dictionary = {}
                        for row in burnup_boc[8]:
                            for col in row:
                                type_dictionary[str(col[0] + col[1])] = 1
                                output_dictionary[str(col[0])] = 1
                        for an_object in type_dictionary.keys():
                            keys.append(an_object)
                        for an_object in output_dictionary.keys():
                            output_keys.append(an_object)
                        keys.sort()
                        output_keys.sort()
                        numb_type = len(keys)

                    s_batch_init1 = numpy.zeros((1, 17, 17, numb_type), dtype=numpy.int8)
                    burnup_id = burnup_boc[8][1:-1, 1:-1, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = keys.index(col[0] + col[1])
                            s_batch_init1[0][row_i][col_i][the_index] = 1

                    readout_t0 = model.predict([s_batch_init, s_batch_init1])

                    y_batch_init_temp = []
                    max_value = numpy.zeros(24, dtype=int)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)

                    y_batch_init1 = numpy.zeros((15, 15, 1))
                    burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()
                    for bu, burnup1 in enumerate(depl_set):
                        y_batch_init = numpy.array(burnup1[9][2:-2, 2:-2, :]).flatten()
                        position = 0
                        for value, fresh in zip(y_batch_init, y_batch_init1):
                            if fresh:
                                diff = abs(readout_t0[bu][0][position] - value)
                                diff = int(diff*1000)
                                if max_value[bu]< diff:
                                    max_value[bu] = diff
                                    max_row[bu] = int(position/15)
                                    max_col[bu] = position%15
                            position += 1
                    """
                    for bu, burnup1 in enumerate(depl_set):
                        values = numpy.zeros((8, 8, 1))
                        a = burnup1[2].reshape((10, 10))
                        values[:, :, 0] = a[:8, :8]

                        second = np.rot90(values)
                        first = np.rot90(second)
                        third = np.delete(np.rot90(first), 0, 0)
                        fourth = np.delete(values, 0, 0)

                        second = np.array(second, dtype=float)
                        first = np.array(first, dtype=float)
                        third = np.array(third, dtype=float)
                        fourth = np.array(fourth, dtype=float)

                        half1 = np.concatenate((first, third))
                        half2 = np.concatenate((second, fourth))

                        full_matrix = np.concatenate((half1, half2), axis=1)
                        full_matrix = np.delete(full_matrix, [7, ], axis=1)
                        full_matrix = full_matrix.flatten()
                    """

                    #float_formatter = lambda x : "%.2f" % x
                    #numpy.set_printoptions(threshold=10)
                    print(max_value)
                    for value_i, value in enumerate(max_value):
                        max_sum[value_i] += value/200
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_node_matrix_24_merged_pd(optimize_id,
              input_id,
              model,
              include = ["y314"],
              exclude = [],
              folders = ["/media/youndukn/lastra/plants_data/"],
              den=False):

    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "", folders)
        counter = 0
        keys = []
        output_keys = []
        numb_type = 0
        for file_read in files:
            try:
                print(file_read.name)

                for _ in range(200):

                    depl_set = pickle.load(file_read)
                    counter += 1
                    burnup_boc = depl_set[0]
                    s_batch_init = burnup_boc[input_id]
                    s_batch_init = s_batch_init[2:-2, 2:-2, :]
                    s_batch_init = numpy.array(s_batch_init)
                    s_batch_init = np.expand_dims(s_batch_init,axis=0)

                    if numb_type == 0:
                        type_dictionary = {}
                        output_dictionary = {}
                        for row in burnup_boc[8]:
                            for col in row:
                                type_dictionary[str(col[0] + col[1])] = 1
                                output_dictionary[str(col[0])] = 1
                        for an_object in type_dictionary.keys():
                            keys.append(an_object)
                        for an_object in output_dictionary.keys():
                            output_keys.append(an_object)
                        keys.sort()
                        output_keys.sort()
                        numb_type = len(keys)

                    s_batch_init1 = numpy.zeros((1, 17, 17, numb_type), dtype=numpy.int8)
                    burnup_id = burnup_boc[8][1:-1, 1:-1, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = keys.index(col[0] + col[1])
                            s_batch_init1[0][row_i][col_i][the_index] = 1

                    readout_t0 = model.predict([s_batch_init, s_batch_init1])

                    y_batch_init_temp = []
                    max_value = numpy.zeros(24, dtype=int)
                    max_row = numpy.zeros(24)
                    max_col = numpy.zeros(24)

                    y_batch_init1 = numpy.zeros((15, 15, 1))
                    burnup_id = burnup_boc[8][2:-2, 2:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            the_index = output_keys.index(str(col[0]))
                            multiplier = 0
                            if the_index == 3:
                                multiplier = 1.

                            y_batch_init1[row_i][col_i][0] = multiplier
                    y_batch_init1 = y_batch_init1.flatten()

                    for bu, burnup1 in enumerate(depl_set):
                        values = numpy.zeros((8, 8, 1))
                        a = burnup1[2].reshape((10, 10))
                        values[:, :, 0] = a[:8, :8]

                        second = np.rot90(values)
                        first = np.rot90(second)
                        third = np.delete(np.rot90(first), 0, 0)
                        fourth = np.delete(values, 0, 0)

                        second = np.array(second, dtype=float)
                        first = np.array(first, dtype=float)
                        third = np.array(third, dtype=float)
                        fourth = np.array(fourth, dtype=float)

                        half1 = np.concatenate((first, third))
                        half2 = np.concatenate((second, fourth))

                        full_matrix = np.concatenate((half1, half2), axis=1)
                        full_matrix = np.delete(full_matrix, [7, ], axis=1)
                        full_matrix = full_matrix.flatten()
                        position = 0
                        for value, fresh in zip(full_matrix, y_batch_init1):
                            if fresh:
                                diff = abs(readout_t0[bu][0][position] - value)
                                diff = int(diff * 1000)
                                if max_value[bu] < diff:
                                    max_value[bu] = diff
                                    max_row[bu] = int(position / 15)
                                    max_col[bu] = position % 15
                            position += 1

                    #float_formatter = lambda x : "%.2f" % x
                    #numpy.set_printoptions(threshold=10)
                    print(max_value)


            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass


def print_ixs_fully(optimize_id,
              input_id,
              model,
              include = ["y314"],
              exclude = [],
              folders = ["/media/youndukn/lastra/plants_data/"],
              den=False):

    print()
    for x_man in range(1, 2):

        files = get_files_with(include, exclude, "_s", folders)

        for file_read in files:
            try:
                print(file_read.name)
                average_abs_diff_cl = 0
                average_abs_diff_fxy = 0

                for indexed_image in range(200):

                    depl_set = pickle.load(file_read)

                    burnup_boc = depl_set[0]
                    burnup_eoc = depl_set[-1]
                    s_batch_init = burnup_boc[input_id]

                    temp_array = []

                    for ab in ab_si:
                        temp_array += s_batch_init[ab[1], ab[2]]
                    for ab in ab_in:
                        temp_array += s_batch_init[ab[1], ab[2]]

                    s_batch_init = numpy.array(temp_array)

                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    if optimize_id == cl:
                        y_batch_init = burnup_eoc[0][0]

                        print(("{:1.4f} "*4).format(y_batch_init,
                              readout_t0[0][0],
                              (readout_t0[0][0] - y_batch_init),
                              100 * (readout_t0[0][0] - y_batch_init) / y_batch_init))

                    elif optimize_id == fxy:

                        y_batch_init = 0.0
                        # Clear gradients
                        for burnup in depl_set:
                            if burnup[0][5] > y_batch_init:
                                y_batch_init = burnup[0][5]

                        print(("{:1.4f} "*4).format(y_batch_init,
                              readout_t0[0][0],
                              (readout_t0[0][0] - y_batch_init),
                                  100 * (readout_t0[0][0] - y_batch_init) / y_batch_init))
                    elif optimize_id == both:
                        y_batch_init = burnup_eoc[0][0]

                        y_batch_init_1 = 0.0
                        # Clear gradients
                        for burnup in depl_set:
                            if burnup[0][5] > y_batch_init_1:
                                y_batch_init_1 = burnup[0][5]

                        predict_cl =readout_t0[0][0]*cl_ra+cl_rb
                        predict_fxy = readout_t0[0][1]*fxy_ra+fxy_rb

                        average_abs_diff_cl += abs(100 * (predict_cl - y_batch_init) / y_batch_init)
                        average_abs_diff_fxy += abs(100 * (predict_fxy - y_batch_init_1) / y_batch_init_1)

                        print(("{:1.4f} "*4).format(y_batch_init,
                                                    predict_cl,
                              (predict_cl - y_batch_init),
                              100 * (predict_cl - y_batch_init) / y_batch_init),
                              ("{:1.4f} " * 4).format(y_batch_init_1,
                                                      predict_fxy,
                                                      (predict_fxy - y_batch_init_1),
                                                      100 * (predict_fxy - y_batch_init_1) / y_batch_init_1)
                              )

                    elif optimize_id == pd:
                        max_p = 0
                        max_i = 0
                        average = 0
                        diff_abs_avg = 0
                        diff_avg = 0
                        added_number = 0
                        y_batch_temp = []
                        for si in ab_si:
                            y_batch_temp.append(burnup_boc[2][si[0]])
                        for in_tmp in ab_in:
                            y_batch_temp.append(burnup_boc[2][in_tmp[0]])
                        y_batch_init_temp.append(y_batch_temp)
                        for i, p in enumerate(y_batch_temp):
                            if p != 0:
                                added_number+=1
                                diff_abs_avg += abs(p - readout_t0[0][i])
                                diff_avg +=  abs(p - readout_t0[0][i])/p*100
                            average = average+p
                            if p > max_p:
                                max_p = p
                                max_i = i

                        average /= 100
                        diff_avg /= added_number
                        diff_abs_avg /= added_number

                        y_batch_init = max_p

                        print(("{:1.4f} "*6).format(y_batch_init,
                              readout_t0[0][max_i],
                              (readout_t0[0][max_i] - y_batch_init),
                              100 * (readout_t0[0][max_i]- y_batch_init) / y_batch_init, diff_abs_avg, diff_avg))

                print("Averages : ", average_abs_diff_cl/200, average_abs_diff_fxy/200)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
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
                    s_batch_init = burnup_boc[1][1:-1,1:-1,:]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    y_batch_init = burnup_eoc[0][0]

                    max_fxy = 0.0
                    # Clear gradients
                    for burnup in depl_set:
                        if burnup[0][5] > max_fxy:
                            max_fxy = burnup[0][5]

                    print(y_batch_init,
                          readout_t0[0][0]*18000.0,
                          (readout_t0[0][0]*18000.0 - y_batch_init),
                          100 * (readout_t0[0][0]*18000.0 - y_batch_init) / y_batch_init)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass



def print_all_weights_fxy(index, model, include = ["y314"], exclude = [], folders = ["/media/youndukn/lastra/plants_data/"], den=False):

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
                    s_batch_init = burnup_boc[1][1:-1,1:-1,:]
                    s_batch_init = numpy.array(s_batch_init)
                    if den==True:
                        s_batch_init_den = burnup_boc[4][1:-1,1:-1,:]
                        s_batch_init_den = numpy.array(s_batch_init_den)
                        s_batch_init = numpy.concatenate((s_batch_init, s_batch_init_den), axis=2)
                    s_batch_init = np.expand_dims(s_batch_init, axis=0)
                    readout_t0 = model.predict(s_batch_init)

                    y_batch_init = burnup_eoc[0][0]

                    max_fxy = 0.0
                    # Clear gradients
                    for burnup in depl_set:
                        if burnup[0][5] > max_fxy:
                            max_fxy = burnup[0][5]

                    print(max_fxy,
                          readout_t0[0][0]*2.0,
                          (readout_t0[0][0]*2.0 - max_fxy),
                          100 * (readout_t0[0][0]*2.0 - max_fxy) / max_fxy)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

"""
file_name ="01_y310_se_MD_BN_pd_lr_ND_karma"

print(file_name)

model, feature_model, output_model = ResNetI7_MD_BN((15, 15, 5), classes=100)

#shuffle_data([],
#                  ["_s"],
#                  ["/media/youndukn/lastra/idata2/"], )

model.compile(loss='mse', optimizer=Adam(lr=0.00001))
feature_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
output_model.compile(loss='mse', optimizer=Adam(lr=0.00001))

model.load_weights("{}_main.hdf5".format(file_name))

for ix in range(132, 163):
    train_ixs(pd,
              karma,
              model,
                   [],
                   [],
                   ["/media/youndukn/lastra/idata2/"],
                  False)
    print_ixs(pd,
              karma,
              model,
                           [],
                           ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                           ["/media/youndukn/lastra/idata2/"],
                          False)

model.save_weights("{}_main.hdf5".format(file_name))
feature_model.save_weights("{}_feature.hdf5".format(file_name))
output_model.save_weights("{}_output.hdf5".format(file_name))



file_name ="01_y310_se_MD_BN_pd_lr_ND_astra"

print(file_name)

model = ResNetI7_MD_BN((15, 15, 5), classes=100)

#shuffle_data([],
#                  ["_s"],
#                  ["/media/youndukn/lastra/idata1/"], )

model.compile(loss='mse', optimizer=Adam(lr=0.00001))


model.load_weights("{}.hdf5".format(file_name))
for ix in range(132, 163):
    train_ixs(pd,
              astra,
              model,
                   [],
                   [],
                   ["/media/youndukn/lastra/idata2/"],
                  False)
    print_ixs(pd,
              astra,
              model,
                           [],
                           ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                           ["/media/youndukn/lastra/idata2/"],
                          False)

model.save_weights("{}.hdf5".format(file_name))
"""

"""

file_name ="01_astra_u516_nep_depl_ram_aux_node.inp"

print(file_name)


#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_aux((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],
#                  ["G:/MyPRJ/idata/"], )

model.compile(loss='mse', optimizer=Adam(lr=0.00001))
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

model.save_weights("{}.hdf5".format(file_name))


s_batch_init_temp, y_batch_init_temp, y_batch_init_temp1= prepare_ixs_bu_node(pd,
                  karma_node,
                  model,
                  [],
                  [],
                  ["./idata/node/"])

for jx in range(8):
    for ix in range(30):
        model.fit(x=s_batch_init_temp,
                     y=[y_batch_init_temp, y_batch_init_temp1],
                     batch_size=10,
                     epochs=1,
                     validation_split=1000)


        print_ixs_node(pd,
                  karma_node,
                  model,
                  [],
                  ["data_1", "data_2", "data_3", "data_4"],
                  ["./idata/node/"],
                  False)

"""
"""
file_name ="01_astra_u516_nep_depl_ram_node_bu_at.inp"

print(file_name)

s_batch_init_temp, y_batch_init_temp= prepare_ixs_bu_node(pd,
                  karma_node,
                  None,
                  [],
                  [],
                  ["/media/youndukn/lastra/jin_data1/"])
print("Hello")
#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.000005))
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)


for jx in range(10):
    for ix in range(10):
        model.fit(x=s_batch_init_temp,
                     y=y_batch_init_temp,
                     batch_size=10,
                     epochs=5,
                     validation_split=1000)

        model.save_weights("{}.hdf5".format(file_name))
"""
"""
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)

print("Hello")
#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/jin_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()


print_ixs_node_matrix_24(pd,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/jin_data1/"],
                         False)


filepath="./check_p/weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=150,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""
#SIGMOID Function
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_SIG((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/jin_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()


print_ixs_node_matrix_24(pd,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/jin_data1/"],
                         False)


filepath="./check_p/sig_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=150,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""
#No Batch Norm Function
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_NBN((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/jin_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()


print_ixs_node_matrix_24(pd,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/jin_data1/"],
                         False)


filepath="./check_p/nbn_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""

"""
#No SE block Function
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_SE((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/jin_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()


#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/se_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""


"""

#No SKIP block Function
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_SKIP((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/jin_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()


#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/skip_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""
#Depth 20 block Function
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/jin_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()


#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/depth20_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""
#Input deck
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/jin_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()


#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/depth20_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2_sigmoid.inp"

print(file_name)

print("Hello")
#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_SIG((34, 34, 5), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],111

model.compile(loss='mse', optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

print_ixs_node_matrix_24(pd,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/jin_data1/"],
                         False)
"""


"""
for jx in range(20):
    for ix in range(10):


        model.save_weights("{}.hdf5".format(file_name))

        print_ixs_node_matrix_24(pd,
                       karma_node,
                       model,
                       ["data_0"],
                       [],
                       ["/media/youndukn/lastra/jin_data1/"],
                       False)
"""


"""

file_name ="01_y310_se_MD_BN_pd_lr_ND_karma"

print(file_name)


#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model= FullyConnected((140,), classes=100)
model.summary()
#shuffle_data([],
#                  ["_s"],
#                  ["/media/youndukn/lastra/idata2/"], )

model.compile(loss='mse', optimizer=Adam(lr=0.00001))
print_ixs_fully(pd,
          karma,
          model,
          [],
          ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
          ["/media/youndukn/lastra/idata2/"],
          False)

for jx in range(8):
    for ix in range(132, 163):
        train_ixs_fully(pd,
                  karma,
                  model,
                  [],
                  [],
                  ["/media/youndukn/lastra/idata2/"],
                  False)

        model.save_weights("{}.hdf5".format(file_name))

        print_ixs_fully(pd,
                  karma,
                  model,
                  [],
                  ["data_1", "data_2", "data_3", "data_4", "data_5", "data_6"],
                  ["/media/youndukn/lastra/idata2/"],
                  False)

"""
"""
file_name ="01_astra_u516_nep_depl.inp"

print(file_name)


#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model= FullyConnected((140,), classes=29)
model.summary()
#shuffle_data([],
#                  ["_s"],
#                  ["G:/MyPRJ/idata/"], )

model.compile(loss='mse', optimizer=Adam(lr=0.01))
#model.load_weights("{}.hdf5".format(file_name))

print_ixs(pd,
          karma,
          model,
          [],
          ["data_1", "data_2", "data_3", "data_4"],
          ["./idata/"],
          False)

for jx in range(8):
    for ix in range(132, 163):
        train_ixs_fully(pd,
                  karma,
                  model,
                  [],
                  [],
                  ["./idata/"],
                  False)

        model.save_weights("{}.hdf5".format(file_name))

        print_ixs_fully(pd,
                  karma,
                  model,
                  [],
                  ["data_1", "data_2", "data_3", "data_4"],
                  ["./idata/"],
                  False)
    """
"""
def custom_loss(y_true,y_pred):
    y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)*y_mult))

def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))


#Input deck
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp, numb_type = prepare_ixs_bu_node_matrix_24_merged(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/type_data1/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp1 = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()
    numb_type = 16

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_Merged((34, 34, 5), (17, 17, int(numb_type)), classes=29)
model.summary()


model.compile(loss=custom_loss, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



print_ixs_node_matrix_24_merged(pd,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/type_data1/"],
                         False)


filepath="./check_p/type_no_loss_fxy_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

model.fit(x=[s_batch_init_temp, s_batch_init_temp1],
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""
print("Fr")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2.inp"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp, numb_type = prepare_ixs_bu_node_matrix_24_merged(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_peak_data1/",
                                                                       "/media/youndukn/lastra/3d_peak_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp1 = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()
    numb_type = 16

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_Merged_1((34, 34, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/type_nmerged_nloss_fr_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
print(s_batch_init_temp1.shape)
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)

"""
"""
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

def custom_loss(y_true,y_pred):
    y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)*y_mult))

#Input deck
file_name ="01_astra_u516_nep_depl_ram_node_bu_at_matrix_2_pd.inp"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp, numb_type = prepare_ixs_bu_node_matrix_24_merged_pd(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_peak_data1/",
                                                                       "/media/youndukn/lastra/3d_peak_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp1 = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()
    numb_type = 16

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24_Merged_1((34, 34, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



print_ixs_node_matrix_24_merged_pd(pd,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/type_data1/"],
                         False)

filepath="./check_p/type_nmerged_nloss_pd_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
print(s_batch_init_temp1.shape)
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""

"""
print("Fr")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="fr_3d.inp"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 15), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/3d_fr_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""

"""
print("pd")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="pd_3d.inp"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_pd(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 15), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/3d_pd_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""


"""
print("Fxy 3d")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="fxy_3d"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 15), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/fxy_3d_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)

"""
"""
print("pd 2d")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="pd_2d.inp"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/2d_pd_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)

"""

"""
print("Fxy 2d")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="fxy_2d"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_fxy(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/fxy_2d_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""
print("Fr 2d")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="fr_2d"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_fr(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)
model = ResNetI7_MD_BN_BU_AT_MATRIX_24((34, 34, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/fr_2d_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)

"""
"""
print("Fxy conv 3d")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d((34, 34, 3, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)

#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)


"""
"""
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))


print("PD conv 3d dense")
#Input deck
file_name ="pd_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d((34, 34, 3, 5), classes=29)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name))

print_ixs_node_matrix_24_3d_conv_pd(pd,
                        karma_node,
                         model,
                         ["data_0"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)
"""
"""
print("Fxy conv 3d dense")
#Input deck
file_name_load ="fxy_3d_conv_dense"
file_name ="fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense((34, 34, 3, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])


model.load_weights("{}.hdf5".format(file_name_load))

print_ixs_node_matrix_24_3d_conv(pd,
                        karma_node,
                         model,
                         ["data_0"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)


pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

filepath="./check_p/fxy_3d_conv_dense_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=5,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""


"""
print("Fr conv 3d")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="fr_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d((34, 34, 3, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

#model = resnet1.ResnetBuilder.build_resnet_152((17, 17, 5), 2)

#model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node(pd,
#               karma_node,
#               model,
#               [],
#               ["data_1", "data_2", "data_3", "data_4"],
#               ["./idata/node/"],
#               False)



#print_ixs_node_matrix_24(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/jin_data1/"],
#                         False)


filepath="./check_p/fr_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""

"""
print("PD conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    y_pred = y_pred[:,:225]
    y_true = y_true[:,:225]
    return K.mean(K.square((y_pred - y_true)))

#Input deck
file_name ="pd_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d((34, 34, 3, 5), classes=29)
model.summary()


model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])


model.load_weights("{}.hdf5".format(file_name))

#print_ixs_node_matrix_24_3d_conv(pd,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_pd(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp =  pickle.load(file_load)
    file_load.close()

filepath="./check_p/pd_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)

"""
"""
print("Fxy Pin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name ="pin_fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_1_3d_dense_pin((34, 34, 3, 5), (8, 8, 1, 64), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name))

print_ixs_node_matrix_1_3d_conv_pin(fxy,
                         karma_node,
                         model,
                        ["data_1"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_pin(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp1 = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

s_batch_init_temp1 = numpy.array(s_batch_init_temp1, dtype=np.float16)
filepath="./check_p/pin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=[s_batch_init_temp, s_batch_init_temp1],
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""

"""
print("Custom Loss Fxy woPin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true)*(y_true+1)/2)))

#Input deck
file_name_load ="custom_loss_wopin_fxy_3d_conv"
file_name ="pin_fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_1_3d_dense((34, 34, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

print_ixs_node_matrix_1_3d_conv_wopin(fxy,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)


pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_pin(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp1 = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

s_batch_init_temp1 = numpy.array(s_batch_init_temp1, dtype=np.float16)
filepath="./check_p/custom_loss_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}-{val_acc:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=7,
          epochs=500,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""
"""
print("Fr Pin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name ="pin_fr_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_1_3d_dense_pin((34, 34, 3, 5), (8, 8, 1, 64), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name))

print_ixs_node_matrix_1_3d_conv_pin(pd,
                         karma_node,
                         model,
                        ["data_1"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_pin(pd,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp1 = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

s_batch_init_temp1 = numpy.array(s_batch_init_temp1, dtype=np.float16)
filepath="./check_p/pin_fr_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}-{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=[s_batch_init_temp, s_batch_init_temp1],
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""
"""
print("Custom Loss Fxy woPin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="wopin_fr_3d_conv"
file_name ="pin_fr_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_1_3d_dense((34, 34, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_1_3d_conv_wopin(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)



pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_pin(fr,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"s_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp1 = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

s_batch_init_temp1 = numpy.array(s_batch_init_temp1, dtype=np.float16)
filepath="./check_p/wopin_fr_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}-{val_acc:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""

"""
print("Custom Loss Fxy woPin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="quad_wopin_fxy_3d_conv"
file_name ="wopin_fxy_3d_conv_quad"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_quad((34, 34, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

print_ixs_node_matrix_24_3d_conv_quad(fxy,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)



pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_quad(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/quad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""

"""
print("Custom Loss Fxy woPin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="max_wopin_fxy_3d_conv"
file_name ="max_wopin_fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_max((34, 34, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.000001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

print_ixs_node_matrix_24_3d_conv_max(fxy,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)


pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_max(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/max_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)

"""
"""
print("Max Loss Fr woPin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="max_wopin_fr_3d_conv"
file_name ="max_wopin_fr_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_max((34, 34, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_quad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_max(fr,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/max_wopin_fr_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=5,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""
"""

print("Custom Loss Fxy woPin conv 3d dense")
def custom_loss_test(y_true,y_pred):all_1_large_iquad_wopin_fxy_3d_conv
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="local_max_wopin_fxy_3d_conv"
file_name ="local_max_wopin_fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_max_local((34, 34, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)


pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_max_local(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/local_max_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=3,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)

"""

"""
print("Max Loss Fr woPin conv 3d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="iquad_max_wopin_fxy_3d_conv"
file_name ="iquad_max_wopin_fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_max_iquad((17, 17, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_max_iquad(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/iquad_max_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=500,
          validation_split=.05,
          verbose=0,
          callbacks=callbacks_list)
"""

"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="iquad_max_wopin_fxy_2d_conv"
file_name ="iquad_max_wopin_fxy_2d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_2d_dense_max_iquad((17, 17, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_2d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)
pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_2d_conv_max_iquad(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/iquad_max_wopin_fxy_2d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=20,
          epochs=1000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""
"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="max_wopin_fxy_2d_conv"
file_name ="max_wopin_fxy_2d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_2d_dense_max((34, 34, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_2d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_2d_conv_max(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/max_wopin_fxy_2d_conv_weights-improvement-{epoch:02d}-{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=20,
          epochs=1000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""

"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="iquad_max_wopin_cbc_2d_conv"
file_name ="iquad_max_wopin_cbc_2d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_2d_dense_max_iquad_cbc((17, 17, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_2d_conv_max_iquad(cbc,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/fil_iquad_max_wopin_cbc_2d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=20,
          epochs=500,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""
"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="iquad_wopin_fxy_3d_conv"
file_name ="iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad((17, 17, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.000001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_iquad(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=10,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""
"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="averaged_iquad_wopin_fxy_3d_conv"
file_name ="averaged_iquad_wopin_fxy_3d_conv"

print(file_name)
#shuffle_data([], ["_s"], ["/media/youndukn/lastra/type_data1/"], )

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad((17, 17, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.000001), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_iquad_averaged(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/averaged_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=20,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""

"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="large_iquad_wopin_fxy_3d_conv"
file_name ="iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large((17, 17, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_24_3d_conv_iquad(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/large_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=20,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""

"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="9_large_iquad_wopin_fxy_3d_conv"
file_name ="9_iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large((17, 17, 3, 5), classes=9)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp, y_batch_init_temp1 = prepare_ixs_bu_node_matrix_24_3d_conv_iquad(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp1"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp1, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp1"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp1 = pickle.load(file_load)
    file_load.close()

filepath="./check_p/9_large_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp1,
          batch_size=20,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""

"""
print("Max Loss Fr woPin conv 2d dense")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="1_large_iquad_wopin_fxy_3d_conv"
file_name ="1_large_iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large((17, 17, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/1_large_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=20,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""
"""
print("3D Fr ")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="divide_1_large_iquad_wopin_fxy_3d_conv"
file_name ="divide_1_large_iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large((17, 17, 3, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_divide(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/divide_1_large_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=20,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""
"""
print("3D Fr ")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="all_1_large_iquad_wopin_fxy_3d_conv"
file_name ="all_1_large_iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all((17, 17, 14, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad_all(fxy,
#                         karma_node,
#                        model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data4/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_all(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/all_1_large_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=5,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""

"""
print("3D Fr ")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="boc_1_large_iquad_wopin_fxy_3d_conv"
file_name ="boc_1_large_iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all((17, 17, 14, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.00001), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_boc(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/boc_1_large_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=3,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""


"""
print("3D Fr ")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="global_all_1_large_iquad_wopin_fxy_3d_conv"
file_name ="global_all_1_large_iquad_wopin_fxy_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global((17, 17, 14, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_all_global(fxy,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/global_all_1_large_iquad_wopin_fxy_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=5,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""

print("3D Fr ")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="global_all_1_large_iquad_wopin_fr_3d_conv"
file_name ="global_all_1_large_iquad_wopin_fr_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global((17, 17, 14, 5), classes=1)
model.summary()

model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_all_global(fr,
                                                                      karma_node,
                                                                      None,
                                                                      [],
                                                                      [],
                                                                      ["/media/youndukn/lastra/3d_xs_data1/",
                                                                       "/media/youndukn/lastra/3d_xs_data2/",
                                                                       "/media/youndukn/lastra/3d_xs_data3/",
                                                                       "/media/youndukn/lastra/3d_xs_data4/"])
    file_name1 = file_name+"s_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(s_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_dump = open(file_name1, 'wb')
    pickle.dump(y_batch_init_temp, file_dump, protocol=pickle.HIGHEST_PROTOCOL)
    file_dump.close()
else:
    file_name1 = file_name+"s_batch_init_temp"
    file_load = open(file_name1, 'rb')
    s_batch_init_temp = pickle.load(file_load)
    file_load.close()
    file_name1 = file_name+"y_batch_init_temp"
    file_load = open(file_name1, 'rb')
    y_batch_init_temp = pickle.load(file_load)
    file_load.close()

filepath="./check_p/global_all_1_large_iquad_wopin_fr_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=s_batch_init_temp,
          y=y_batch_init_temp,
          batch_size=3,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)


"""
"""
import numpy as np
from keras.layers import Input, Add, Dense, Activation, Concatenate, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,GlobalAveragePooling2D, \
    Conv3D, AveragePooling2D, MaxPooling2D, Reshape, Multiply, LeakyReLU, LSTM, GRU, Lambda,GlobalAveragePooling3D, MaxPooling1D, MaxPooling3D,ZeroPadding3D
from keras.layers import Conv3DTranspose
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

import train_util as tu

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

cl = 0
fxy = 9
fr = 10
both = 3
pd = 2
cbc = -1

astra = 1
karma = 4
karma_node = 6

cl_ra, cl_rb = 10000, 10000
fxy_ra, fxy_rb = 1, 1


def convolution_block_se_bn_jin_3d_iquad_C(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv3D(filters=F1, kernel_size=(1, 1, 1), strides=(s, s, s), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = ZeroPadding3D(padding=( (1, 1), (0, 1), (0, 1)))(X)
    X_Col = tu.nodeColPermute3D_C()(X)
    X_Row = tu.nodeRowPermute3D_C()(X)
    X_Cen = tu.nodeCen3D_C()(X)
    X_Row = Concatenate(axis=3)([X_Cen, X_Row])
    X = Concatenate(axis=3)([X_Col, X])
    X = Concatenate(axis=2)([X_Row, X])

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


def convolution_block_resize_3d_wopin_iquad_all_global(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X = ZeroPadding3D(padding=( (0, 0), (0, 1), (0, 1),))(X)
    X_Col = tu.nodeColPermute3D_C()(X)
    X_Row = tu.nodeRowPermute3D_C()(X)
    X_Cen = tu.nodeCen3D_C()(X)
    X_Row = Concatenate(axis=3)([X_Cen, X_Row])
    X = Concatenate(axis=3)([X_Col, X])
    X = Concatenate(axis=2)([X_Row, X])
    X = Conv3D(filters=F1, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = tu.gSlicer3D_C()(X)

    X = ZeroPadding3D(padding=((1, 1), (0, 1), (0, 1)))(X)
    X_Col = tu.assColPermute3D_C()(X)
    X_Row = tu.assRowPermute3D_C()(X)
    X_Cen = tu.assCen3D_C()(X)
    X_Row = Concatenate(axis=3)([X_Cen, X_Row])
    X = Concatenate(axis=3)([X_Col, X])
    X = Concatenate(axis=2)([X_Row, X])

    X = Conv3D(filters=F2, kernel_size=(16, 1, 1), strides=(1, 1, 1), padding='valid',
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



def convolution_block_resize_3d_wopin_iquad_all_global_pin_max(X, X_pin, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X = ZeroPadding3D(padding=( (0, 0), (0, 1), (0, 1),))(X)
    X_Col = tu.nodeColPermute3D_C()(X)
    X_Row = tu.nodeRowPermute3D_C()(X)
    X_Cen = tu.nodeCen3D_C()(X)
    X_Row = Concatenate(axis=3)([X_Cen, X_Row])
    X = Concatenate(axis=3)([X_Col, X])
    X = Concatenate(axis=2)([X_Row, X])
    X = Conv3D(filters=F1, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = tu.gSlicer3D_C()(X)

    X = ZeroPadding3D(padding=((1, 1), (0, 1), (0, 1)))(X)
    X_Col = tu.assColPermute3D_C()(X)
    X_Row = tu.assRowPermute3D_C()(X)
    X_Cen = tu.assCen3D_C()(X)
    X_Row = Concatenate(axis=3)([X_Cen, X_Row])
    X = Concatenate(axis=3)([X_Col, X])
    X = Concatenate(axis=2)([X_Row, X])

    X = Conv3D(filters=F2, kernel_size=(16, 1, 1), strides=(1, 1, 1), padding='valid',
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

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Multiply()([X, X_pin])
    X = MaxPooling3D(pool_size=(1, 16, 16))(X)
    X = Activation('relu')(X)

    return X




def convolution_block_resize_3d_wopin_iquad_all_global_pin(X, X_pin, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X = ZeroPadding3D(padding=( (0, 0), (0, 1), (0, 1),))(X)
    X_Col = tu.nodeColPermute3D_C()(X)
    X_Row = tu.nodeRowPermute3D_C()(X)
    X_Cen = tu.nodeCen3D_C()(X)
    X_Row = Concatenate(axis=3)([X_Cen, X_Row])
    X = Concatenate(axis=3)([X_Col, X])
    X = Concatenate(axis=2)([X_Row, X])
    X = Conv3D(filters=F1, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = tu.gSlicer3D_C()(X)

    X = ZeroPadding3D(padding=((1, 1), (0, 1), (0, 1)))(X)
    X_Col = tu.assColPermute3D_C()(X)
    X_Row = tu.assRowPermute3D_C()(X)
    X_Cen = tu.assCen3D_C()(X)
    X_Row = Concatenate(axis=3)([X_Cen, X_Row])
    X = Concatenate(axis=3)([X_Col, X])
    X = Concatenate(axis=2)([X_Row, X])

    X = Conv3D(filters=F2, kernel_size=(16, 1, 1), strides=(1, 1, 1), padding='valid',
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

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3DTranspose(filters=F2, kernel_size=(1, 3, 3), strides=(1, s, s), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=F3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Multiply()([X, X_pin])
    X = Conv3D(filters=F3, kernel_size=(1, 16, 16), strides=(1, 16, 16), padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)

    return X

def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global(input_shape=(3, 17, 17,  3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad_C(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad_C(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad_all_global(pd_X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=2)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input], outputs=outputs, name="ResNetCen")

    return model

def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global_pin(input_shape=(3, 17, 17,  3), input_shape2=(1, 128, 128, 1), classes=6):
    X_input = Input(input_shape)
    X_pin = Input(input_shape2)

    stage = 0;
    X = X_input

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad_C(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad_C(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad_all_global_pin(pd_X, X_pin, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=2)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input, X_pin], outputs=outputs, name="ResNetCen")

    return model


def ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global_pin_max(input_shape=(3, 17, 17,  3), input_shape2=(1, 128, 128, 1), classes=6):
    X_input = Input(input_shape)
    X_pin = Input(input_shape2)

    stage = 0;
    X = X_input

    for i in range(20):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_bn_jin_3d_iquad_C(X, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=1)
        X = Concatenate()([X, X_input])
        stage = stage + 1

    outputs = []
    for i in range(classes):
        pd_X = X

        for j in range(2):
            stage_filters = [64, 64, 256]
            pd_X = convolution_block_se_bn_jin_3d_iquad_C(pd_X, f=3, filters=stage_filters, stage=stage,
                                                     block='ca_{}'.format(i), s=1)
            pd_X = Concatenate()([pd_X, X_input])
            stage = stage + 1

        stage_filters = [int(256/2), 64, 1]
        pd_X = convolution_block_resize_3d_wopin_iquad_all_global_pin_max(pd_X, X_pin, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i), s=2)
        pd_X = Flatten()(pd_X)
        outputs.append(pd_X)

    model = Model(inputs=[X_input, X_pin], outputs=outputs, name="ResNetCen")

    return model

def prepare_ixs_bu_node_matrix_1_3d_conv_iquad_boc_global(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = tu.get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    ffl = FFL("HIPER_U56_HANA6.FF", "/home/youndukn/Plants/db/tset/")

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
                s_batch_init = numpy.swapaxes(s_batch_init[-19:-2, -19:-2, 6:20, :], 1, 2)
                s_batch_init = numpy.swapaxes(s_batch_init, 0, 1)
                s_batch_init_temp.append(s_batch_init)

                y_batch_init = numpy.array(burnup_boc[2].reshape(10,10,1))[:8, :8, :].flatten()
                y_batch_init_temp[0].append(y_batch_init)

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
    return s_batch_init_temp, s_batch_init_temp1,  y_batch_init_temp


def prepare_ixs_bu_node_matrix_1_3d_conv_iquad_boc_global_pin(optimize_id,
                input_id,
                model,
                include=["y3"],
                exclude=["y314"],
                folders=["/media/youndukn/lastra/plants_data/"]):
    files = tu.get_files_with(include, exclude, "", folders)
    for file in files:
        print(file.name)
    print("Files {}".format(len(files)))

    counter = 0

    y_batch_init_temp = []
    s_batch_init_temp = []
    s_batch_init_temp1 = []

    ffl = FFL("HIPER_U56_HANA6.FF", "/home/youndukn/Plants/db/tset/")

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
                s_batch_init = numpy.swapaxes(s_batch_init[-19:-2, -19:-2, 6:20, :], 1, 2)
                s_batch_init = numpy.swapaxes(s_batch_init, 0, 1)

                s_batch_init_temp.append(s_batch_init)

                s_batch_init2 = numpy.zeros((1, 8*16, 8*16, 1))
                burnup_ass =  burnup_boc[7][-10:-2, -10:-2, :]
                burnup_id = burnup_boc[8][-10:-2, -10:-2, :]
                for row_i, row in enumerate(burnup_id):
                    for col_i, col in enumerate(row):
                        ass_burnup= burnup_ass[row_i, col_i, 0]
                        ass_burnup_index = 0
                        for a_index, depl_value in enumerate(ffl.depltion):
                            if ass_burnup <= int(depl_value):
                                ass_burnup_index = a_index
                                break
                        if tu.isInt(col[1]):
                            ahole = ffl.ff["HIPER_X"+str(col[1])+"C"][ass_burnup_index,0,:,:]
                        elif "c" == str(col[1]).lower():
                            ahole = ffl.ff["HIPER_220C"][ass_burnup_index, 0, :, :]
                        else:
                            ahole = numpy.zeros((16, 16))
                        s_batch_init2[0,row_i*16:(row_i+1)*16,col_i*16:(col_i+1)*16,0] = ahole[:,:]

                s_batch_init_temp1.append(s_batch_init2)
                """
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
                """
                y_batch_init = numpy.array(burnup_boc[optimize_id][-10:-2, -10:-2, :]).flatten()
                y_batch_init_temp[0].append(y_batch_init)

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
    return s_batch_init_temp, s_batch_init_temp1,  y_batch_init_temp


def print_ixs_node_matrix_24_3d_conv_max_iquad_all(optimize_id,
                                     input_id,
                                     model,
                                     include=["y314"],
                                     exclude=[],
                                     folders=["/media/youndukn/lastra/plants_data/"],
                                     den=False):
    print()
    ffl = FFL("HIPER_U56_HANA6.FF", "/home/youndukn/Plants/db/tset/")

    axial_numb = 26
    for x_man in range(1, 2):

        files = tu.get_files_with(include, exclude, "", folders)
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

                    s_batch_init2 = numpy.zeros((1, 8 * 16, 8 * 16, 1))
                    burnup_ass = burnup_boc[7][-10:-2, -10:-2, :]
                    burnup_id = burnup_boc[8][-10:-2, -10:-2, :]
                    for row_i, row in enumerate(burnup_id):
                        for col_i, col in enumerate(row):
                            ass_burnup = burnup_ass[row_i, col_i, 0]
                            ass_burnup_index = 0
                            for a_index, depl_value in enumerate(ffl.depltion):
                                if ass_burnup <= int(depl_value):
                                    ass_burnup_index = a_index
                                    break
                            if tu.isInt(col[1]):
                                ahole = ffl.ff["HIPER_X" + str(col[1]) + "C"][ass_burnup_index, 0, :, :]
                            elif "c" == str(col[1]).lower():
                                ahole = ffl.ff["HIPER_220C"][ass_burnup_index, 0, :, :]
                            else:
                                ahole = numpy.zeros((16, 16))
                            s_batch_init2[0, row_i * 16:(row_i + 1) * 16, col_i * 16:(col_i + 1) * 16, 0] = ahole[:, :]

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

                    y_batch_init = numpy.array(burnup_boc[optimize_id][-10:-2, -10:-2, :]).flatten()
                    maxi_index = numpy.argmax(y_batch_init)
                    diff = readout_t0[0,:] - y_batch_init
                    diff2 = abs(readout_t0[0, maxi_index] - y_batch_init[maxi_index])

                    print("dif {:0.2f} {:0.2f} {:0.2f} {:0.2f} ".format(diff2, y_batch_init[maxi_index], numpy.amin(diff), numpy.amax(diff)))

                    if maximum < diff2:
                        maximum =diff2
                        print(diff2, y_batch_init[maxi_index])
                print("Max average", max_sum)

            except (AttributeError, EOFError, ImportError, IndexError) as e:
                print(e)
                pass

def custom_loss(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="global_all_1_large_iquad_pin_fr_3d_conv"
file_name ="global_all_1_large_iquad_pin_fr_3d_conv"

pre_prepared = True
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_boc_global_pin(fr,
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


model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global_pin((14, 17, 17, 5), (1, 128, 128, 1), classes=1)
model.summary()

model.compile(loss=custom_loss, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

model.load_weights("{}.hdf5".format(file_name_load))

print_ixs_node_matrix_24_3d_conv_max_iquad_all(fxy,
                         karma_node,
                         model,
                         ["data_1"],
                         [],
                         ["/media/youndukn/lastra/3d_xs_data5/"],
                         False)

"""
filepath="./check_p/global_all_1_large_iquad_pin_fr_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=[s_batch_init_temp, s_batch_init_temp1],
          y=y_batch_init_temp,
          batch_size=5,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)

"""
"""
print("PD BOC ")
def custom_loss_test(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="global_boc_1_large_iquad_wopin_pd_3d_conv"
file_name ="global_boc_1_large_iquad_wopin_pd_3d_conv"

print(file_name)

model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global((14, 17, 17, 5), classes=1)
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

    s_batch_init_temp,s_batch_init_temp1, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_boc_global(fr,
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

filepath="./check_p/global_boc_1_large_iquad_wopin_pd_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
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
def custom_loss(y_true,y_pred):
    return K.mean(K.square(((y_pred - y_true))))

#Input deck
file_name_load ="global_all_1_large_iquad_pin_fr_3d_conv"
file_name ="max_global_all_1_large_iquad_pin_fr_3d_conv"

pre_prepared = False
if pre_prepared == False:

    s_batch_init_temp, s_batch_init_temp1, y_batch_init_temp = prepare_ixs_bu_node_matrix_1_3d_conv_iquad_boc_global_pin(fr,
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


model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all_global_pin_max((14, 17, 17, 5), (1, 128, 128, 1), classes=1)
model.summary()

model.compile(loss=custom_loss, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])

#model.load_weights("{}.hdf5".format(file_name_load))

#print_ixs_node_matrix_24_3d_conv_max_iquad(fxy,
#                         karma_node,
#                         model,
#                         ["data_1"],
#                         [],
#                         ["/media/youndukn/lastra/3d_xs_data5/"],
#                         False)

filepath="./check_p/max_global_all_1_large_iquad_pin_fr_3d_conv_weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
callbacks_list = [checkpoint]
model.fit(x=[s_batch_init_temp, s_batch_init_temp1],
          y=y_batch_init_temp,
          batch_size=5,
          epochs=2000,
          validation_split=.05,
          verbose=2,
          callbacks=callbacks_list)
"""
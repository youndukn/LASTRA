import tkinter as tk
import random
from tkinter import messagebox

import time
from astra_rd import AstraRD

import keras.backend as K
import numpy as np
from keras.layers import Input, Add, Dense, Activation, Concatenate, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,GlobalAveragePooling2D, \
    Conv3D, AveragePooling2D, MaxPooling2D, Reshape, Multiply, LeakyReLU, LSTM, GRU, Lambda,GlobalAveragePooling3D, MaxPooling1D, MaxPooling3D,ZeroPadding3D
from keras.initializers import glorot_uniform
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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



def custom_loss_test(y_true,y_pred):
    #y_mult = y_true[:,225:]
    return K.mean(K.square(((y_pred - y_true))))


class MemoryTile:
    def __init__(self, parent, calculated, predicted, cross_section, model):
        self.max_row = 8
        self.max_col = 8
        self.parent = parent
        self.calculated = calculated
        self.predicted = predicted
        self.cross_section = cross_section
        self.model = model
        self.index = 0
        self.buttons = [[tk.Button(root,
                                   width=6,
                                   height=4,
                                   command=lambda row=row, col=col: self.choose_tile(row, col)
                                   ) for col in range(self.max_col)] for row in range(self.max_row)]

        for row in range(self.max_row):
            for column in range(self.max_col):
                self.buttons[row][column].grid(row=row, column=column)

        self.first = None
        self.draw_board()

    def draw_board(self):
        for row, row_buttons in enumerate(self.buttons):
            for col, button in enumerate(row_buttons):
                if self.calculated[row][col][self.index] == 0 and self.predicted[row][col][self.index] == 0:
                    cal_pre = ""
                    mycolor = "white"
                else:
                    mycolor = '#%02x%02x%02x' % (0, 0, 256-int(self.calculated[row][col][self.index]/3*256))
                    cal_pre = "{:0.2f}\n{:0.2f}\n{:0.2f}".format(self.calculated[row][col][self.index],
                                                                 self.predicted[row][col][self.index],
                                                                 abs(self.calculated[row][col][self.index]-self.predicted[row][col][self.index]))
                button.config(text=cal_pre, fg='white', font=('Helvetica', '14'),  bg=mycolor, state=tk.NORMAL)
        self.buttons[0][0].config(state=tk.DISABLED)
        self.start_time = time.monotonic()

    def choose_tile(self, row, col):
        self.index += 1
        self.parent.after(30, self.draw_board)

        """
        if not self.first:

            self.buttons[row][col].config(state=tk.DISABLED)
            self.first = (row, col)
        else:
            first_row, first_col = self.first

            self.buttons[first_row][first_col].config(state=tk.NORMAL)
            first_row = first_row*2-1
            first_col = first_col*2-1
            first_row1 = first_row*2+1
            first_col1 = first_col*2+1

            if first_row == -1:
                first_row = 0
                first_row1 = 1
            if first_col == -1:
                first_col = 0
                first_col1 = 1

            row = row*2-1
            col = col*2-1
            row1 = row*2+1
            col1 = col*2+1

            if row == -1:
                row = 0
                row1 = 1
            if col == -1:
                col = 0
                col1 = 1

            first_lp = self.cross_section[0:1, first_row:first_row1, first_col:first_col1, :, :]
            self.cross_section[0:1, first_row:first_row1,first_col:first_col1, :, :] = \
                self.cross_section[0:1, row:row1, col:col1, :, :]
            self.cross_section[0:1, row:row1, col:col1, :, :] = first_lp
            readout_t0 = self.model.predict(self.cross_section)
            self.predicted = readout_t0[0][0].reshape(8, 8)
            self.first = None
            self.parent.after(300, self.draw_board)
        """


#Input deck
file_name_load ="averaged_iquad_wopin_fxy_3d_conv"
file_name_load ="iquad_wopin_fxy_3d_conv"
file_name_load ="all_1_large_iquad_wopin_fxy_3d_conv"

astra = AstraRD("/home/youndukn/Plants/1.4.0/ucn5/c16/depl/022_astra_u516_nep_depl.inp",
                main_directory="/home/youndukn/Plants/1.4.0/ucn5/c16/depl/", thread_id=0)

cross_set = astra.get_cross_set()


s_batch = cross_set[0].iinput3n_tensor_full

if file_name_load == "averaged_iquad_wopin_fxy_3d_conv":

    calculated = cross_set[0].peak_tensor_full[-10:-2, -10:-2, :]
    calculated = np.amax(calculated, axis=2)

    model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad((17, 17, 3, 5), classes=1)
    model.summary()

    model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
    model.load_weights("{}.hdf5".format(file_name_load))

    first = [9.05, 6.19, 2.86, 12.38, 7.62, 20, 18.1, 18.1, 20]
    second = [20, 18.1, 18.1, 20, 20, 18.1, 18.1, 20]
    third = [18.1, 20, 18.1, 20, 7.62, 12.38, 2.86, 6.19, 9.05]

    first_sum = 114.3
    second_sum = 152.4
    third_sum = 114.3

    first_batch = s_batch[-19:-2, -19:-2, 0:1, :] * first[0]
    for xxx in range(1, 9):
        xxx1 = xxx - 0
        first_batch[:, :, 0, :] += s_batch[-19:-2, -19:-2, xxx, :] * first[xxx1]
    first_batch /= first_sum

    second_batch = s_batch[-19:-2, -19:-2, 9:10, :] * second[0]
    for xxx in range(10, 17):
        xxx1 = xxx - 9
        second_batch[:, :, 0, :] += s_batch[-19:-2, -19:-2, xxx, :] * second[xxx1]
    second_batch /= second_sum

    third_batch = s_batch[-19:-2, -19:-2, 17:18, :] * third[0]
    for xxx in range(18, 26):
        xxx1 = xxx - 17
        third_batch[:, :, 0, :] += s_batch[-19:-2, -19:-2, xxx, :] * third[xxx1]
    third_batch /= third_sum

    s_batch = np.concatenate(
        (first_batch,
         second_batch,
         third_batch), axis=-2)


elif file_name_load == "all_1_large_iquad_wopin_fxy_3d_conv":

    calculated = cross_set[0].pp3d_tensor_full[-10:-2, -10:-2, 6:20, :]
    for burnup in cross_set:
        calculated = np.concatenate((calculated,burnup.pp3d_tensor_full[-10:-2, -10:-2, 6:20, :]), axis=3)
    calculated = np.amax(calculated, axis=3)

    model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad_large_all((17, 17, 14, 5), classes=1)
    model.summary()

    model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
    model.load_weights("{}.hdf5".format(file_name_load))
    s_batch = s_batch[-19:-2, -19:-2, 6:20, :]
else:

    calculated = cross_set[0].peak_tensor_full[-10:-2, -10:-2, :]
    calculated = np.amax(calculated, axis=2)

    model = ResNetI7_MD_BN_BU_AT_MATRIX_24_3d_dense_iquad((17, 17, 3, 5), classes=1)
    model.summary()

    model.compile(loss=custom_loss_test, optimizer=Adam(lr=0.0000025), metrics=['accuracy'])
    model.load_weights("{}.hdf5".format(file_name_load))

    s_batch = np.concatenate(
        (s_batch[-19:-2, -19:-2, 5:5 + 1, :],
         (s_batch[-19:-2, -19:-2, 12:12 + 1, :] + s_batch[-19:-2, -19:-2, 13:13 + 1, :]) / 2,
         s_batch[-19:-2, -19:-2, 20:21, :]), axis=-2)

s_batch = np.expand_dims(s_batch, axis=0)
readout_t0 = model.predict(s_batch)
predicted = readout_t0[0].reshape(8,8,14)

root = tk.Tk()
memory_tile = MemoryTile(root, calculated, predicted, s_batch, model)
memory_tile.calculated = calculated
memory_tile.predicted = predicted
memory_tile.draw_board()
root.mainloop()
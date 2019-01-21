from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D,GlobalAveragePooling2D, \
    Reshape, Multiply

from keras.models import Model

from keras.initializers import glorot_uniform
from keras.optimizers import Adam


def identity_block_se_bn(X, f, filters, stage, block):
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


def ResNet(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)

    stage = 0;
    X = X_input
    for i in range(20):
        stage_filters = [256, 256, 256]
        X = convolution_block_se_bn(X, f = 3, filters = stage_filters, stage = stage, block='ca_{}'.format(i), s = 1)
        stage = stage + 1

    X = Flatten()(X)
    X = Dense(classes, name='cfc' + str(classes) + str(0), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = [X_input], outputs = X, name = "ResNetCen")

    model.compile(loss='mse', optimizer=Adam(lr=0.00001))
    model._make_predict_function()
    model.summary()
    return model
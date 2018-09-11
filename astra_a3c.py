from keras.layers import Input, Add, Dense, BatchNormalization, Flatten, Conv2D,GlobalAveragePooling2D, \
                         Reshape, Multiply, LeakyReLU, Convolution2D, Concatenate, Lambda, Dropout
from keras.models import Sequential, Model
from keras.layers.advanced_activations import ELU

from keras.initializers import glorot_uniform

from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np
import threading, queue
import time

from astra import Astra
import glob
import os
import enviroments
import pickle

from data.assembly import FreshAssembly

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import multiprocessing as mp

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000
# 환경 생성

ab55 = [
    [0, 9, 9],
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
    [8, 9, 17],
    [9, 9, 18],
    [11, 10, 10],
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [18, 10, 17],
    [19, 10, 18],
    [22, 11, 11],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [28, 11, 17],
    [29, 11, 18],
    [33, 12, 12],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [37, 12, 16],
    [38, 12, 17],
    [39, 12, 18],
    [44, 13, 13],
    [45, 13, 14],
    [46, 13, 15],
    [47, 13, 16],
    [48, 13, 17],
    [49, 13, 18],
    [55, 14, 14],
    [56, 14, 15],
    [57, 14, 16],
    [58, 14, 17],
    [59, 14, 18],
    [66, 15, 15],
    [67, 15, 16],
    [68, 15, 17],
    [69, 15, 18],
    [77, 16, 16],
    [78, 16, 17],
    [79, 16, 18],
    [88, 17, 17],
    [89, 17, 18],
    [99, 18, 18],
]

ab_si = [
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


cl_base = 17000
fxy_base = 1.55

def convolution_block_se_lk(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c', trainable=False)(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1', trainable=False)(X_shortcut)

    X = Add()([X_shortcut, X])
    X = LeakyReLU()(X)

    return X

def identity_block_se_lk(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c', trainable=False)(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X = Add()([X, X_shortcut])
    X = LeakyReLU()(X)

    return X

def build_feature_map(input_shape, output_dim=1280):
    input = Input(shape=input_shape)
    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                    s=1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
        stage = stage + 1
    for i in range(5, 7):
        stage_filters = [int(64//2), int(64//2), int(256//2)]
        X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                    s=1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
        stage = stage + 1

    for i in range(8, 9):
        stage_filters = [64 // 4, 64 // 4, 256 // 4]
        X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                    s=1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
        stage = stage + 1

    conv = Flatten()(X)
    fc = Dense(output_dim, name="feature", activation='relu')(conv)
    model = Model(inputs=input, outputs=fc)
    return model

def build_feature_map_full(input_shape, output_dim=248, name = "ori"):
    input = Input(shape=input_shape)
    classes = output_dim
    X = Dense(248, name='fc1' + str(classes) + str(name), kernel_initializer=glorot_uniform(seed=0))(input)
    X = ELU()(X)
    X = Dense(248*2, name='fc2' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)
    X = Dense(248*3, name='fc3' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)
    X = Dense(248*2, name='fc4' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)
    X = Dense(248, name='fc5' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)

    fc = Dense(output_dim, name="feature"+ str(name), activation='relu')(X)
    model = Model(inputs=input, outputs=fc)
    return model

def build_feature_map_full_connected(input_shape, output_dim=1280, name = "global"):
    input = Input(shape=input_shape, name="power_input")
    classes = output_dim
    X = Dense(248, name='fc1' + str(classes) + str(name), kernel_initializer=glorot_uniform(seed=0))(input)
    X = ELU()(X)
    X = Dense(248*2, name='fc2' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)
    X = Dense(248*3, name='fc3' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)
    X = Dense(248*2, name='fc4' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)
    X = Dense(248, name='fc5' + str(classes)+ str(name), kernel_initializer=glorot_uniform(seed=0))(X)
    X = ELU()(X)

    fc = Dense(output_dim, name="feature"+ str(name), activation='relu')(X)

    return input, fc

def inverse_model(output_dim=3300):
    """
    s_t, s_t+1 -> a_t
    """
    def func(ft0, ft1):
        h = Concatenate()([ft0, ft1])
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='sigmoid')(h)
        return h
    return func

def forward_model(output_dim=1280):
    """
    s_t, a_t -> s_t+1
    """
    def func(ft, at):
        h = Concatenate()([ft, at])
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='linear')(h)
        return h
    return func

def get_reward_intrinsic(model, intrinsic, x):
    return K.function([model.get_layer("state0").input,
                       model.get_layer("state1").input,
                       model.get_layer("action").input],
                      [intrinsic])(x)[0]

# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self):
        # 상태크기와 행동크기를 갖고옴
        self.spaces = enviroments.Environment(action_main_shape=(400,), action_sub_shapes=(55,),observation_shape=(19,19,25))

        self.state_size = self.spaces.observation_space.shape
        self.action_size = self.spaces.action_space.shape[0]
        # A3C 하이퍼파라미터
        self.discount_factor = 0.70
        self.no_op_steps = 30
        self.actor_lr = 2.5e-5
        self.critic_lr = 2.5e-5
        # 쓰레드의 갯수
        self.threads = 6
        self.policies = []

        for index in range(9):
            policy_array = []
            for jnjex in range(self.spaces.action_space.shape[0]):
                policy_array.append(0)
            self.policies.append(policy_array)

        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()

        self.icm, self.icm2, self.r_in, self.icm_optimizer= self.build_icm_model((500,), self.spaces.action_space.shape)

        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer(), self.icm_optimizer]

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/astra_a3c', self.sess.graph)


    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        my_queue = queue.Queue()

        #self.load_model("./save_model/astra_a3c")
        # 쓰레드 수만큼 Agent 클래스 생성
        the_directory = "{}{}{}{}{}depl{}".format("/home/youndukn/Plants/1.4.0/",
                                              "ygn3",
                                              os.path.sep,
                                              'c{0:02}'.format(10),
                                              os.path.sep,
                                              os.path.sep)



        agents = []
        for thread_id in range(self.threads):
            agents.append(Agent(self.action_size, self.state_size,
                  [self.actor, self.critic], self.sess,
                  self.optimizer, self.discount_factor,
                  [self.summary_op, self.summary_placeholders,
                   self.update_ops, self.summary_writer], the_directory, thread_id, self.icm,self.icm2, self.r_in, my_queue))

        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()

        fig, axs = plt.subplots(3, 3, figsize=(20, 15))

        lns = []
        for row in range(3):
            columns = []
            for column in range(3):
                columns.append(axs[row, column].plot([], [], 'bo'))
            lns.append(columns)
        plt.show(block=False)

        last_summary_time = time.time()
        summary_interval = 1000

        while True:
            item = my_queue.get()
            row = int(item[0] / 2)%3
            axs[row, 0].set_title("thread{:d}".format(item[0]))
            axs[row, 0].imshow(item[1][:8][:8])
            #axs[row, 1].imshow(item[10])
            #lns[row][2][0].set_xdata(range(0, self.spaces.action_space.shape[0]))
            #lns[row][2][0].set_ydata(item[2])
            #axs[row, 2].relim()
            #axs[row, 2].autoscale_view(True, True, True)

            #axs[row, 2].imshow(item[2])
            axs[row, 1].imshow(np.array(item[5][:8][:8]))
            #axs[row, 4].imshow(np.array(item[3][0]))
            #axs[row, 4].set_title("pre cl fxy {:3.2f} {:3.2f}".format(item[6], item[7]))
            axs[row, 2].imshow(np.array(item[4][0][:8][:8]))
            axs[row, 2].set_title("{:3.2f} {:3.2f}->{:3.2f} {:3.2f}".format(item[6], item[7], item[8], item[9]))
            fig.canvas.draw()
            now = time.time()
            if now - last_summary_time > summary_interval:
                self.save_model("./save_model/astra_a3c")
                last_summary_time=now

        """
        fig, axs = plt.subplots(3, 3, figsize=(13, 10))

        lns = []
        for thread_id in range(9):
            x = int(thread_id % 3)
            y = int(thread_id / 3)

            lns.append(axs[x, y].plot([], [], 'bo'))
        plt.show(block=False)

        last_summary_time = time.time()
        summary_interval = 1000

        while True:
            # if show_training:
            #    for env in envs:
            #        env.render()

            item = my_queue.get()
            if item:
                x = int(item[0] % 3)
                y = int(item[0] / 3)
                # ln.set_xdata(range(0, get_num_actions()))
                # ln.set_ydata(item[1])

                lns[item[0]][0].set_xdata(range(0, self.spaces.action_space.shape[0]))
                # indexes = np.where(np.logical_and(item[1] >= np.percentile(item[1], 5), item[1] <= np.percentile(item[1], 95)))
                # item[1][indexes] = 0
                array = item[1]
                lns[item[0]][0].set_ydata(array)
                # axs[x,y].plot(range(0, get_num_actions()), item[1], block=False)
                axs[x, y].relim()
                axs[x, y].autoscale_view(True, True, True)
                axs[x, y].set_title("thread{:d}".format(item[0]))
                fig.canvas.draw()

            now = time.time()
            if now - last_summary_time > summary_interval:
                self.save_model("./save_model/astra_a3c")
                last_summary_time=now
        
        
        """

    # 정책신경망과 가치신경망을 생성
    def build_model(self):
        input = Input(shape=self.state_size, name="xs_input")
        stage = 0;

        for i in range(5):
            stage_filters = [64, 64, 256]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cb_{}'.format(i))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
            stage = stage + 1

        for i in range(5, 7):
            stage_filters = [64 // 2, 64 // 2, 256 // 2]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
            stage = stage + 1

        for i in range(8, 9):
            stage_filters = [64 // 4, 64 // 4, 256 // 4]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
            stage = stage + 1

        conv = Flatten()(X)
        fc = Dense(3800, activation='relu')(conv)
        f_input, f_output =build_feature_map_full_connected((500, ))

        fc = Concatenate()([f_output, fc])

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=[input, f_input], outputs=policy)
        critic = Model(inputs=[input, f_input], outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic


    def build_icm_model(self, state_shape, action_shape, lmd=0.2, beta=0.01):
        s_t0 = Input(shape=state_shape, name="state0")
        s_t1 = Input(shape=state_shape, name="state1")
        a_t = Input(shape=action_shape, name="action")
        fmap = build_feature_map_full(state_shape)
        f_t0 = fmap(s_t0)
        f_t1 = fmap(s_t1)
        act_hat = inverse_model(action_shape[0])(f_t0, f_t1)
        f_t1_hat = forward_model(output_dim=248)(f_t0, a_t)

        r_in = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1), name="intrinsic_reward")([f_t1, f_t1_hat])
        l_i = Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1))([a_t, act_hat])
        loss0 =Lambda(lambda x: beta * x[0] + (1.0 - beta) * x[1])([r_in, l_i])
        rwd = Input(shape=(1,))
        loss = Lambda(lambda x: (-lmd * x[0] + x[1]))([rwd, loss0])
        """
        r_in = merge([f_t1, f_t1_hat], mode=lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),
                     output_shape=(1,), name="reward_intrinsic")
        l_i = merge([a_t, act_hat], mode=lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1),
                    output_shape=(1,))
        loss0 = merge([r_in, l_i],
                      mode=lambda x: beta * x[0] + (1.0 - beta) * x[1],
                      output_shape=(1,))
        loss = merge([rwd, loss0],
                     mode=lambda x: (-lmd * x[0].T + x[1]).T,
                     output_shape=(1,))
        """
        model2 = Model([s_t0, s_t1, a_t], [r_in])
        model2._make_predict_function()
        model2.summary()

        model = Model([s_t0, s_t1, a_t, rwd], loss)
        model._make_predict_function()
        model.summary()

        optimizer = RMSprop(lr=self.actor_lr, rho=0.999)
        updates = optimizer.get_updates(model.trainable_weights, [],loss)
        train = K.function([s_t0, s_t1, a_t, rwd],
                           [loss], updates=updates)

        return model, model2, r_in, train

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.999)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.get_layer("xs_input").input,
                            self.actor.get_layer("power_input").input,
                            action,
                            advantages],
                           [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.999)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.actor.get_layer("xs_input").input,
                            self.actor.get_layer("power_input").input,
                            discounted_prediction],
                           [loss], updates=updates)
        return train


    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")
        self.icm.load_weights(name+"_icm.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")
        self.icm.save_weights(name + "_icm.h5")

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 액터러너 클래스(쓰레드)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops, input_directory, thread_id, icm,icm2, r_in, my_queue):
        threading.Thread.__init__(self)

        #Action Space
        self.spaces = enviroments.Environment(action_main_shape=(400,), action_sub_shapes=(55,),observation_shape=(19,19,25))

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops
        self.icm = icm
        self.icm2 = icm2
        self.r_in = r_in
        self.thread_id = thread_id
        self.queue = my_queue

        self.target = (16300, 1.55)

        self.epsilon = 0.99
        self.epsilon_decay =0.999

        self.pre_actions = np.zeros(self.action_size)

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards, self.next_states, self.outputs, self.next_outputs= [], [], [], [], [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # 모델 업데이트 주기
        self.t_max = 30
        self.t = 0

        self.T = 0


        #create enviroment with directory
        directory = "{}{}{}".format(input_directory, os.path.sep, thread_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        input_name = glob.glob("{}01_*.inp".format(input_directory))
        self.env = Astra(input_name[0], reward_list_target=
        (16300, 0, 0, 0, 1.55, 0), main_directory = input_directory, working_directory=directory)

        self.file = open('/media/youndukn/lastra/plants_data6/{}_data_{}'.
                         format(input_name[0].replace(input_directory, ""), self.thread_id), 'wb')

        self.file_trainable = open('/media/youndukn/lastra/trainable_data6/{}_data_{}'.
                         format(input_name[0].replace(input_directory, ""), self.thread_id), 'wb')


    def my_input_output(self, values):
        burnup_boc = values[0]
        burnup_eoc = values[-1]
        """
        s_batch_init = burnup_boc.input_tensor_full
        s_batch_init_den = burnup_boc.density_tensor_full
        s_batch_init_den = np.array(s_batch_init_den)
        my_state = np.concatenate((s_batch_init, s_batch_init_den), axis=2)
        """
        selected_range = [4, 5, 6, 7]
        s_batch_init = np.array(burnup_boc.input_tensor_full)
        for index in selected_range:
            s_batch_init_selected = values[index].input_tensor_full
            s_batch_init_selected = np.array(s_batch_init_selected)
            s_batch_init = np.concatenate((s_batch_init, s_batch_init_selected), axis=2)

        my_state = s_batch_init

        my_output = np.array(burnup_boc.output_tensor)
        for index in selected_range:
            my_output = np.concatenate([my_output, values[index].output_tensor])

        o_batch_init = np.zeros((5, 10, 10))
        for indexes in ab55:
            o_batch_init[0][indexes[1]-9][indexes[2]-9] = burnup_boc.output_tensor[indexes[0]]
            o_batch_init[0][indexes[2] - 9][indexes[1] - 9] = burnup_boc.output_tensor[indexes[0]]

        for e_index, index in enumerate(selected_range):
            for indexes in ab55:
                o_batch_init[e_index+1][indexes[1] - 9][indexes[2] - 9] = \
                    values[index].output_tensor[indexes[0]]
                o_batch_init[e_index + 1][indexes[2] - 9][indexes[1] - 9] = \
                    values[index].output_tensor[indexes[0]]

        my_cl = burnup_eoc.summary_tensor[0]

        my_fxy = 0

        for burnup_point in values:
            if my_fxy < burnup_point.summary_tensor[5]:
                my_fxy = burnup_point.summary_tensor[5]
        return my_state, my_output, my_cl, my_fxy, o_batch_init

    def convert_f_index(self, action):

        if action < len(ab_in) * len(ab_in):

            pre_pos = action // len(ab_in)
            next_pos = action % len(ab_in)
            position = ab_in[pre_pos][0]
            posibilities = ab_in[next_pos][0]

        else:
            changed_action = action - len(ab_in) * len(ab_in)

            pre_pos = changed_action // len(ab_si)
            next_pos = changed_action % len(ab_si)

            position = ab_si[pre_pos][0]
            posibilities = ab_si[next_pos][0]

        for index, value in enumerate(ab55):
            if value[0] == position:
                a_position = index
            if value[0] == posibilities:
                a_posibilities = index

        return a_position, a_posibilities

    def run(self):
        global episode

        step = 0


        while episode < EPISODES:
            done = False

            score = 0
            self.env.reset()
            self.pre_actions = np.zeros(self.action_size)

            start_state, start_output, start_cl, start_fxy, start_output_matrix = self.my_input_output(self.env.get_cross_set())

            history = np.reshape([start_state], (1,
                                             self.spaces.observation_space.shape[0],
                                             self.spaces.observation_space.shape[1],
                                             self.spaces.observation_space.shape[2]))

            history_output = np.reshape([start_output], (1, 500))

            best_cl = start_cl
            best_fxy = start_fxy
            best_score = 0

            current_cl = start_cl
            current_fxy = start_fxy
            current_output = start_output
            current_output_matrix = start_output_matrix

            nonChanged = False

            while not done:

                step += 1

                action, policy, m_policy = self.get_action(history, history_output)

                policy_matrix = np.zeros((10, 10))
                m_policy_matrix = np.zeros((10, 10))
                max_policy_matrix = np.zeros((10, 10))
                next_max_policy_matrix = np.zeros((10, 10))

                max_value = 0

                for p_index, police in enumerate(policy):

                    position, posibilities = self.convert_f_index(p_index)

                    if posibilities == 1 or posibilities == 11:
                        max_value = 0

                    policy_matrix[ab55[position][1] - 9][ab55[position][2] - 9] += police
                    if  ab55[position][2] != ab55[position][1]:
                        policy_matrix[ab55[position][2] - 9][ab55[position][1] - 9] += police

                    if police > max_value:
                        max_policy_matrix[ab55[position][1] - 9][ab55[position][2] - 9] = police
                        max_policy_matrix[ab55[position][2] - 9][ab55[position][1] - 9] = police
                        max_value = police

                max_value = 0

                for p_index, police in enumerate(m_policy):

                    position, posibilities = self.convert_f_index(p_index)

                    if posibilities == 0:
                        max_value = 0

                    if police > max_value:
                        m_policy_matrix[ab55[position][1] - 9][ab55[position][2] - 9] = police
                        m_policy_matrix[ab55[position][2] - 9][ab55[position][1] - 9] = police
                        max_value = police

                """
                max_policy_index = np.argmax(policy)

                max_policy_position = max_policy_index // (self.spaces.action_space.shapes[0])

                p_index = max_policy_position * (self.spaces.action_space.shapes[0])
                p_index_1 = p_index+(self.spaces.action_space.shapes[0])

                for n_dex, police in enumerate(m_policy[p_index:p_index_1]):
                    next_max_policy_matrix[ab55[n_dex][1] - 9][ab55[n_dex][2] - 9] = police
                    next_max_policy_matrix[ab55[n_dex][2] - 9][ab55[n_dex][1] - 9] = police
                """
                """
                posibilities = action % (self.spaces.action_space.shapes[0] + 3 + 2)
                position = action // (self.spaces.action_space.shapes[0] + 3 + 2)

                
                if posibilities < self.spaces.action_space.shapes[0]:
                    action_index = (position * self.spaces.action_space.shapes[0]) + posibilities
                    s_t1, r_t, changed, info, satisfied = self.env.step_shuffle(action_index, [0, 4])
                elif posibilities >= self.spaces.action_space.shapes[0] and posibilities < self.spaces.action_space.shapes[0] + 3:
                    action_index = (position * 3) + (posibilities - self.spaces.action_space.shapes[0])
                    s_t1, r_t, changed, info, satisfied = self.env.step_rotate(action_index, [0, 4])
                elif posibilities >= self.spaces.action_space.shapes[0] + 3 and posibilities < self.spaces.action_space.shapes[0] + 5:
                    action_index = (position * 2) + (posibilities - self.spaces.action_space.shapes[0] - 3)
                    s_t1, r_t, changed, info, satisfied = self.env.step_bp(action_index, [0, 4])
                """

                position, posibilities = self.convert_f_index(action)

                position_matrix = np.zeros((10, 10))

                core = self.env.get_last_core()
                for index1 in range(10):
                    for index2 in range(10):
                        if type(core.assemblies[index1][index2]) is FreshAssembly:
                            position_matrix[index1][index2] = 0.5

                if position_matrix[ab55[position][1]-9][ab55[position][2]-9] == 0.5:
                    position_matrix[ab55[position][1]-9][ab55[position][2]-9] = 1.5
                else:
                    position_matrix[ab55[position][1] - 9][ab55[position][2] - 9] = 1

                if position_matrix[ab55[posibilities][1] - 9][ab55[posibilities][2] - 9] == 0.5:
                    position_matrix[ab55[posibilities][1] - 9][ab55[posibilities][2] - 9] = 1.5
                else:
                    position_matrix[ab55[posibilities][1] - 9][ab55[posibilities][2] - 9] = 1

                action_index = (position * self.spaces.action_space.shapes[0]) + posibilities
                s_t1, r_t, changed, info, satisfied, best, cross_set = self.env.step_shuffle(action_index, [0, 4])

                done = not info

                if best:
                    pre_cl = current_cl
                    pre_fxy = current_fxy

                pre_output_matrix = current_output_matrix

                current_state, \
                current_output, \
                current_cl, \
                current_fxy, \
                current_output_matrix = \
                    self.my_input_output(cross_set)

                next_history = np.reshape([current_state], (1,
                                                  self.spaces.observation_space.shape[0],
                                                  self.spaces.observation_space.shape[1],
                                                  self.spaces.observation_space.shape[2]))

                next_history_output = np.reshape([current_output], (1, 500))
                """
                reward = (10 * (min(self.target[0], current_cl) - min(self.target[0], pre_cl)) / cl_base + \
                          10 * (max(self.target[1], pre_fxy)    - max(self.target[1], current_fxy)) / fxy_base)/4
                """
                reward = (10 * (min(current_cl - self.target[0], 0)) / cl_base + \
                          10 * (min(self.target[1] - current_fxy, 0)) / fxy_base)/4

                #r_in = get_reward_intrinsic(self.icm, self.r_in,  )
                one_hot_action = np.zeros(self.spaces.action_space.shape)
                one_hot_action[action] = 1
                #r_in = self.icm2.predict([history_output, next_history_output, np.array([one_hot_action])])[0]
                r_in = 0
                # 정책의 최대값
                self.avg_p_max += np.amax(self.actor.predict([history, history_output]))

                non_clipped = reward

                reward = reward + r_in

                # score_addup
                score += reward

                if best_cl < current_cl and best_fxy > current_fxy:
                    best_cl = min(self.target[0], current_cl)
                    best_fxy = max(self.target[1], current_fxy)
                    best_score = score

                reward = np.clip(reward, -1., 1.)

                self.append_sample(history,
                                   action,
                                   reward,
                                   next_history,
                                   history_output,
                                   next_history_output)
                self.t += 1
                if changed:

                    # 샘플을 저장
                    """if nonChanged:
                        self.append_sample(u_history,
                                           u_action,
                                           u_reward,
                                           u_next_history,
                                           u_history_output,
                                           u_next_history_output)
                    """

                    self.queue.put([self.thread_id,
                                    max_policy_matrix,
                                    policy,
                                    pre_output_matrix,
                                    current_output_matrix,
                                    position_matrix,
                                    pre_cl,
                                    pre_fxy,
                                    current_cl,
                                    current_fxy,
                                    policy_matrix])
                    self.T += 1

                    pickle.dump([history, action, reward, done], self.file_trainable, protocol=pickle.HIGHEST_PROTOCOL)

                    dump_list = []
                    for value in cross_set:
                        a_list = [value.summary_tensor,
                                  value.input_tensor_full,
                                  value.output_tensor,
                                  value.flux_tensor,
                                  value.density_tensor_full]
                        dump_list.append(a_list)

                    values = self.critic.predict([next_history, next_history_output])[0]
                    values2 = self.local_critic.predict([next_history, next_history_output])[0]

                    print("|{:4d} |".format(self.thread_id),
                          "{:4d} |".format(ab55[position][1]-9),
                          "{:4d} |".format(ab55[position][2]-9),
                          "{:4d} |".format(ab55[posibilities][1]-9),
                          "{:4d} |".format(ab55[posibilities][2]-9),
                          "{:3.2f} |".format(current_cl),
                          "{:3.2f} |".format(current_fxy),
                          "{:3.2f} |".format(non_clipped),
                          "{:3.2f} |".format(r_in),
                          "{:1.4f} |".format(values[0]),
                          "{:1.4f} |".format(values2[0]),
                          )


                    pickle.dump(dump_list, self.file, protocol=pickle.HIGHEST_PROTOCOL)
                    nonChanged = True

                else:
                    nonChanged = False
                    u_history = history
                    u_action = action
                    u_reward = reward
                    u_next_history = next_history
                    u_history_output = history_output
                    u_next_history_output = next_history_output

                if best:

                    history = next_history
                    history_output = next_history_output
                else:
                    self.env.step_back()

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t >= self.t_max or done:
                    print("{}".format(self.thread_id), ' '.join('{:3d}'.format(np.argmax(k)) for k in self.actions))
                    print("{}".format(self.thread_id), ' '.join('{:3d}'.format(int(k*100)) for k in self.rewards))
                    self.train_model(False)
                    self.update_local_model()
                    self.t = 0

                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1

                    print(
                          "{:4d} |".format(self.thread_id),
                          "{:4d} |".format(self.T),
                          "{:4d} |".format(step),
                          "{:3.2f} |".format(start_cl),
                          "{:3.2f} |".format(start_fxy),
                          "{:3.2f} |".format(best_cl),
                          "{:3.2f} |".format(best_fxy),
                          "{:3.2f} |".format(best_score),
                          "{:3.2f} |".format(score),
                          "{:3.2f} |".format(self.epsilon),
                          )

                    stats = [score, self.avg_p_max / float(step), step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0
                    self.T = 0

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done, next_states, next_outputs, reverse = False):
        discounted_prediction = np.zeros_like(rewards)
        """
        running_add = 0
        
        if not done:
            running_add = self.critic.predict(np.float32(
                states[-1]))[0]
        """
        for t in reversed(range(0, len(rewards))):
            """running_add = self.critic.predict([np.float32(
                next_states[t]),
                np.float32(
                    next_outputs[t]) ])[0]
            """
            running_add = 0
            reward = rewards[t]
            if reverse:
                reward = reward*-1

            running_add = (running_add * self.discount_factor + reward)
            discounted_prediction[t] = running_add

        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):

        states = np.zeros((len(self.states),
                           self.spaces.observation_space.shape[0],
                           self.spaces.observation_space.shape[1],
                           self.spaces.observation_space.shape[2]))


        for i in range(len(self.states)):
            states[i] = self.states[i]


        states = np.float32(states)

        next_states = np.zeros((len(self.next_states),
                           self.spaces.observation_space.shape[0],
                           self.spaces.observation_space.shape[1],
                           self.spaces.observation_space.shape[2]))

        for i in range(len(self.next_states)):
            next_states[i] = self.next_states[i]

        next_states = np.float32(next_states)

        outputs = np.zeros((len(self.outputs), 500))

        for i in range(len(self.outputs)):
            outputs[i] = self.outputs[i]

        outputs = np.float32(outputs)

        next_outputs = np.zeros((len(self.next_outputs), 500))

        for i in range(len(self.next_outputs)):
            next_outputs[i] = self.next_outputs[i]

        next_outputs = np.float32(next_outputs)

        discounted_prediction = self.discounted_prediction(self.rewards, done, self.next_states, self.next_outputs)

        values = self.critic.predict([states, outputs])
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in discounted_prediction))
        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in values))

        """
        if len(self.rewards) >2:
            reversed_discounted_prediction = self.discounted_prediction(self.rewards[:-1], done, self.states, self.outputs)

            reversed_values = self.critic.predict([next_states[1:], next_outputs[1:]])
            reversed_values = np.reshape(reversed_values, len(reversed_values))

            reversed_advantages = reversed_discounted_prediction - reversed_values

            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_discounted_prediction))
            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_values))

            self.optimizer[0]([next_states[1:], next_outputs[1:], np.array(self.actions[1:]), reversed_advantages])
            self.optimizer[1]([next_states[1:], next_outputs[1:], reversed_discounted_prediction])
        """
        """
        reversed_discounted_prediction = self.discounted_prediction(self.rewards, done, self.states, self.outputs, True)

        reversed_values = self.critic.predict([next_states, next_outputs])
        reversed_values = np.reshape(reversed_values, len(reversed_values))

        reversed_advantages = reversed_discounted_prediction - reversed_values

        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_discounted_prediction))
        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_values))
        self.optimizer[0]([next_states, next_outputs, np.array(self.actions), reversed_advantages])
        self.optimizer[1]([next_states, next_outputs, reversed_discounted_prediction])
        """

        """
        if len(states)>2:

            reversed_reward = []

            for reward in reversed(self.rewards[:-1]):
                reversed_reward.append(reward)

            reversed_discounted_prediction = self.discounted_prediction(reversed_reward, False,
                                                                        list(reversed(self.states[1:])),
                                                                        list(reversed(self.outputs[1:])))

            reversed_states = np.zeros((len(self.states) - 1,
                                        self.spaces.observation_space.shape[0],
                                        self.spaces.observation_space.shape[1],
                                        self.spaces.observation_space.shape[2]))

            for i, state in enumerate(list(reversed(self.states[1:]))):
                reversed_states[i] = state

            reversed_outputs = np.zeros((len(self.outputs)-1, 500))

            for i, output in enumerate(list(reversed(self.outputs[1:]))):
                reversed_outputs[i] = output

            reversed_outputs = np.float32(reversed_outputs)

            reversed_states = np.float32(reversed_states)

            reversed_values = self.critic.predict([reversed_states, reversed_outputs])
            reversed_values = np.reshape(reversed_values, len(reversed_values))

            reversed_advantages = reversed_discounted_prediction - reversed_values

            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_discounted_prediction))
            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_values))

            self.optimizer[0]([reversed_states, reversed_outputs, np.array(list(reversed(self.actions[:-1]))), reversed_advantages])
            self.optimizer[1]([reversed_states, reversed_outputs, reversed_discounted_prediction])
        """


        self.optimizer[0]([states, outputs, np.array(self.actions), advantages])
        self.optimizer[1]([states, outputs, discounted_prediction])
        self.optimizer[2]([outputs, next_outputs, np.array(self.actions), np.array(discounted_prediction).reshape(-1, 1)])

        #self.icm.train_on_batch([states, next_states, np.array(self.actions),dddd], np.zeros((length_state,)))

        self.states, self.actions, self.rewards, self.next_states, self.outputs, self.next_outputs= [], [], [], [], [], []

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):
        input = Input(shape=self.state_size, name="xs_input")
        stage = 0;

        for i in range(5):
            stage_filters = [64, 64, 256]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='ca_{}_{}'.format(i, self.thread_id),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cb_{}_{}'.format(i, self.thread_id))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}_{}'.format(i, self.thread_id))
            stage = stage + 1

        for i in range(5, 7):
            stage_filters = [64 // 2, 64 // 2, 256 // 2]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
            stage = stage + 1

        for i in range(8, 9):
            stage_filters = [64 // 4, 64 // 4, 256 // 4]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
            stage = stage + 1

        conv = Flatten()(X)
        fc = Dense(3800, activation='relu')(conv)

        f_input, f_output =build_feature_map_full_connected((500, ))

        fc = Concatenate()([f_output, fc])

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        local_actor = Model(inputs=[input, f_input], outputs=policy)
        local_critic = Model(inputs=[input, f_input], outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic


    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, history, output):

        policy = self.local_actor.predict([history, output])[0]
        m_policy = policy

        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            action_index = np.random.choice(self.action_size, 1)[0]
        else:
            for action, action_number in enumerate(self.pre_actions):
                if 3 <= action_number:
                    value = m_policy[action]
                    distributed = value/self.action_size
                    m_policy[action] = 0
                    m_policy = np.add(m_policy, distributed)
                    print("V", self.thread_id, action)

            action_index = np.random.choice(self.action_size, 1, p=m_policy)[0]
            self.pre_actions[action_index] += 1
        return action_index, policy, m_policy


    # 샘플을 저장
    def append_sample(self, history, action, reward, next_history, output, next_output):
        self.states.append(history)
        self.next_states.append(next_history)

        self.outputs.append(output)
        self.next_outputs.append(next_output)

        act = np.zeros(self.action_size)
        act[action] = 1
        """
        posibilities = action % (self.spaces.action_space.shapes[0])
        position = action // (self.spaces.action_space.shapes[0])

        act[posibilities*self.spaces.action_space.shapes[0]+position] = 1
        """
        self.actions.append(act)
        self.rewards.append(reward*1.5+0.75)

if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()

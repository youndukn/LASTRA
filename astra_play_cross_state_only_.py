# -*- coding: utf-8 -*-
"""
Teaching a machine to play an Atari game (Pacman by default) by implementing
a 1-step Q-learning with TFLearn, TensorFlow and OpenAI gym environment. The
algorithm is described in "Asynchronous Methods for Deep Reinforcement Learning"
paper. OpenAI's gym environment is used here for providing the Atari game
environment for handling games logic and states. This example is originally
adapted from Corey Lynch's repo (url below).

Requirements:
    - gym environment (pip install gym)
    - gym Atari environment (pip install gym[atari])

References:
    - Asynchronous Methods for Deep Reinforcement Learning. Mnih et al, 2015.

Links:
    - Paper: http://arxiv.org/pdf/1602.01783v1.pdf
    - OpenAI's gym: https://gym.openai.com/
    - Original Repo: https://github.com/coreylynch/async-rl

"""
from __future__ import division, print_function, absolute_import

import threading, queue
import random
import numpy as np
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque


import gym
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d, conv_3d, max_pool_3d, avg_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.initializations import normal

from data.astra_train_set import AstraTrainSet, TrainSet
from astra import Astra
import pickle
import os
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy
import copy
# Fix for TF 0.12
try:
    writer_summary = tf.summary.FileWriter
    merge_all_summaries = tf.summary.merge_all
    histogram_summary = tf.summary.histogram
    scalar_summary = tf.summary.scalar
except Exception:
    writer_summary = tf.train.SummaryWriter
    merge_all_summaries = tf.merge_all_summaries
    histogram_summary = tf.histogram_summary
    scalar_summary = tf.scalar_summary

# Change that value to test instead of train
testing = False
# Model path (to load when testing)
test_model_path = './qlearning.tflearn.ckpt'
# Atari game to learn
# You can also try: 'Breakout-v0', 'Pong-v0', 'SpaceInvaders-v0', ...
game = 'Astra'
# Learning threads
n_threads = 10

# =============================
#   Training Parameters
# =============================
# Max training steps
TMAX = 1000000
# Current training step
T = 0
# Consecutive screen frames when performing training
action_repeat = 3
# Async gradient update frequency of each learning thread
I_AsyncUpdate = 20
# Timestep to reset the target network
I_target = 40000
# Learning rate
learning_rate = 0.001
# Reward discount rate
gamma = 0.70
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = 10000

# =============================
#   Utils Parameters
# =============================
# Display or not gym evironment screens
show_training = True
# Directory for storing tensorboard summaries
summary_dir = './tflearn_logs/'
summary_interval = 100
checkpoint_path = './qlearning.tflearn.ckpt'
checkpoint_interval = 2000
# Number of episodes to run gym evaluation
num_eval_episodes = 100


# =============================
#   TFLearn Deep Q Network
# =============================
def build_dqn(num_actions, action_repeat):
    """
    Building a DQN.
    """
    inputs = tf.placeholder(tf.float32, [None, 20, 20, action_repeat])
    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tflearn.conv_2d(inputs, 32, 4, strides=2, activation='relu')
    net = tflearn.conv_2d(net, 64, 2, strides=1, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='relu')
    q_values = tflearn.fully_connected(net, num_actions)
    return inputs, q_values


# =============================
#   TFLearn Deep Q Network
# =============================
def build_resnext():
    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5

    # Building Residual Network
    inputs = tflearn.input_data(shape=[None, 10, 10, 6])
    net = tflearn.conv_2d(inputs, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.resnext_block(net, n, 16, 32)
    net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
    net = tflearn.resnext_block(net, n - 1, 32, 32)
    net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
    net = tflearn.resnext_block(net, n - 1, 64, 32)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    net = tflearn.fully_connected(net, 100, activation='softmax')

    return inputs, net

# =============================
#   TFLearn Deep Q Network
# =============================
def build_dqn_fully(x, y, action_repeat, num_actions):
    """
    Building a DQN.
    """
    inputs = tf.placeholder(tf.float32, [None, x, y, action_repeat])
    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tflearn.fully_connected(inputs, 2000, activation='relu', weights_init = normal(stddev=0.01))
    net = tflearn.fully_connected(net, 2000*2, activation='relu', weights_init = normal(stddev=0.01))
    net = tflearn.fully_connected(net, 2000*3, activation='relu', weights_init = normal(stddev=0.01))
    net = tflearn.fully_connected(net, 2000*3, activation='relu', weights_init = normal(stddev=0.01))
    net = tflearn.fully_connected(net, 2000*2, activation='relu', weights_init = normal(stddev=0.01))
    net = tflearn.fully_connected(net, 2000*2, activation='relu', weights_init = normal(stddev=0.01))
    q_values = tflearn.fully_connected(net, num_actions)
    return inputs, q_values

# =============================
#   TFLearn Deep Q Network
# =============================
def build_dqn_exact_fully(x, y, action_repeat, num_actions):
    """
    Building a DQN.
    """
    inputs = tf.placeholder(tf.float32, [None, x, y, action_repeat])
    # Inputs shape: [batch, channel, height, width] need to be changed into
    # shape [batch, height, width, channel]
    net = tflearn.fully_connected(inputs, 2000, activation='relu', weights_init = normal(stddev=0.001))
    net = tflearn.fully_connected(net, 2000*2, activation='relu', weights_init = normal(stddev=0.001))
    net = tflearn.fully_connected(net, 2000*3, activation='relu', weights_init = normal(stddev=0.001))
    net = tflearn.fully_connected(net, 2000*3, activation='relu', weights_init = normal(stddev=0.001))
    net = tflearn.fully_connected(net, 2000*3, activation='relu', weights_init = normal(stddev=0.001))
    net = tflearn.fully_connected(net, 2000*3, activation='relu', weights_init = normal(stddev=0.001))
    net = tflearn.fully_connected(net, 2000*2, activation='relu', weights_init = normal(stddev=0.001))
    net = tflearn.fully_connected(net, 1000, activation='relu', weights_init = normal(stddev=0.001))
    q_values = tflearn.fully_connected(net, num_actions)
    return inputs, q_values

def inception_v3_3d(width, height, frame_count, output=9, model_name='sentnet_color.model'):
    inputs = input_data(shape=[None, width, height, frame_count, 1])

    inception_3a_1_1 = conv_3d(inputs, 64, 1, activation='relu')
    inception_3a_3_3_reduce = conv_3d(inputs, 96, 1, activation='relu')
    inception_3a_3_3 = conv_3d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu')
    inception_3a_5_5_reduce = conv_3d(inputs, 16, filter_size=1, activation='relu')
    inception_3a_5_5 = conv_3d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu')
    inception_3a_pool = max_pool_3d(inputs, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_3d(inception_3a_pool, 32, filter_size=1, activation='relu')
    tflearn.initializations.truncated_normal()
    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                mode='concat', axis=4)

    inception_3b_1_1 = conv_3d(inception_3a_output, 128, filter_size=1, activation='relu')
    inception_3b_3_3_reduce = conv_3d(inception_3a_output, 128, filter_size=1, activation='relu')
    inception_3b_3_3 = conv_3d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu')
    inception_3b_5_5_reduce = conv_3d(inception_3a_output, 32, filter_size=1, activation='relu')
    inception_3b_5_5 = conv_3d(inception_3b_5_5_reduce, 96, filter_size=5)
    inception_3b_pool = max_pool_3d(inception_3a_output, kernel_size=3, strides=1)
    inception_3b_pool_1_1 = conv_3d(inception_3b_pool, 64, filter_size=1, activation='relu')

    # merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                mode='concat', axis=4)

    pool3_3_3 = max_pool_3d(inception_3b_output, kernel_size=3, strides=2)
    inception_4a_1_1 = conv_3d(pool3_3_3, 192, filter_size=1, activation='relu')
    inception_4a_3_3_reduce = conv_3d(pool3_3_3, 96, filter_size=1, activation='relu')
    inception_4a_3_3 = conv_3d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu')
    inception_4a_5_5_reduce = conv_3d(pool3_3_3, 16, filter_size=1, activation='relu')
    inception_4a_5_5 = conv_3d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu')
    inception_4a_pool = max_pool_3d(pool3_3_3, kernel_size=3, strides=1)
    inception_4a_pool_1_1 = conv_3d(inception_4a_pool, 64, filter_size=1, activation='relu')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                mode='concat', axis=4)

    inception_4b_1_1 = conv_3d(inception_4a_output, 160, filter_size=1, activation='relu')
    inception_4b_3_3_reduce = conv_3d(inception_4a_output, 112, filter_size=1, activation='relu')
    inception_4b_3_3 = conv_3d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu')
    inception_4b_5_5_reduce = conv_3d(inception_4a_output, 24, filter_size=1, activation='relu')
    inception_4b_5_5 = conv_3d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu')

    inception_4b_pool = max_pool_3d(inception_4a_output, kernel_size=3, strides=1)
    inception_4b_pool_1_1 = conv_3d(inception_4b_pool, 64, filter_size=1, activation='relu')

    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                mode='concat', axis=4)

    inception_4c_1_1 = conv_3d(inception_4b_output, 128, filter_size=1, activation='relu')
    inception_4c_3_3_reduce = conv_3d(inception_4b_output, 128, filter_size=1, activation='relu')
    inception_4c_3_3 = conv_3d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu')
    inception_4c_5_5_reduce = conv_3d(inception_4b_output, 24, filter_size=1, activation='relu')
    inception_4c_5_5 = conv_3d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu')

    inception_4c_pool = max_pool_3d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_3d(inception_4c_pool, 64, filter_size=1, activation='relu')

    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                mode='concat', axis=4)

    inception_4d_1_1 = conv_3d(inception_4c_output, 112, filter_size=1, activation='relu')
    inception_4d_3_3_reduce = conv_3d(inception_4c_output, 144, filter_size=1, activation='relu')
    inception_4d_3_3 = conv_3d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu')
    inception_4d_5_5_reduce = conv_3d(inception_4c_output, 32, filter_size=1, activation='relu')
    inception_4d_5_5 = conv_3d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu')
    inception_4d_pool = max_pool_3d(inception_4c_output, kernel_size=3, strides=1)
    inception_4d_pool_1_1 = conv_3d(inception_4d_pool, 64, filter_size=1, activation='relu')

    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                mode='concat', axis=4)

    inception_4e_1_1 = conv_3d(inception_4d_output, 256, filter_size=1, activation='relu')
    inception_4e_3_3_reduce = conv_3d(inception_4d_output, 160, filter_size=1, activation='relu')
    inception_4e_3_3 = conv_3d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu')
    inception_4e_5_5_reduce = conv_3d(inception_4d_output, 32, filter_size=1, activation='relu')
    inception_4e_5_5 = conv_3d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu')
    inception_4e_pool = max_pool_3d(inception_4d_output, kernel_size=3, strides=1)
    inception_4e_pool_1_1 = conv_3d(inception_4e_pool, 128, filter_size=1, activation='relu')

    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=4,
                                mode='concat')

    pool4_3_3 = max_pool_3d(inception_4e_output, kernel_size=3, strides=2)

    inception_5a_1_1 = conv_3d(pool4_3_3, 256, filter_size=1, activation='relu')
    inception_5a_3_3_reduce = conv_3d(pool4_3_3, 160, filter_size=1, activation='relu')
    inception_5a_3_3 = conv_3d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu')
    inception_5a_5_5_reduce = conv_3d(pool4_3_3, 32, filter_size=1, activation='relu')
    inception_5a_5_5 = conv_3d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu')
    inception_5a_pool = max_pool_3d(pool4_3_3, kernel_size=3, strides=1)
    inception_5a_pool_1_1 = conv_3d(inception_5a_pool, 128, filter_size=1, activation='relu')

    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=4,
                                mode='concat')

    inception_5b_1_1 = conv_3d(inception_5a_output, 384, filter_size=1, activation='relu')
    inception_5b_3_3_reduce = conv_3d(inception_5a_output, 192, filter_size=1, activation='relu')
    inception_5b_3_3 = conv_3d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu')
    inception_5b_5_5_reduce = conv_3d(inception_5a_output, 48, filter_size=1, activation='relu')
    inception_5b_5_5 = conv_3d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu')
    inception_5b_pool = max_pool_3d(inception_5a_output, kernel_size=3, strides=1)
    inception_5b_pool_1_1 = conv_3d(inception_5b_pool, 128, filter_size=1, activation='relu')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=4,
                                mode='concat')

    pool5_7_7 = avg_pool_3d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)

    q_values = fully_connected(pool5_7_7, output)

    return inputs, q_values

def inception_v3_3d_init(width, height, frame_count, output=9, model_name='sentnet_color.model'):
    inputs = input_data(shape=[None, width, height, frame_count, 1])

    inception_3a_1_1 = conv_3d(inputs, 64, 1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_3_3_reduce = conv_3d(inputs, 96, 1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_3_3 = conv_3d(inception_3a_3_3_reduce, 128, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_5_5_reduce = conv_3d(inputs, 16, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_5_5 = conv_3d(inception_3a_5_5_reduce, 32, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_pool = max_pool_3d(inputs, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_3d(inception_3a_pool, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    # merge the inception   _3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                mode='concat', axis=4)

    inception_3b_1_1 = conv_3d(inception_3a_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_3_3_reduce = conv_3d(inception_3a_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_3_3 = conv_3d(inception_3b_3_3_reduce, 192, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_5_5_reduce = conv_3d(inception_3a_output, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_5_5 = conv_3d(inception_3b_5_5_reduce, 96, filter_size=5)
    inception_3b_pool = max_pool_3d(inception_3a_output, kernel_size=3, strides=1)
    inception_3b_pool_1_1 = conv_3d(inception_3b_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    # merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                mode='concat', axis=4)

    pool3_3_3 = max_pool_3d(inception_3b_output, kernel_size=3, strides=2)
    inception_4a_1_1 = conv_3d(pool3_3_3, 192, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_3_3_reduce = conv_3d(pool3_3_3, 96, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_3_3 = conv_3d(inception_4a_3_3_reduce, 208, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_5_5_reduce = conv_3d(pool3_3_3, 16, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_5_5 = conv_3d(inception_4a_5_5_reduce, 48, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_pool = max_pool_3d(pool3_3_3, kernel_size=3, strides=1)
    inception_4a_pool_1_1 = conv_3d(inception_4a_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                mode='concat', axis=4)

    inception_4b_1_1 = conv_3d(inception_4a_output, 160, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_3_3_reduce = conv_3d(inception_4a_output, 112, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_3_3 = conv_3d(inception_4b_3_3_reduce, 224, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_5_5_reduce = conv_3d(inception_4a_output, 24, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_5_5 = conv_3d(inception_4b_5_5_reduce, 64, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4b_pool = max_pool_3d(inception_4a_output, kernel_size=3, strides=1)
    inception_4b_pool_1_1 = conv_3d(inception_4b_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                mode='concat', axis=4)

    inception_4c_1_1 = conv_3d(inception_4b_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_3_3_reduce = conv_3d(inception_4b_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_3_3 = conv_3d(inception_4c_3_3_reduce, 256, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_5_5_reduce = conv_3d(inception_4b_output, 24, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_5_5 = conv_3d(inception_4c_5_5_reduce, 64, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4c_pool = max_pool_3d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_3d(inception_4c_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                mode='concat', axis=4)

    inception_4d_1_1 = conv_3d(inception_4c_output, 112, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_3_3_reduce = conv_3d(inception_4c_output, 144, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_3_3 = conv_3d(inception_4d_3_3_reduce, 288, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_5_5_reduce = conv_3d(inception_4c_output, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_5_5 = conv_3d(inception_4d_5_5_reduce, 64, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_pool = max_pool_3d(inception_4c_output, kernel_size=3, strides=1)
    inception_4d_pool_1_1 = conv_3d(inception_4d_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                mode='concat', axis=4)

    inception_4e_1_1 = conv_3d(inception_4d_output, 256, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_3_3_reduce = conv_3d(inception_4d_output, 160, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_3_3 = conv_3d(inception_4e_3_3_reduce, 320, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_5_5_reduce = conv_3d(inception_4d_output, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_5_5 = conv_3d(inception_4e_5_5_reduce, 128, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_pool = max_pool_3d(inception_4d_output, kernel_size=3, strides=1)
    inception_4e_pool_1_1 = conv_3d(inception_4e_pool, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=4,
                                mode='concat')

    pool4_3_3 = max_pool_3d(inception_4e_output, kernel_size=3, strides=2)

    inception_5a_1_1 = conv_3d(pool4_3_3, 256, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_3_3_reduce = conv_3d(pool4_3_3, 160, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_3_3 = conv_3d(inception_5a_3_3_reduce, 320, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_5_5_reduce = conv_3d(pool4_3_3, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_5_5 = conv_3d(inception_5a_5_5_reduce, 128, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_pool = max_pool_3d(pool4_3_3, kernel_size=3, strides=1)
    inception_5a_pool_1_1 = conv_3d(inception_5a_pool, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=4,
                                mode='concat')

    inception_5b_1_1 = conv_3d(inception_5a_output, 384, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_3_3_reduce = conv_3d(inception_5a_output, 192, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_3_3 = conv_3d(inception_5b_3_3_reduce, 384, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_5_5_reduce = conv_3d(inception_5a_output, 48, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_5_5 = conv_3d(inception_5b_5_5_reduce, 128, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_pool = max_pool_3d(inception_5a_output, kernel_size=3, strides=1)
    inception_5b_pool_1_1 = conv_3d(inception_5b_pool, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=4,
                                mode='concat')

    pool5_7_7 = avg_pool_3d(inception_5b_output, kernel_size=3, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)

    q_values = fully_connected(pool5_7_7, output)

    return inputs, q_values


def inception_v3_3d_init_kernel(width, height, frame_count, output=9, model_name='sentnet_color.model'):
    inputs = input_data(shape=[None, width, height, frame_count, 1])

    inception_3a_1_1 = conv_3d(inputs, 64, 1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_3_3_reduce = conv_3d(inputs, 96, 1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_3_3 = conv_3d(inception_3a_3_3_reduce, 128, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_5_5_reduce = conv_3d(inputs, 16, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_5_5 = conv_3d(inception_3a_5_5_reduce, 32, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3a_pool = max_pool_3d(inputs, kernel_size=[1,3,3,1,1], strides=1, )
    inception_3a_pool_1_1 = conv_3d(inception_3a_pool, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                mode='concat', axis=4)

    inception_3b_1_1 = conv_3d(inception_3a_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_3_3_reduce = conv_3d(inception_3a_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_3_3 = conv_3d(inception_3b_3_3_reduce, 192, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_5_5_reduce = conv_3d(inception_3a_output, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_3b_5_5 = conv_3d(inception_3b_5_5_reduce, 96, filter_size=5)
    inception_3b_pool = max_pool_3d(inception_3a_output, kernel_size=[1,3,3,1,1], strides=1)
    inception_3b_pool_1_1 = conv_3d(inception_3b_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    # merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                mode='concat', axis=4)

    pool3_3_3 = max_pool_3d(inception_3b_output, kernel_size=[1,3,3,1,1], strides=2)
    inception_4a_1_1 = conv_3d(pool3_3_3, 192, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_3_3_reduce = conv_3d(pool3_3_3, 96, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_3_3 = conv_3d(inception_4a_3_3_reduce, 208, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_5_5_reduce = conv_3d(pool3_3_3, 16, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_5_5 = conv_3d(inception_4a_5_5_reduce, 48, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4a_pool = max_pool_3d(pool3_3_3, kernel_size=[1,3,3,1,1], strides=1)
    inception_4a_pool_1_1 = conv_3d(inception_4a_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                mode='concat', axis=4)

    inception_4b_1_1 = conv_3d(inception_4a_output, 160, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_3_3_reduce = conv_3d(inception_4a_output, 112, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_3_3 = conv_3d(inception_4b_3_3_reduce, 224, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_5_5_reduce = conv_3d(inception_4a_output, 24, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4b_5_5 = conv_3d(inception_4b_5_5_reduce, 64, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4b_pool = max_pool_3d(inception_4a_output, kernel_size=[1,3,3,1,1], strides=1)
    inception_4b_pool_1_1 = conv_3d(inception_4b_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                mode='concat', axis=4)

    inception_4c_1_1 = conv_3d(inception_4b_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_3_3_reduce = conv_3d(inception_4b_output, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_3_3 = conv_3d(inception_4c_3_3_reduce, 256, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_5_5_reduce = conv_3d(inception_4b_output, 24, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4c_5_5 = conv_3d(inception_4c_5_5_reduce, 64, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4c_pool = max_pool_3d(inception_4b_output, kernel_size=[1,3,3,1,1], strides=1)
    inception_4c_pool_1_1 = conv_3d(inception_4c_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                mode='concat', axis=4)

    inception_4d_1_1 = conv_3d(inception_4c_output, 112, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_3_3_reduce = conv_3d(inception_4c_output, 144, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_3_3 = conv_3d(inception_4d_3_3_reduce, 288, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_5_5_reduce = conv_3d(inception_4c_output, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_5_5 = conv_3d(inception_4d_5_5_reduce, 64, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4d_pool = max_pool_3d(inception_4c_output, kernel_size=[1,3,3,1,1], strides=1)
    inception_4d_pool_1_1 = conv_3d(inception_4d_pool, 64, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                mode='concat', axis=4)

    inception_4e_1_1 = conv_3d(inception_4d_output, 256, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_3_3_reduce = conv_3d(inception_4d_output, 160, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_3_3 = conv_3d(inception_4e_3_3_reduce, 320, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_5_5_reduce = conv_3d(inception_4d_output, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_5_5 = conv_3d(inception_4e_5_5_reduce, 128, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_4e_pool = max_pool_3d(inception_4d_output, kernel_size=[1,3,3,1,1], strides=1)
    inception_4e_pool_1_1 = conv_3d(inception_4e_pool, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], axis=4,
                                mode='concat')

    pool4_3_3 = max_pool_3d(inception_4e_output, kernel_size=[1,3,3,1,1], strides=2)

    inception_5a_1_1 = conv_3d(pool4_3_3, 256, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_3_3_reduce = conv_3d(pool4_3_3, 160, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_3_3 = conv_3d(inception_5a_3_3_reduce, 320, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_5_5_reduce = conv_3d(pool4_3_3, 32, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_5_5 = conv_3d(inception_5a_5_5_reduce, 128, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5a_pool = max_pool_3d(pool4_3_3, kernel_size=[1,3,3,1,1], strides=1)
    inception_5a_pool_1_1 = conv_3d(inception_5a_pool, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))

    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=4,
                                mode='concat')

    inception_5b_1_1 = conv_3d(inception_5a_output, 384, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_3_3_reduce = conv_3d(inception_5a_output, 192, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_3_3 = conv_3d(inception_5b_3_3_reduce, 384, filter_size=3, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_5_5_reduce = conv_3d(inception_5a_output, 48, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_5_5 = conv_3d(inception_5b_5_5_reduce, 128, filter_size=5, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_pool = max_pool_3d(inception_5a_output, kernel_size=[1,3,3,1,1], strides=1)
    inception_5b_pool_1_1 = conv_3d(inception_5b_pool, 128, filter_size=1, activation = 'relu', weights_init = normal(stddev=0.02))
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=4,
                                mode='concat')

    pool5_7_7 = avg_pool_3d(inception_5b_output, kernel_size=[1,3,3,1,1], strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)

    q_values = fully_connected(pool5_7_7, output)

    return inputs, q_values
# =============================
#   ATARI Environment Wrapper
# =============================
class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size action_repeat from which environment state is constructed.
    """
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(gym_env.action_space.n)
        # Screen buffer of size action_repeat to be able to build
        # state arrays of size [1, action_repeat, 84, 84]
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack([x_t for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        0) Atari frames: 210 x 160
        1) Get image grayscale
        2) Rescale image 110 x 84
        3) Crop center 84 x 84 (you can crop top/bottom according to the game)
        """
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.action_repeat, 84, 84))
        s_t1[:self.action_repeat-1, :] = previous_frames
        s_t1[self.action_repeat-1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info


# =============================
#   1-step Q-Learning
# =============================
def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.05, .01, .1])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


def actor_learner_thread(thread_id, env, session, graph_ops, num_actions,
                         summary_ops, saver, q):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX, T

    s = graph_ops["s"]
    q_values = graph_ops["q_values"]
    a = graph_ops["a"]
    y = graph_ops["y"]
    grad_update = graph_ops["grad_update"]

    summary_placeholders, assign_ops, summary_op = summary_ops

    directory = ".{}{}".format(os.path.sep, thread_id)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists("{}{}astra".format(directory, os.path.sep)):
        copyfile(".{}astra".format(os.path.sep), "{}{}astra".format(directory, os.path.sep))

    env = Astra("test3.job", working_directory=directory)


    if os.path.isfile('data_{}.temp'.format(thread_id)):
        os.remove('data_{}.temp'.format(thread_id))

    file = open('data_{}'.format(thread_id), 'wb')


    # Initialize network gradients
    s_batch_l0 = []
    a_batch = []
    y_batch2 = np.ones((100,), dtype=np.int)
    r_batch = []

    if os.path.isfile('data_{}.temp'.format(thread_id)):
        os.remove('data_{}.temp'.format(thread_id))
    for i in range(0,10):

        if os.path.isfile('./data_cp_real_2/data_{}'.format(thread_id)):
            try:
                file_read = open('./data_cp_real_2/data_{}'.format(thread_id), 'rb')
                print('Processing File ./data_cp_real_2/data_{}'.format(thread_id))

                while True:
                    y_batch_init_temp = []
                    a_batch_init_temp = []
                    s_batch_init_temp = []
                    for num_batch in range(0, 100):
                        train_set = pickle.load(file_read)

                        for _ in range(0, 50):
                            # Clear gradients
                            s_batch_init = train_set[0]
                            action_index = random.randrange(100)
                            a_batch_init = np.zeros((100,), dtype=np.int)
                            a_batch_init[action_index] = 1
                            y_batch_init = train_set[1][action_index]

                            y_batch_init_temp.append(y_batch_init)
                            a_batch_init_temp.append(a_batch_init)
                            s_batch_init_temp.append(s_batch_init)

                    session.run(grad_update, feed_dict={y: y_batch_init_temp,
                                                        a: a_batch_init_temp,
                                                        s: s_batch_init_temp})

                    train_set = pickle.load(file_read)

                    # Clear gradients
                    s_batch_init = train_set[0]
                    y_batch_init = train_set[1]

                    readout_t0 = q_values.eval(session=session, feed_dict={s: [s_batch_init, ]})
                    #print(y_batch_init,numpy.average(abs(y_batch_init - readout_t0)))
                    print((readout_t0[0][3]-y_batch_init[3])/y_batch_init[3], readout_t0[0][3], y_batch_init[3])

                file_read.close()
            except:
                pass

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.3
    epsilon = 0.3

    r_batch = []
    s_batch_l0 = []
    s_batch_l2 = []
    y_batch0 = []
    y_batch2 = []
    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(thread_id)

    t = 0

    satisfied_data = open('satisfied_numb.txt', 'w')
    t_run = 0
    is_checkable = False
    while T < TMAX:
        s_t = env.get_initial_state()
        terminal = False

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        r_t_p = 0

        while True:

            # Forward the deep q network, get Q(s,a) values
            #readout_t = q_values.eval(session=session, feed_dict={s: [s_t]})

            #q.put([thread_id, copy.deepcopy(readout_t[0])])

            # Choose next action based on e-greedy policy
            """
            if random.random() > epsilon:
                indexes = np.where(readout_t > np.percentile(readout_t, 90))[0].
                if len(indexes) == 0:
                    action_index = np.argmax(readout_t)
                else:
                    action_i    ndex = random.choice(indexes)
            else:
                action_index = random.randrange(num_actions)
            """
            action_index = random.randrange(num_actions)

            s_t1, r_t, r_t_l, changed, info, satisfied = env.step_shuffle_full(action_index, [0, 1, 4])


            satisfied_data.write("{}".format(T))
            s_t1 = AstraTrainSet(s_t1, None, None, not info, r_t).state

            a_t = np.zeros([num_actions])
            a_t[action_index] = 1

            terminal = not info

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

            clipped_r_t = np.clip((r_t - r_t_p)*100, -1, 1)
            #y_batch.append(clipped_r_t)
            if changed and not terminal:
                is_checkable = True
                y_batch0.append(r_t_l[0])
                #y_batch1.append(r_t_l[1])
                y_batch2.append(r_t_l[4])

                s_batch_l0.append(copy.deepcopy(s_t1))
                s_batch_l2.append(copy.deepcopy(s_t1))
                # Forward the deep q network, get Q(s,a) values
                #readout_t0 = q_values0.eval(session=session, feed_dict={s0: [s_t1]})

                # Forward the deep q network, get Q(s,a) values
                #readout_t1 = q_values1.eval(session=session, feed_dict={s1: [s_t1]})

                # Forward the deep q network, get Q(s,a) values
                #readout_t2 = q_values2.eval(session=session, feed_dict={s2: [s_t1]})
                """
                print("| Thread {:2d}".format(int(thread_id)), "| Step", t,
                      "| Reward0: {:5.4f}, {:5.4f}".format(r_t_l[0], readout_t0[0][0]),
                      "| Reward1: {:5.4f}, {:5.4f}".format(r_t_l[1], readout_t0[0][0]),
                      "| Reward2: {:5.4f}, {:5.4f}".format(r_t_l[4], readout_t2[0][0]), )
                """
                #print("| Thread {:2d}".format(int(thread_id)), "| Step", t,
                #      "| Reward0: {:5.4f}, {:5.4f}".format(r_t_l[0], readout_t0[0][0]),
                #      "| Reward2: {:5.4f}, {:5.4f}".format(r_t_l[4], readout_t2[0][0]), )
                if env.cross_set:
                    r_batch.append(env.cross_set)

            #print("| Thread {:2d}".format(int(thread_id)), "| Step", t,
            #      "| Reward: {:4d}".format(int(clipped_r_t)), "| Batch: {:4d}".format(int(y_batch[-1])))

            #a_batch.append(a_t)
            #s_batch.append(s_t)

            # Update the state and counters
            s_t = s_t1
            r_t_p = r_t
            T += 1
            t += 1
            if changed:
                t_run += 1
            ep_t += 1
            ep_reward += r_t
            #episode_ave_max_q += np.max(readout_t)

            # Optionally update online network

            if t % I_AsyncUpdate == 0 or terminal:
                """
                if s_batch:
                    session.run(grad_update, feed_dict={y: y_batch,
                                                        a: a_batch,
                                                        s: s_batch})
                """
                #if is_checkable:
                    #session.run(grad_update0, feed_dict={y0: y_batch0,
                    #                                    a0: [[1,],],
                    #                                    s0: s_batch_l0})

                #if is_checkable:
                #    session.run(grad_update1, feed_dict={y1: y_batch1,
                #                                        a1: a_batch_1,
                #                                        s1: s_batch_l1})
                if is_checkable:
                    #session.run(grad_update2, feed_dict={y2: y_batch2,
                    #                                    a2: [[1,],],
                    #                                    s2: s_batch_l2})

                    append_data = TrainSet(s_batch_l0, [[1, ], ], r_batch)
                    # append_data = TrainSet(s_batch, a_batch, r_batch, s_  batch_l0, total_reward=y_batch)
                    for values in r_batch:
                        pickle.dump([values.input_matrix, values.output_matrix], file, protocol=pickle.HIGHEST_PROTOCOL)


                is_checkable = False



                # Clear gradients
                #s_batch = []
                #a_batch = []
                #y_batch = []
                y_batch0 = []
                #y_batch1 = []
                y_batch2 = []
                s_batch_l0 = []

                s_batch_l2 = []
                r_batch = []

            # Save model progress
            if t % checkpoint_interval == 0:
                saver.save(session, "qlearning.ckpt", global_step=t)

            # Print end of episode stats
            if terminal:
                env.reset()
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                for i in range(len(stats)):
                    session.run(assign_ops[i],
                                {summary_placeholders[i]: float(stats[i])})
                print("| Thread {:2d}".format(int(thread_id)), "| Step", t, "| Stepreal", t_run,
                      "| Reward: {:3.4f}".format(float(ep_reward)), " Qmax: {:5.4f}".format(
                      (episode_ave_max_q/float(ep_t))),
                      " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                      (t/float(anneal_epsilon_timesteps)))
                break

    file.close()


def build_models(num_actions):
    # Create shared deep q network
    #s, q_network = build_dqn(num_actions=num_actions,
    #                         action_repeat=action_repeat)
    s, q_network = build_dqn_fully(20, 20, action_repeat, num_actions)
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    #st, target_q_network = build_dqn(num_actions=num_actions,
    #                                 action_repeat=action_repeat)
    st, target_q_network = build_dqn_fully(20, 20, action_repeat, num_actions)
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # Op for periodically updating target network with online network weights
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)

    return s, q_values, st, target_q_values, reset_target_network_params, a, y, grad_update

def build_exact_model():
    # Create shared deep q network
    #s, q_network = build_dqn(num_actions=num_actions,
    #                         action_repeat=action_repeat)
    s0, q_network0 = build_dqn_exact_fully(20, 20, action_repeat, 1)
    network_params0 = tf.trainable_variables()
    q_values0 = q_network0

    # Define cost and gradient update op
    a0 = tf.placeholder("float", [None, 1])
    y0 = tf.placeholder("float", [None])
    action_q_values0 = tf.reduce_sum(tf.multiply(q_values0, a0), reduction_indices=1)
    cost0 = tflearn.mean_square(action_q_values0, y0)
    optimizer0 = tf.train.RMSPropOptimizer(learning_rate)

    grad_update0 = optimizer0.minimize(cost0, var_list=network_params0)

    return s0, q_values0, a0, y0, grad_update0

def build_resnext_model():
    # Create shared deep q network
    #s, q_network = build_dqn(num_actions=num_actions,
    #                         action_repeat=action_repeat)
    #s0, q_network0 = build_resnext()
    s0, q_network0 = build_dqn_fully(10, 10, 6, 100)
    network_params0 = tf.trainable_variables()
    q_values0 = q_network0

    # Define cost and gradient update op
    a0 = tf.placeholder("float", [None, 100])
    y0 = tf.placeholder("float", [None])
    action_q_values0 = tf.reduce_sum(tf.multiply(q_values0, a0), reduction_indices=1)
    cost0 = tflearn.mean_square(action_q_values0, y0)
    optimizer0 = tf.train.RMSPropOptimizer(learning_rate)

    grad_update0 = optimizer0.minimize(cost0, var_list=network_params0)

    return s0, q_values0, a0, y0, grad_update0


def build_graph(num_actions):
    """
    # Create shared deep q network
    s, q_network = build_dqn(num_actions=num_actions,
                             action_repeat=action_repeat)
    network_params = tf.trainable_variables()
    q_values = q_network

    # Create shared target network
    st, target_q_network = build_dqn(num_actions=num_actions,
                                     action_repeat=action_repeat)
    target_network_params = tf.trainable_variables()[len(network_params):]
    target_q_values = target_q_network

    # Op for periodically updating target network with online network weights
    reset_target_network_params = \
        [target_network_params[i].assign(network_params[i])
         for i in range(len(target_network_params))]

    # Define cost and gradient update op
    a = tf.placeholder("float", [None, num_actions])
    y = tf.placeholder("float", [None])
    action_q_values = tf.reduce_sum(tf.multiply(q_values, a), reduction_indices=1)
    cost = tflearn.mean_square(action_q_values, y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    grad_update = optimizer.minimize(cost, var_list=network_params)
    """
    #s, q_values, st, target_q_values, reset_target_network_params, a, y, grad_update = build_models(num_actions)
    #s0, q_values0, a0, y0, grad_update0 = build_exact_model()

    s, q_values, a, y, grad_update = build_resnext_model()
    #s, q_values, a, y, grad_update = build_dqn_fully(10, 10, 6, 100)

    #s1, q_values1, a1, y1, grad_update1 = build_exact_model()
    #s2, q_values2, a2, y2, grad_update2 = build_exact_model()
    """graph_ops = {
                 "s0": s0,
                 "q_values0": q_values0,
                 "a0": a0,
                 "y0": y0,
                 "grad_update0": grad_update0,
                 "s2": s2,
                 "q_values2": q_values2,
                 "a2": a2,
                 "y2": y2,
                 "grad_update2": grad_update2
                 }
                 
    """
    graph_ops = {
                 "s": s,
                 "q_values": q_values,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update
                 }
    """
    graph_ops = {"s": s,
                 "q_values": q_values,
                 "st": st,
                 "target_q_values": target_q_values,
                 "reset_target_network_params": reset_target_network_params,
                 "a": a,
                 "y": y,
                 "grad_update": grad_update,
                 "s0": s0,
                 "q_values0": q_values0,
                 "a0": a0,
                 "y0": y0,
                 "grad_update0": grad_update0,
                 "s1": s1,
                 "q_values1": q_values1,
                 "a1": a1,
                 "y1": y1,
                 "grad_update1": grad_update1,
                 "s2": s2,
                 "q_values2": q_values2,
                 "a2": a2,
                 "y2": y2,
                 "grad_update2": grad_update2
                 }
    """
    return graph_ops


# Set up some episode summary ops to visualize on tensorboard.
def build_summaries():
    episode_reward = tf.Variable(0.)
    scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    scalar_summary("Qmax Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    scalar_summary("Epsilon", logged_epsilon)
    # Threads shouldn't modify the main graph, so we use placeholders
    # to assign the value of every summary (instead of using assign method
    # in every thread, that would keep creating new ops in the graph)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = merge_all_summaries()
    return summary_placeholders, assign_ops, summary_op


def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    # Figure out number of actions from gym env

    num_actions = int(10 * (10 - 1) / 2 + 10)*int(10 * (10 - 1) / 2 + 10)
    return num_actions


def train(session, graph_ops, num_actions, saver):
    """
    Train a model.
    """

    summary_ops = build_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.global_variables_initializer())
    writer = writer_summary(summary_dir + "/qlearning", session.graph)

    # Initialize target network weights
    #session.run(graph_ops["reset_target_network_params"])

    #queue for ploting
    q = queue.Queue()

    # Start n_threads actor-learner training threads
    actor_learner_threads = \
        [threading.Thread(target=actor_learner_thread,
                          args=(thread_id, None, session,
                                graph_ops, num_actions, summary_ops, saver, q))
         for thread_id in range(n_threads)]
    for t in actor_learner_threads:
        t.start()
        time.sleep(0.01)

    # Show the agents training and write summary statistics

    for t in actor_learner_threads:
        t.join()


def evaluation(session, graph_ops, saver):
    """
    Evaluate a model.
    """
    saver.restore(session, test_model_path)
    print("Restored model weights from ", test_model_path)
    monitor_env = gym.make(game)
    monitor_env.monitor.start("qlearning/eval")

    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env,
                           action_repeat=action_repeat)

    for i_episode in range(num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session=session, feed_dict={s : [s_t]})
            action_index = np.argmax(readout_t)
            s_t1, r_t, info, terminal = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)
    monitor_env.monitor.close()


def main(_):
    with tf.Session() as session:
        num_actions = get_num_actions()
        graph_ops = build_graph(num_actions)
        saver = tf.train.Saver(max_to_keep=5)

        if testing:
            evaluation(session, graph_ops, saver)
        else:
            train(session, graph_ops, num_actions, saver)

if __name__ == "__main__":
    tf.app.run()

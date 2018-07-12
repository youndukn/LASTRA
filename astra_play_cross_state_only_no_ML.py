from __future__ import division, print_function, absolute_import

import threading, queue
import random
import numpy as np
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

import glob
import gym

from data.astra_train_set import AstraTrainSet, TrainSet
from astra import Astra
import pickle
import os
from shutil import copyfile
import copy
import tensorflow as tf
import keras
from multiprocessing import Process


# Fix for TF 0.12

# Change that value to test instead of train
testing = False
# Model path (to load when testing)
test_model_path = './qlearning.tflearn.ckpt'
# Atari game to learn
# You can also try: 'Breakout-v0', 'Pong-v0', 'SpaceInvaders-v0', ...
game = 'Astra'
# Learning threads
n_threads = 7

# =============================
#   Training Parameters
# =============================
# Max training steps
TMAX = 1000
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


dictionary = {}

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


def actor_learner_thread(thread_id, env, num_actions, saver, q, input_directory):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX, T

    directory = "{}{}{}".format(input_directory, os.path.sep, thread_id)

    if not os.path.exists(directory):
        os.makedirs(directory)

    #env = Astra("..{}ASTRA{}01_astra_y310_nom_boc.job".format(os.path.sep, os.path.sep), working_directory=directory)
    input_name = glob.glob("{}01_*.inp".format(input_directory))

    env = Astra(input_name[0], main_directory = input_directory, working_directory=directory)

    file = open('/media/youndukn/lastra/plants_data_1/{}_data_{}'.format(input_name[0].replace(input_directory, ""), thread_id), 'wb')

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

    resuable = {}

    satisfied_data = open('satisfied_numb.txt', 'w')
    t_run = 0
    is_checkable = False
    while T < TMAX:

        # Set up per-episode counters
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0
        r_t_p = 0

        while True:

            posibilities = random.randrange(num_actions+3+2)
            position = random.randrange(num_actions)
            if posibilities < num_actions:
                action_index = (position*num_actions)+posibilities
                s_t1, r_t, changed, info, satisfied = env.step_shuffle(action_index, [0, 1, 4])
            elif posibilities >= num_actions and posibilities < num_actions+3:
                action_index = (position * 3) + (posibilities-num_actions)
                s_t1, r_t, changed, info, satisfied = env.step_rotate(action_index, [0, 1, 4])
            elif posibilities >= num_actions+3 and posibilities < num_actions+5:
                action_index = (position*2) + (posibilities-num_actions-3)
                s_t1, r_t, changed, info, satisfied = env.step_bp(action_index, [0, 1, 4])
            """
            glob_action = random.randrange(num_actions*(num_actions+3+2))

            numb = int(glob_action/(num_actions+3+2))
            posibilities = glob_action%(num_actions+3+2)
            """
            satisfied_data.write("{}".format(T))
            s_t1 = AstraTrainSet(s_t1, None, None, not info, r_t).state

            terminal = not info

            # Scale down epsilon
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

            clipped_r_t = np.clip((r_t - r_t_p)*100, -1, 1)
            #y_batch.append(clipped_r_t)
            if changed and not terminal:
                is_checkable = True
                if env.cross_set:
                    r_batch.append(env.cross_set)

            #print("| Thread {:2d}".format(int(thread_id)), "| Step", t,
            #      "| Reward: {:4d}".format(int(clipped_r_t)), "| Batch: {:4d}".format(int(y_batch[-1])))

            #a_batch.append(a_t)
            #s_batch.append(s_t)

            # Update the state and counters
            s_t = s_t1
            r_t_p = r_t
            t += 1
            if changed:
                T += 1
                t_run += 1
            ep_t += 1
            ep_reward += r_t
            #episode_ave_max_q += np.max(readout_t)

            # Optionally update online network

            if t % I_AsyncUpdate == 0 or terminal:

                if is_checkable:
                    append_data = TrainSet(s_batch_l0, [[1, ], ], r_batch)
                    # append_data = TrainSet(s_batch, a_batch, r_batch, s_batch_l0, total_reward=y_batch)
                    for values in r_batch:
                        dump_list = []

                        first_step = values[0]
                        state = first_step.input_tensor_full
                        key = hash(state.tostring())

                        if not (key in resuable):
                            for value in values:
                                a_list = [value.summary_tensor, value.input_tensor_full, value.output_tensor, value.flux_tensor, value.density_tensor_full]
                                dump_list.append(a_list)
                            pickle.dump(dump_list, file, protocol=pickle.HIGHEST_PROTOCOL)
                            dictionary[key] = 1
                        else:
                            dictionary[key] += 1


                is_checkable = False

                y_batch0 = []
                y_batch2 = []
                s_batch_l0 = []

                s_batch_l2 = []
                r_batch = []

            # Print end of episode stats
            if terminal:
                env.reset()
                stats = [ep_reward, episode_ave_max_q/float(ep_t), epsilon]
                print("| Thread {:2d}".format(int(thread_id)), "| Step", t, "| Stepreal", t_run,
                      "| Reward: {:3.4f}".format(float(ep_reward)), " Qmax: {:5.4f}".format(
                      (episode_ave_max_q/float(ep_t))),
                      " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                      (t/float(anneal_epsilon_timesteps)))
                break

    file.close()

def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    # Figure out number of actions from gym env

    num_actions = int(10 * (10 - 1) / 2 + 10)
    return num_actions


def train(num_actions):
    """
    Train a model.
    """


    q = queue.Queue()

    main_directory = "/home/youndukn/Plants/1.4.0/"
    directories = ["ucn4", "ygn3"]
    """
    for subDirectory in ['c{0:02}'.format(x) for x in range(5, 6)]:

        the_directory = "{}{}{}{}{}depl{}".format(main_directory,
                                              "ucn4",
                                              os.path.sep,
                                              subDirectory,
                                              os.path.sep,
                                              os.path.sep)

        # Start n_threads actor-learner training threads
        actor_learner_threads = \
            [Process(target=actor_learner_thread,
                     args=(thread_id, None, num_actions, None, q, the_directory))
             for thread_id in range(n_threads)]
        for t in actor_learner_threads:
            t.start()
            time.sleep(0.01)

        for t in actor_learner_threads:
            t.join()
    """
    for subDirectory in ['c{0:02}'.format(x) for x in range(5, 16)]:

        the_directory = "{}{}{}{}{}depl{}".format(main_directory,
                                              "ygn3",
                                              os.path.sep,
                                              subDirectory,
                                              os.path.sep,
                                              os.path.sep)

        # Start n_threads actor-learner training threads
        actor_learner_threads = \
            [Process(target=actor_learner_thread,
                     args=(thread_id, None, num_actions, None, q, the_directory))
             for thread_id in range(n_threads)]
        for t in actor_learner_threads:
            t.start()
            time.sleep(0.01)

        for t in actor_learner_threads:
            t.join()


def main(_):
    num_actions = get_num_actions()
    train(num_actions)

if __name__ == "__main__":
    main(None)

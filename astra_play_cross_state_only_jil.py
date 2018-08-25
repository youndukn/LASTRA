from __future__ import division, print_function, absolute_import

import threading, queue
import numpy as np
import time

import glob

from data.astra_train_set import AstraTrainSet
from astra import Astra
import pickle
import os
import enviroments
import a3c_non


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
I_AsyncUpdate = 30
# Async gradient update frequency of each learning thread
I_ModelUpdate = 30
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

cl_base = 15000
fxy_base = 1.55

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

def my_input_output(values):
    burnup_boc = values[0]
    burnup_eoc = values[-1]
    s_batch_init = burnup_boc.input_tensor_full
    s_batch_init_den = burnup_boc.density_tensor_full
    s_batch_init_den = np.array(s_batch_init_den)
    my_state = np.concatenate((s_batch_init, s_batch_init_den), axis=2)
    my_cl = burnup_eoc.summary_tensor[0]

    my_fxy = 0

    for burnup_point in values:
        if my_fxy < burnup_point.summary_tensor[5]:
            my_fxy = burnup_point.summary_tensor[5]
    return my_state, my_cl, my_fxy

def actor_learner_thread(target, actor_critic, thread_id, env, num_actions, saver, q, input_directory):
    """
    Actor-learner thread implementing asynchronous one-step Q-learning, as specified
    in algorithm 1 here: http://arxiv.org/pdf/1602.01783v1.pdf.
    """
    global TMAX
    directory =  "{}{}{}".format(input_directory, os.path.sep, thread_id)

    if not os.path.exists(directory):
        os.makedirs(directory)

    #env = Astra("..{}ASTRA{}01_astra_y310_nom_boc.job".format(os.path.sep, os.path.sep), working_directory=directory)
    input_name = glob.glob("{}01_*.inp".format(input_directory))

    env = Astra(input_name[0], main_directory = input_directory, working_directory=directory)

    file = open('/media/youndukn/lastra/plants_data3/{}_data_{}'.format(input_name[0].replace(input_directory, ""), thread_id), 'wb')

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.3
    epsilon = 0.3

    r_batch = []

    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))

    time.sleep(thread_id)

    t = 0
    ep_t = 0
    T = 0
    ep_T = 0

    ep_r_t = 0
    ep_r_T = 0

    ep_reward = 0


    my_state, my_cl, my_fxy = my_input_output(env.cross_set)

    while T < TMAX:

        my_state = my_state.reshape((1, actor_critic.env.observation_space.shape[0], actor_critic.env.observation_space.shape[1], actor_critic.env.observation_space.shape[2]))

        action, isRandom = actor_critic.act(my_state, thread_id)
        max_action = np.argmax(action)
        posibilities = max_action%(num_actions+3+2)
        position = int(max_action/(num_actions+3+2))
        action = action.reshape((1, actor_critic.env.action_space.shape[0]))

        """
        posibilities = random.randrange(num_actions+3+2)
        position = random.randrange(num_actions)
        """
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

        s_t1 = AstraTrainSet(s_t1, None, None, not info, r_t).state

        terminal = not info

        # Scale down epsilon
        if epsilon > final_epsilon:
            epsilon -= (initial_epsilon - final_epsilon) / anneal_epsilon_timesteps

        #y_batch.append(clipped_r_t)

        pre_state = my_state
        pre_cl = my_cl
        pre_fxy = my_fxy

        my_state, my_cl, my_fxy = my_input_output(env.cross_set)
        my_state = my_state.reshape((1,
                                     actor_critic.env.observation_space.shape[0],
                                     actor_critic.env.observation_space.shape[1],
                                     actor_critic.env.observation_space.shape[2]))

        reward = 100 * (min(target[0], my_cl) - min(target[0], pre_cl)) / cl_base + \
                 100 * (max(target[1], pre_fxy) - max(target[1], my_fxy)) / fxy_base

        if terminal:
            reward = -1

        if changed:
            if not terminal:
                if env.cross_set:
                    r_batch.append(env.cross_set)

        t += 1
        ep_t += 1

        if not isRandom:
            ep_r_t += 1

        if changed:
            T += 1
            ep_T += 1
            if not isRandom:
                ep_r_T += 1

        ep_reward += reward

        if len(r_batch) % I_AsyncUpdate == I_AsyncUpdate-1 or terminal:

            for index, values in enumerate(r_batch):

                dump_list = []

                for value in values:
                    a_list = [value.summary_tensor,
                              value.input_tensor_full,
                              value.output_tensor,
                              value.flux_tensor,
                              value.density_tensor_full,
                              index]
                    dump_list.append(a_list)
                pickle.dump(dump_list, file, protocol=pickle.HIGHEST_PROTOCOL)

            print("| Thread {:2d}".format(int(thread_id)),
                  "| GS {:5d}".format(t), "| GSR {:5d}".format(T),
                  "| ES {:5d}".format(ep_t),"| ESR {:5d}".format(ep_T), "| ESRR {:5d}".format(ep_r_t),
                  "| Reward: {:3.2f} |".format(float(ep_reward)))

            r_batch = []
            env.reset()
            ep_reward = 0
            ep_T = 0
            ep_t = 0
            ep_r_t = 0

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
    global graph

    q = queue.Queue()
    env = enviroments.Environment()
    actor_critic = a3c_non.ActorCritic(env)

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
    for subDirectory in ['c{0:02}'.format(x) for x in range(5, 13)]:
        print(subDirectory)
        the_directory = "{}{}{}{}{}depl{}".format(main_directory,
                                              "ygn3",
                                              os.path.sep,
                                              subDirectory,
                                              os.path.sep,
                                              os.path.sep)

        # Start n_threads actor-learner training threads
        actor_learner_threads = \
            [threading.Thread(target=actor_learner_thread,
                     args=((17100, 1.52), actor_critic, thread_id, None, num_actions, None, q, the_directory))
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

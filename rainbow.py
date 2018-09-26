import numpy as np

import os

from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
from dopamine.colab import utils as colab
from absl import flags


BASE_PATH = '/tmp/colab_dope_run'
GAME = 'astra'

LOG_PATH = os.path.join(BASE_PATH, 'basic_agent', GAME)


class BasicAgent(object):
    """This agent randomly selects an action and sticks to it. It will change
    actions with probability switch_prob."""

    def __init__(self, sess, num_actions, switch_prob=0.1):
        # tensorflow session
        self._sess = sess
        # how many possible actions can it take?
        self._num_actions = num_actions
        # probability of switching actions in the next timestep?
        self._switch_prob = switch_prob
        # initialize the action to take (randomly)
        self._last_action = np.random.randint(num_actions)
        # not debugging
        self.eval_mode = False

    # How select an action?
    # we define our policy here
    def _choose_action(self):
        if np.random.random() <= self._switch_prob:
            self._last_action = np.random.randint(self._num_actions)
        return self._last_action

    # when it checkpoints during training, anything we should do?
    def bundle_and_checkpoint(self, unused_checkpoint_dir, unused_iteration):
        pass

    # loading from checkpoint
    def unbundle(self, unused_checkpoint_dir, unused_checkpoint_version,
                 unused_data):
        pass

    # first action to take
    def begin_episode(self, unused_observation):
        return self._choose_action()

    # cleanup
    def end_episode(self, unused_reward):
        pass

    # we can update our policy here
    # using the reward and observation
    # dynamic programming, Q learning, monte carlo methods, etc.
    def step(self, reward, observation):
        return self._choose_action()

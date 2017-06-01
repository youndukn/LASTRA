from astra import Astra
import numpy as np
from queue import Queue
import copy
import os
from threading import Thread
import matplotlib.pyplot as plt

from data.astra_train_set import AstraTrainSet
import time
from dqn import LogQValues, LogReward, EpsilonGreedy, LinearControlSignal, ReplayMemory, NeuralNetwork

class ReinforcementLearning():
    learning_rate = 1e-3
    goal_steps = 200
    initial_games = 12000
    score_requirement = 0
    gamma = 0.95

    def __init__(self, thread_numb, astra, target_rewards,
                 env_name="hello", training=True, use_logging=True):

        # The astra runner to work with
        self.astra = astra

        # The number of Thread to work with
        self.thread_numb = thread_numb

        # The target reward
        self.target_rewards = target_rewards

        # The number of calculations
        self.cal_numb = 0

        # The number of possible actions that the agent may take in every step.
        self.num_actions = int(self.astra.max_position * self.astra.n_move)

        # The number of possible actions that the agent may take in shuffle step
        self.num_actions_2 = int(self.astra.max_position)

        # Whether we are training (True) or testing (False).
        self.training = training

        # Whether to use logging during training.
        self.use_logging = use_logging

        if self.use_logging and self.training:
            # Used for logging Q-values and rewards during training.
            self.log_q_values = LogQValues()
            self.log_reward = LogReward()
        else:
            self.log_q_values = None
            self.log_reward = None

        # Epsilon-greedy policy for selecting an action from the Q-values.
        # During training the epsilon is decreased linearly over the given
        # number of iterations. During testing the fixed epsilon is used.
        self.epsilon_greedy = EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=1e6,
                                            num_actions=self.num_actions,
                                            epsilon_testing=0.01)

        # Epsilon-greedy policy for selecting an action from the Q-values.
        # During training the epsilon is decreased linearly over the given
        # number of iterations. During testing the fixed epsilon is used.
        self.epsilon_greedy_2 = EpsilonGreedy(start_value=0.1,
                                            end_value=0.01,
                                            num_iterations=1e6,
                                            num_actions=self.num_actions_2,
                                            epsilon_testing=0.01)

        if self.training:
            # The following control-signals are only used during training.

            # The learning-rate for the optimizer decreases linearly.
            self.learning_rate_control = LinearControlSignal(start_value=1e-2,
                                                             end_value=1e-3,
                                                             num_iterations=5e6)

            # The loss-limit is used to abort the optimization whenever the
            # mean batch-loss falls below this limit.
            self.loss_limit_control = LinearControlSignal(start_value=0.1,
                                                          end_value=0.015,
                                                          num_iterations=5e6)

            # The maximum number of epochs to perform during optimization.
            # This is increased from 5 to 10 epochs, because it was found for
            # the Breakout-game that too many epochs could be harmful early
            # in the training, as it might cause over-fitting.
            # Later in the training we would occasionally get rare events
            # and would therefore have to optimize for more iterations
            # because the learning-rate had been decreased.
            self.max_epochs_control = LinearControlSignal(start_value=5.0,
                                                          end_value=10.0,
                                                          num_iterations=5e6)

            # The fraction of the replay-memory to be used.
            # Early in the training, we want to optimize more frequently
            # so the Neural Network is trained faster and the Q-values
            # are learned and updated more often. Later in the training,
            # we need more samples in the replay-memory to have sufficient
            # diversity, otherwise the Neural Network will over-fit.
            self.replay_fraction = LinearControlSignal(start_value=0.1,
                                                       end_value=1.0,
                                                       num_iterations=5e6)
        else:
            # We set these objects to None when they will not be used.
            self.learning_rate_control = None
            self.loss_limit_control = None
            self.max_epochs_control = None
            self.replay_fraction = None

        if self.training:
            # We only create the replay-memory when we are training the agent,
            # because it requires a lot of RAM. The image-frames from the
            # game-environment are resized to 105 x 80 pixels gray-scale,
            # and each state has 2 channels (one for the recent image-frame
            # of the game-environment, and one for the motion-trace).
            # Each pixel is 1 byte, so this replay-memory needs more than
            # 3 GB RAM (105 x 80 x 2 x 200000 bytes).
            self.replay_memory = ReplayMemory(size=10000,
                                              num_actions=self.num_actions)
        else:
            self.replay_memory = None

        # Create the Neural Network used for estimating Q-values.
        self.model = NeuralNetwork(num_actions=self.num_actions,
                                   replay_memory=self.replay_memory)

        self.model_2 = NeuralNetwork(num_actions=self.num_actions_2,
                                     replay_memory=self.replay_memory)

        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []

    def initial_population(self):

        self.astra.reset()

        # Set up some global variables
        enclosure_queue = Queue()

        out_queue = Queue()

        # Create
        for i in range(self.thread_numb):

            # Create queues according to the thread number
            oct_move = self.astra.get_oct_move_action()
            second_move = self.astra.get_oct_shuffle_space(oct_move) if oct_move % Astra.n_move == self.astra.shuffle else None
            enclosure_queue.put([oct_move, second_move])

            # Create astra according to the thread number
            new_astra = copy.deepcopy(self.astra)

            directory = ".{}{}".format(os.path.sep, i)
            if not os.path.exists(directory):
                os.makedirs(directory)

            new_astra.working_directory = directory

            # Create threads according to the thread number
            worker = Thread(target=self.update_dev, args=(new_astra, enclosure_queue, out_queue,))
            worker.setDaemon(True)
            worker.start()

        trainer = Thread(target=self.learning, args=(enclosure_queue, out_queue,))
        trainer.setDaemon(True)
        trainer.start()

        # Now wait for the queue to be empty, indicating that we have
        # processed all of the downloads.)
        print('*** Main thread waiting')
        """
                while True:

                    time.sleep(10)

                    for i in range(self.replay_memory.num_used):

                        state = self.replay_memory.states[i]
                        H = np.zeros((10, 10))  # added some commas and array creation code

                        q_values = self.model.get_q_values(states=[state])[0]
                        counter = 0
                        for q_i in range(10):
                            for q_j in range(q_i, 10):
                                H[q_i, q_j] = q_values[counter]
                                counter += 1

                        fig = plt.figure(figsize=(6, 3.2))

                        ax = fig.add_subplot(111)
                        ax.set_title('colorMap')
                        plt.imshow(H)
                        ax.set_aspect('equal')

                        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
                        cax.get_xaxis().set_visible(False)
                        cax.get_yaxis().set_visible(False)
                        cax.patch.set_alpha(0)
                        cax.set_frame_on(False)
                        plt.colorbar(orientation='vertical')
                        plt.show()

                        if i > 0 and self.replay_memory.end_episode[i]:
                            break
        """
        enclosure_queue.join()
        print('*** Done')

    def update_dev(self, astra, queue, out_queue):
        while True:
            if self.cal_numb < ReinforcementLearning.initial_games:

                oct_move = astra.get_oct_move_action()
                second_move = astra.get_oct_shuffle_space(
                    oct_move) if oct_move % Astra.n_move == astra.shuffle else None
                queue.put([oct_move, second_move])

            points = queue.get()

            count_states = self.model.get_count_states()

            # Create another
            # Use the Neural Network to estimate the Q-values for the state.
            # Note that the function assumes an array of states and returns
            # a 2-dim array of Q-values, but we just have a single state here.
            q_values = self.model.get_q_values(states=[astra.change_data[len(astra.change_data)-1].state])[0]

            # Determine the action that the agent must take in the game-environment.
            # The epsilon is just used for printing further below.
            action, epsilon = self.epsilon_greedy.get_action(q_values=q_values,
                                                             iteration=count_states,
                                                             training=self.training)
            points[0] = action
            # Run astra to get lists
            core, lists, changed, info = astra.change(points[0], points[1])

            if changed and info:

                if len(astra.change_data) > 50:
                    astra.change_data.append(AstraTrainSet(core, points, lists, True, None))
                    out_queue.put(astra.change_data)
                    astra.reset()
                else:
                    astra.change_data.append(AstraTrainSet(core, points, lists, False, None))
                self.cal_numb += 1

            if changed and not info:
                if lists:
                    astra.change_data.append(AstraTrainSet(core, points, lists, True, None))
                    out_queue.put(astra.change_data)
                astra.reset()
            queue.task_done()

    def learning(self, queue, out_queue):

        while True:
            if out_queue.empty():
                time.sleep(5)
            else:
                pre_reward = None
                lists = out_queue.get()

                cumulative_rewards = 0

                for i, train_set in enumerate(lists):

                    count_episodes = self.model.get_count_episodes()

                    total_reward = 0

                    if pre_reward:
                        all_satisfied = True
                        if train_set.reward:
                            for j in range(len(train_set.reward)):
                                if j == 0:
                                    if self.target_rewards[j] > train_set.reward[j]:
                                        all_satisfied = False
                                    total_reward = total_reward + \
                                                   (max(self.target_rewards[j], pre_reward[j]) - \
                                                    max(self.target_rewards[j], train_set.reward[j])) / \
                                                   self.target_rewards[j]*1000
                                else:
                                    if self.target_rewards[j] < train_set.reward[j]:
                                        all_satisfied = False
                                    total_reward = total_reward + \
                                                   (min(self.target_rewards[j], train_set.reward[j]) -
                                                    min(self.target_rewards[j], pre_reward[j])) / \
                                                   self.target_rewards[j]*1000
                        if all_satisfied:
                            print("This one is good")
                        train_set.total_reward = total_reward
                        total_reward = train_set.total_reward
                        done = train_set.done
                    else:
                        done = False

                    cumulative_rewards += total_reward

                    pre_reward = train_set.reward

                    """                   for a_list in train_set.input.get_value_matrix(4):
                                           print(a_list)
                   """

                    #print([prettyfloat(n) for n in train_set.reward])

                    #print(np.subtract(pre_state2[0], train_set.state2[0]))
                    #print(np.subtract(pre_state2[1], train_set.state2[1]))
                    #print(np.subtract(pre_state2[2], train_set.state2[2]))

                    #pre_state2 = train_set.state2
                    if i > 0:
                        first_move = lists[i].output[0]

                        if lists[i].output[1]:
                            second_move = lists[i].output[1]
                        else:
                            second_move = -1

                        count_states = self.model.increase_count_states()

                        q_values = self.model.get_q_values(states=[lists[i-1].state])[0]

                        print(prettyfloat(cumulative_rewards), prettyfloat(total_reward), first_move, q_values[first_move])

                        self.replay_memory.add(state=lists[i-1].state,
                                               q_values=q_values,
                                               action=first_move,
                                               action2=second_move,
                                               reward=total_reward,
                                               end_life=done,
                                               end_episode=done)

                        # How much of the replay-memory should be used.
                        use_fraction = self.replay_fraction.get_value(iteration=count_states)

                        if train_set.done:
                            pre_reward = None
                            count_episodes = self.model.increase_count_episodes()
                            print("")

                        # When the replay-memory is sufficiently full.
                        if self.replay_memory.is_full() \
                                or self.replay_memory.used_fraction() > use_fraction:

                            # Update all Q-values in the replay-memory through a backwards-sweep.
                            self.replay_memory.update_all_q_values()

                            # Log statistics for the Q-values to file.
                            if self.use_logging:
                                self.log_q_values.write(count_episodes=count_episodes,
                                                        count_states=count_states,
                                                        q_values=self.replay_memory.q_values)

                            # Get the control parameters for optimization of the Neural Network.
                            # These are changed linearly depending on the state-counter.
                            learning_rate = self.learning_rate_control.get_value(iteration=count_states)
                            loss_limit = self.loss_limit_control.get_value(iteration=count_states)
                            max_epochs = self.max_epochs_control.get_value(iteration=count_states)

                            # Perform an optimization run on the Neural Network so as to
                            # improve the estimates for the Q-values.
                            # This will sample random batches from the replay-memory.
                            self.model.optimize(learning_rate=learning_rate,
                                                loss_limit=loss_limit,
                                                max_epochs=max_epochs)

                            # Save a checkpoint of the Neural Network so we can reload it.
                            self.model.save_checkpoint(count_states)

                            # Reset the replay-memory. This throws away all the data we have{
                            # just gathered, so we will have to fill the replay-memory again.
                            self.replay_memory.reset()

class prettyfloat(float):

    def __repr__(self):
        return "%0.3e" % self


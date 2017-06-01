from dqn import LogQValues, LogReward, EpsilonGreedy, LinearControlSignal, ReplayMemory, NeuralNetwork
import numpy as np

class TrainingClass():

    def __init__(self, astra, target_rewards,
                 env_name="hello", training=True, use_logging=True):

        # The astra runner to work with
        self.astra = astra

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

        # Log of the rewards obtained in each episode during calls to run()
        self.episode_rewards = []
        while True:

            replay_memory = ReplayMemory.load_replay_memories()
            replay_memory.rewards = np.multiply(replay_memory.rewards, 10)
            replay_memory.update_all_q_values()

            self.model = NeuralNetwork(num_actions=self.num_actions,
                                       replay_memory=replay_memory)

            count_states = replay_memory.num_used
            # Log statistics for the Q-values to file.
            if self.use_logging:
                self.log_q_values.write(count_episodes=0,
                                        count_states=count_states,
                                        q_values=replay_memory.q_values)

            # Get the control parameters for optimization of the Neural Network.
            # These are changed linearly depending on the state-counter.
            learning_rate = self.learning_rate_control.get_value(iteration=count_states)
            loss_limit = self.loss_limit_control.get_value(iteration=count_states)
            max_epochs = self.max_epochs_control.get_value(iteration=count_states)

            self.model.optimize(learning_rate=learning_rate,
                                loss_limit=loss_limit,
                                max_epochs=max_epochs)

            print(self.model.get_q_values(replay_memory.states[0]))



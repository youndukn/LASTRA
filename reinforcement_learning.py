from astra import Astra
import numpy as np
from queue import Queue
import copy
import os
from threading import Thread

from data.astra_train_set import AstraTrainSet
import time
from convolutional import SimpleConvolutional

class ReinforcementLearning():
    learning_rate = 1e-3
    goal_steps = 200
    initial_games = 10000
    score_requirement = 0
    gamma = 0.95

    def __init__(self, thread_numb, astra, dev, input_data_name=None, output_data_name=None):

        self.astra = astra
        self.thread_numb = thread_numb
        self.dev = np.array(dev)

        if input_data_name:
            self.output_array = np.load(input_data_name)

        if output_data_name:
            self.input_matrix = np.load(output_data_name)

        self.cal_numb = 0

        self.model = SimpleConvolutional()

    def initial_population(self):
        print("start")
        astra = self.astra

        astra.reset()

        # Set up some global variables
        num_fetch_threads = self.thread_numb
        enclosure_queue = Queue()

        out_queue = Queue()

        # Create
        for i in range(num_fetch_threads):

            # Create queues according to the thread number
            oct_move = astra.get_oct_move_action()
            second_move = astra.get_oct_shuffle_space(oct_move) if oct_move % Astra.n_move == astra.shuffle else None
            enclosure_queue.put([oct_move, second_move])

            # Create astra according to the thread number
            new_astra = copy.deepcopy(astra)

            directory = ".{}{}".format(os.path.sep, i)
            if not os.path.exists(directory):
                os.makedirs(directory)

            new_astra.working_directory = directory

            # Create threads according to the thread number
            worker = Thread(target=self.update_dev, args=(new_astra, enclosure_queue, out_queue,))
            worker.setDaemon(True)
            worker.start()

        worker = Thread(target=self.learning, args=(enclosure_queue, out_queue,))
        worker.setDaemon(True)
        worker.start()

        # Now wait for the queue to be empty, indicating that we have
        # processed all of the downloads.)
        print('*** Main thread waiting')
        enclosure_queue.join()
        print('*** Done')

    def update_dev(self, astra, queue, out_queue):
        while True:
            if self.cal_numb < ReinforcementLearning.initial_games:
                # Create another queue
                oct_move = astra.get_oct_move_action()
                second_move = astra.get_oct_shuffle_space(
                    oct_move) if oct_move % Astra.n_move == astra.shuffle else None
                queue.put([oct_move, second_move])

            points = queue.get()

            # Run astra to get lists
            core, lists, changed, info = astra.change(points[0], points[1])

            if changed and info:
                astra.change_data.append(AstraTrainSet(core, points, lists))
                if len(astra.change_data) > 100:
                    out_queue.put(astra.change_data)
                    astra.reset()
                self.cal_numb += 1
                print(lists)

            if changed and not info:
                if lists:
                    astra.change_data.append(AstraTrainSet(core, points, lists))
                    out_queue.put(astra.change_data)
                astra.reset()

            queue.task_done()

    def learning(self, queue, out_queue):

        while not queue.empty():
            if out_queue.empty():
                time.sleep(5)
            else:
                print("Start")
                time.sleep(5)
                lists = out_queue.get()
                pre_input = None
                pre_output = None
                pre_reward = None

                total_reward = 0

                for depth, train_set in enumerate(lists):

                    if pre_input:
                        now_input = train_set.input
                        now_output = train_set.output
                        now_reward = train_set.reward

                        reward = np.subtract(now_reward, pre_reward)
                        reward = np.divide(reward, self.dev)
                        reward = np.multiply(reward, ReinforcementLearning.gamma**depth)
                        reward = np.sum(reward)
                        total_reward += reward

                    pre_input = train_set.input
                    pre_output = train_set.output
                    pre_reward = train_set.reward

                if total_reward > 0:
                    self.model.data = lists
                    self.model.optimize(len(lists))



        """
        env = Astra()
        training_data = []
        scores = []
        accepted_scores = []
        for _ in range(ReinforcementLearning.initial_games):
            score = 0
            game_memory = []
            prev_observation = []
            for _ in range(ReinforcementLearning.goal_steps):
                oct_move = env.get_oct_move_action()
                second_move = None
                if oct_move%env.n_move == env.shuffle:
                    second_move = env.move_sample(oct_move)

                observation, reward, done, info = env.step(oct_move, second_move)

                if len(prev_observation) > 0:
                    game_memory.append([oct_move, second_move])

                prev_observation = observation
                score += reward
                if done:
                    break

            if score >= ReinforcementLearning.score_requirement:
                accepted_scores.append(score)
                for data in game_memory:
                    training_data.append(data)
            env.reset()
            scores.append(score)
        training_data_save = np.array(training_data)
        np.save('saved.npy', training_data_save)
"""

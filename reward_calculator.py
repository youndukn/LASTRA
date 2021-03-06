#system modules
from queue import Queue
from threading import Thread
import copy
import os
import numpy as np
import pickle

from astra import Astra
from error import InputError
from data.astra_train_set import AstraTrainSet

class RewardCalculator:
    def __init__(self, thread_numb, astra):
        self.__astra_input_reader = astra.astra_input_reader
        self.numb = 6
        self.max = (17576, 1174.5, 1.49, 1.15, 1.55, 1.71)
        self.lists = []
        self.dev = [0, 0, 0, 0, 0, 0]
        self.dev_p = (64.39, 22.8, 0.08, 0.004, 0.10, 0.088)
        self.cal_numb = 0
        self.max_numb = 100
        self.thread_numb = thread_numb

    def calculate_rate(self):

        astra = Astra(self.__astra_input_reader)

        astra.reset()

        # Set up some global variables
        num_fetch_threads = self.thread_numb
        enclosure_queue = Queue()

        # Create
        for i in range(num_fetch_threads):

            # Create queues according to the thread number
            oct_move = astra.get_oct_move_action()
            second_move = astra.get_oct_shuffle_space(oct_move) if oct_move%Astra.n_move == astra.shuffle else None
            enclosure_queue.put([oct_move, second_move])

            # Create astra according to the thread number
            new_astra = copy.deepcopy(astra)

            directory = ".{}{}".format(os.path.sep, i)
            if not os.path.exists(directory):
                os.makedirs(directory)

            new_astra.working_directory = directory

            # Create threads according to the thread number
            worker = Thread(target=self.update_dev, args=(new_astra, enclosure_queue, ))
            worker.setDaemon(True)
            worker.start()

        # Now wait for the queue to be empty, indicating that we have
        # processed all of the downloads.)
        print('*** Main thread waiting')
        enclosure_queue.join()
        print('*** Done')

        with open('random_data.pkl', 'wb') as output_file:
            pickle.dump(self.lists, output_file, pickle.HIGHEST_PROTOCOL)

        for value in self.dev:
            print(value)

        return self.dev

    def update_dev(self, astra, queue):
        while True:
            if self.cal_numb < self.max_numb:
                # Create another queue
                oct_move = astra.get_oct_move_action()
                second_move = astra.get_oct_shuffle_space(
                    oct_move) if oct_move % Astra.n_move == astra.shuffle else None
                queue.put([oct_move, second_move])

            points = queue.get()

            # Run astra to get lists
            core, lists, changed, info = astra.change(points[0], points[1])

            if changed and info:
                astra.change_data.append(AstraTrainSet(core, points, lists, False, None))
                if len(astra.change_data) > 20:
                    self.lists.append(astra.change_data)
                    astra.reset()
                self.cal_numb += 1
                #print(astra.working_directory)
                """
                for x in core.assemblies:
                    a_string = ""
                    for y in x:
                        a_string += y.get_batch()+" "
                    print(a_string)
                """
                print(lists)

            if changed and not info:
                if not lists:
                    lists = [0, 0, 0, 0, 0, 0]
                astra.change_data.append(AstraTrainSet(core, points, lists, False, None))
                self.lists.append(astra.change_data)
                astra.reset()

            queue.task_done()

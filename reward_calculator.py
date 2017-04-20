#system modules
from queue import Queue
from threading import Thread
import time
import numpy as np
import copy
import os
from shutil import copyfile

from astra import Astra
from error import InputError


class RewardCalculator:
    def __init__(self, astra_input_reader):
        self.__astra_input_reader = astra_input_reader
        self.numb = 6
        self.max = (100000, 3000, 3, 3, 3, 3)
        self.lists = {}
        self.dev = [0, 0, 0, 0, 0, 0]
        self.dev_p = (64.39, 22.8, 0.08, 0.004, 0.10, 0.088)
        self.cal_numb = 0
        self.max_numb =100
        self.thread_numb = 2

    def calculate_rate(self):

        """
        lists = 1
        changed = False
        info = False

        while not (changed and info):
            oct_move = astra.get_oct_move_action()
            second_move = None
            if oct_move % astra.n_move == astra.shuffle:
                second_move = astra.get_oct_shuffle_space(oct_move)

            core, lists, changed, info = astra.change(oct_move, second_move)
            if changed and not info:
                astra.reset()

        pre = self.get_parameters(lists)
        """

        astra = Astra(self.__astra_input_reader)

        astra.reset()
        core, lists, changed, info = astra.run_process_astra()

        if not info:
            raise InputError("There was error in the input")

        # Set up some global variables
        num_fetch_threads = self.thread_numb
        enclosure_queue = Queue()

        # Create
        for i in range(num_fetch_threads):

            # Create queues according to the thread number
            oct_move = astra.get_oct_move_action()
            second_move = astra.get_oct_shuffle_space(oct_move) if oct_move%astra.n_move == astra.shuffle else None
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

        for key in self.lists:
            for values in self.lists[key]:
                print(values)

        for value in self.dev:
            print(value)
        return self.dev

    def update_dev(self, astra, queue):
        while True:
            if self.cal_numb > self.max_numb:
                # Task is done
                queue.task_done()
            else:
                # Create another queue
                oct_move = astra.get_oct_move_action()
                second_move = astra.get_oct_shuffle_space(
                    oct_move) if oct_move % astra.n_move == astra.shuffle else None
                queue.put([oct_move, second_move])

            points = queue.get()

            # Run astra to get lists
            core, lists, changed, info = astra.change(points[0], points[1])

            if changed and info:
                now = self.get_parameters(lists)
                self.lists.setdefault(Thread.ident, []).append(now)
                self.cal_numb += 1
                print(now)

            if changed and not info:
                self.lists.setdefault(Thread.ident, []).append([0, 0, 0, 0, 0, 0])
                astra.reset()

            queue.task_done()

    @staticmethod
    def get_parameters(lists):

        temp_list = []

        for a_list in lists:
            for values in a_list:
                if len(values) > 0:
                    temp_list.append(values[0:14])

        a = np.array(temp_list, dtype=np.float64)

        return (a.max(axis=0)[1],
                a.max(axis=0)[3],
                a.max(axis=0)[7],
                a.max(axis=0)[8],
                a.max(axis=0)[9],
                a.max(axis=0)[10])
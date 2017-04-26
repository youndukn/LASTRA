#system modules
from queue import Queue
from threading import Thread
import copy
import os
import numpy as np

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
        self.thread_numb = 12

    def calculate_rate(self):

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
            self.lists.setdefault(new_astra.working_directory, [[0, 0, 0, 0, 0, 0]])
            worker = Thread(target=self.update_dev, args=(new_astra, enclosure_queue, ))
            worker.setDaemon(True)
            worker.start()

        # Now wait for the queue to be empty, indicating that we have
        # processed all of the downloads.)
        print('*** Main thread waiting')
        enclosure_queue.join()
        print('*** Done')

        core_matrix = []
        output_array = []

        for key in self.lists:
            for core_outs in self.lists[key]:
                core = core_outs[0]
                out = core_outs[1]
                core_array =[]
                for i in range(6):
                    core_array.append(core.get_value_matrix(i))
                core_matrix.append(core_array)
                output_array.append(out)

        output_np = np.array(output_array)
        core_np = np.array(output_array)

        print(output_np)
        print(core_np)
        np.save('output.npy', output_np)
        np.save('core.npy', core_np)

        for value in self.dev:
            print(value)
        return self.dev

    def update_dev(self, astra, queue):
        while True:
            if self.cal_numb < self.max_numb:
                # Create another queue
                oct_move = astra.get_oct_move_action()
                second_move = astra.get_oct_shuffle_space(
                    oct_move) if oct_move % astra.n_move == astra.shuffle else None
                queue.put([oct_move, second_move])

            points = queue.get()

            # Run astra to get lists
            core, lists, changed, info = astra.change(points[0], points[1])

            if changed and info:
                self.lists[astra.working_directory].append([core, lists])
                astra.change_data.append(lists)
                self.cal_numb += 1
                print(astra.working_directory)
                for x in core.assemblies:
                    a_string = ""
                    for y in x:
                        a_string += y.get_batch()+" "
                    print(a_string)
                print(lists)

            if changed and not info:
                self.lists[astra.working_directory].append([core, [0, 0, 0, 0, 0, 0]])
                astra.reset()

            queue.task_done()


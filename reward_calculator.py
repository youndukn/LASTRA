from astra import Astra
import numpy as np

class RewardCalculator:
    def __init__(self, astra_input_reader):
        self.__astra_input_reader = astra_input_reader
        self.numb = 6
        self.max = (100000, 3000, 3, 3, 3, 3)
        self.dev = (0, 0, 0, 0, 0, 0)
        self.dev_p = (64.39, 22.8, 0.08, 0.004, 0.10, 0.088)


    def calculate_rate(self):
        astra = Astra(self.__astra_input_reader)

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

        astra.reset()

        dev = [i for i in range(self.numb)]

        number = 0

        while number < 100:
            oct_move = astra.get_oct_move_action()
            second_move = None
            if oct_move%astra.n_move == astra.shuffle:
                second_move = astra.get_oct_shuffle_space(oct_move)

            core, lists, changed, info = astra.change(oct_move, second_move)
            if changed and info:
                number += 1
                now = self.get_parameters(lists)

                for i in range(len(pre)):
                    dev[i] += np.abs(now[i]-pre[i])

                print()
                print([i for i in now])
                print([i for i in dev])
                pre = self.get_parameters(lists)

            if changed and not info:
                astra.reset()
        for i, value in enumerate(dev):
            self.dev[i] = value/number

        for value in self.dev:
            print(value)

    @staticmethod
    def get_parameters(lists):

        temp_list = []

        for a_list in lists:
            for values in a_list:
                if len(values) > 0:
                    temp_list.append(values[0:14])

        a = np.array(temp_list, dtype=np.float64)

        return (a.max(axis=0)[1], a.max(axis=0)[3], a.max(axis=0)[7], a.max(axis=0)[8], a.max(axis=0)[9], a.max(axis=0)[10])
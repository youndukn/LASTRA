from astra import Astra

class RewardCalculator:
    def __init__(self, astra_input_reader):
        self.__astra_input_reader = astra_input_reader
        self.numb = 6
        self.max = [100000, 3000, 3, 3, 3, 3]
        self.dev = [0 for i in range(6)]
        self.dev_p = [64.39, 22.8, 0.08, 0.004, 0.10, 0.088]


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

        pre_burnup, pre_cbc, pre_fr, pre_fz, pre_fxy, pre_fq = self.get_parameters(lists)

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
                burnup, cbc, fr, fz, fxy, fq = self.get_parameters(lists)

                dev[0] += abs(burnup - pre_burnup)
                dev[1] += abs(cbc - pre_cbc)
                dev[2] += abs(fr - pre_fr)
                dev[3] += abs(fz - pre_fz)
                dev[4] += abs(fxy - pre_fxy)
                dev[5] += abs(fq - pre_fq)

                print()
                print(burnup, cbc, fr, fxy, fq)
                print(dev[0], dev[1], dev[2], dev[3], dev[4])
                pre_burnup, pre_cbc, pre_fr, pre_fz, pre_fxy, pre_fq = self.get_parameters(lists)

            if changed and not info:
                astra.reset()
        for i, value in enumerate(dev):
            self.dev[i] = value/number

        for value in self.dev:
            print(value)

    @staticmethod
    def get_parameters(lists):
        burnup = 0
        cbc = 0
        fr = 0
        fz = 0
        fxy = 0
        fq = 0

        for a_list in lists:
            for values in a_list:
                if len(values) > 0:
                    burnup = float(values[1]) if burnup < float(values[1]) else burnup
                    cbc = float(values[3]) if cbc < float(values[3]) else cbc
                    fr = float(values[7]) if fr < float(values[7]) else fr
                    fz = float(values[8]) if fz < float(values[8]) else fz
                    fxy = float(values[9]) if fxy < float(values[9]) else fxy
                    fq = float(values[10]) if fq < float(values[10]) else fq

        return burnup, cbc, fr, fz, fxy, fq
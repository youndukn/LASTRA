from astra import Astra

class RewardCalculator:
    def __init__(self, astra_input_reader):
        self.__astra_input_reader = astra_input_reader
        self.__burnup_max = 100000
        self.__cbc_max = 3000
        self.__fr_max = 3
        self.__fz_max = 3
        self.__fxy_max = 3
        self.__fq_max = 3

        self.__burnup_d = 0
        self.__cbc_d = 0
        self.__fr_d = 0
        self.__fz_d = 0
        self.__fxy_d = 0
        self.__fq_d = 0

        self.__burnup_d_pre= 64.39
        self.__cbc_d_pre = 22.8
        self.__fr_d_pre = 0.08
        self.__fz_d_pre = 0.004
        self.__fxy_d_pre = 0.10
        self.__fq_d_pre = 0.088

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

        burnup_d = 0
        cbc_d = 0
        fr_d = 0
        fz_d = 0
        fxy_d = 0
        fq_d = 0
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

                burnup_d += abs(burnup - pre_burnup)
                cbc_d += abs(cbc - pre_cbc)
                fr_d += abs(fr - pre_fr)
                fz_d += abs(fz - pre_fz)
                fxy_d += abs(fxy - pre_fxy)
                fq_d += abs(fq - pre_fq)
                print()
                print(burnup, cbc, fr, fxy, fq)
                print(burnup_d, cbc_d, fr_d, fxy_d, fq_d)
                pre_burnup, pre_cbc, pre_fr, pre_fz, pre_fxy, pre_fq = self.get_parameters(lists)

            if changed and not info:
                astra.reset()

        self.__burnup_d = burnup_d/number
        self.__cbc_d = cbc_d / number
        self.__fr_d = fr_d / number
        self.__fz_d = fz_d / number
        self.__fxy_d = fxy_d / number
        self.__fq_d = fq_d / number
        print(self.__burnup_d, self.__cbc_d, self.__fr_d, self.__fz_d, self.__fxy_d, self.__fq_d)
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
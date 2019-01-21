#System Module
from subprocess import Popen, PIPE
from random import randint
import subprocess
import copy
import os
from data.astra_train_set import AstraTrainSet
from data.cross_power_set import CrossPowerSetN
from astra_io.astra_input_reader import AstraInputReader
from astra_io.astra_output_reader import AstraOutputReader
import random
import numpy as np
from error import InputError
import pickle
import time
from datetime import datetime

from data.core import Core
from data.assembly import FreshAssembly, ShuffleAssembly

ab_pre = [
    [0, 9, 9],
    [7, 9, 16],
    [55, 14, 14],
    [17, 10, 16],
    [27, 11, 16],
    [36, 12, 15],
    [46, 13, 15],
    [0, 16, 9],
    [0, 16, 10],
    [0, 16, 11],
    [0, 15, 12],
    [0, 15, 13]
]

ab_ax = [
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
]
ab_di = [
    [11, 10, 10],
    [22, 11, 11],
    [33, 12, 12],
    [44, 13, 13],
    [55, 14, 14],
]

ab_oc = [
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [45, 13, 14],
    [46, 13, 15],
]

fr_ax_ac = [True, True, True, False, True, True, False]
fr_di_ac = [True, True, True, True, False]
fr_oc_ac = [True, True, True, True, True, False,
            True, True, True, True, False,
            True, True, False,
            True, False]
"""
fr_ax_bc = [[4, 6, 7], [4, 6, 7], [4, 6, 7], [None,], [4, 5, 6, 7], [4, 5, 6, 7], [None,]]
fr_di_bc = [[4, 6, 7], [4, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [None,]]
fr_oc_bc = [[4, 6, 7], [4, 6, 7], [4, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [None,],
            [4, 6, 7], [4, 6, 7],  [4, 5, 6, 7], [4, 5, 6, 7], [None,],
            [4, 6, 7], [4, 5, 6, 7], [None,],
            [4, 5, 6, 7], [None, ]]
"""
fr_ax_bc = [[1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [None,], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [None,]]
fr_di_bc = [[1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [None,]]
fr_oc_bc = [[1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [None,],
            [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [None,],
            [1, 4, 5, 6, 7], [1, 4, 5, 6, 7], [None,],
            [1, 4, 5, 6, 7], [None, ]]

on_ax_ac = [True, True, True, True, True, True, False]
on_di_ac = [True, True, True, True, False]
on_oc_ac = [True, True, True, True, True, False,
            True, True, True, True, False,
            True, True, False,
            True, False]

fr_pl_ch = ((3, 6), (5, 5))

col_index = ["H","J","K","L","M","N","P","R"]
row_index = ["8","9","10","11","12","13","14","15"]

rot_sym = [0, 3, 2, 1]

class AstraRD():

    uniform_dis = 0
    normal_dis = 1
    greedy_dis = 2
    upper_normal_dis = 3
    lower_normal_dis = 4

    distribution = upper_normal_dis

    n_move = 10

    n_shuffle = 0
    n_rotate = 3
    n_bp = 2

    shuffle = 0
    rotate1 = 1
    rotate2 = 2
    rotate3 = 3
    bp_in = 4
    bp_de = 5
    bp_co_in = 6
    bp_co_in = 7
    co_in = 8
    co_de = 9

    def __init__(self, input_name,
                 reward_list_target=(17576, 1174.5, 1.49, 1.15, 1.55, 1.71),
                 main_directory=".{}".format(os.path.sep),
                 working_directory=".{}".format(os.path.sep),
                 thread_id = 0
                 ):
        t = int(time.time() * 1000.0)
        np.random.seed(((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24))
        random.seed(((t & 0xff000000) >> 24) +
             ((t & 0x00ff0000) >>  8) +
             ((t & 0x0000ff00) <<  8) +
             ((t & 0x000000ff) << 24))
        self.action_space = (3300)
        self.observation_space = (10, 10, 28)
        #Input Reader
        self.input_name = input_name

        self.file = open(working_directory+os.path.sep+'cross_power', 'wb')

        self.__astra_input_reader = AstraInputReader(self.input_name, main_directory)

        for astra_block in self.__astra_input_reader.blocks:
            self.__astra_input_reader.parse_block_content(astra_block.block_name)

        #Core information
        max_row = self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core.max_row
        max_col = self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core.max_col

        self.__max_position = max_row * (max_col - 1) / 2 + max_row

        self.__position = self.set_oct_position(max_col)

        #Working Directory
        self.__working_directory = working_directory

        self.found_lps = {}
        self.changed_lps = {}

        #Input
        #output_core, reward_list, _, successful = \
        #    self.run_process_astra_3D(self.__astra_input_reader.blocks[AstraInputReader.shuff_block])

        output_core, reward_list, _, successful, succ_error, cross_set = \
            self.run_process_astra(self.__astra_input_reader.blocks[AstraInputReader.shuff_block])

        if not successful:
            raise InputError("Initial Run", "Initial LP has an error")

        if thread_id == -1:
            self.file = open('/media/youndukn/lastra/running_data/hello', 'wb')
            dump_list = []
            for i, value in enumerate(cross_set):
                if i == 0:
                    a_list = [value.summary_tensor,
                              value.input_tensor_full,
                              value.output_tensor,
                              value.flux_tensor,
                              value.p2dn_tensor_full,
                              value.iinput_tensor_full,
                              value.iinputn_tensor_full,
                              value.binput_tensor_full,
                              value.batch_tensor_full,
                              value.peak_tensor_full,
                              value.fr_tensor_full,
                              value.p3dn_tensor_full,
                              value.p3d_tensor_full,
                              value.pp3d_tensor_full,
                              value.iinput3n_tensor_full]
                else:
                    a_list = [value.summary_tensor,
                              value.input_tensor_full,
                              value.output_tensor,
                              value.flux_tensor,
                              value.p2dn_tensor_full,
                              [],
                              [],
                              value.binput_tensor_full,
                              value.batch_tensor_full,
                              value.peak_tensor_full,
                              value.fr_tensor_full,
                              value.p3dn_tensor_full,
                              value.p3d_tensor_full,
                              value.pp3d_tensor_full,
                              []]
                dump_list.append(a_list)
            pickle.dump(dump_list, self.file, protocol=pickle.HIGHEST_PROTOCOL)
            self.file.close()

        self.__input_core_best = [copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core)]
        self.__train_set_best = [AstraTrainSet(output_core, None, reward_list, False, None)]
        self.__crosspower_set_best = [cross_set]
        self.__start_index = 0

        self.__input_core_history = [copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core)]
        self.__train_set_history = [AstraTrainSet(output_core, None, reward_list, False, None)]
        self.__crosspower_set_history = [cross_set]

        self.__reward_list_target = reward_list_target #target_rewards

        self.pre_shuff = []
        with open("input_shuff.inp") as file:
            self.once_batch_id, self.twice_batch_id = file.readline().split()
            for line in file:
                values = line.split()
                self.pre_shuff.append(values[1:5])

        if len(self.pre_shuff) < 18:
            raise InputError("Shuff Not Found", "input_shuff is needed")

    def reset(self):
        best_count = len(self.__input_core_best)
        if AstraRD.distribution == AstraRD.uniform_dis:
            self.__start_index = random.randint(0, best_count - 1)
        elif AstraRD.distribution == AstraRD.normal_dis:
            index = np.clip(np.random.normal(0, best_count / 4), -best_count/2, best_count/2)
            if index<0:
                index = best_count + index
            self.__start_index = int(index)
        elif AstraRD.distribution == AstraRD.upper_normal_dis:
            index = np.random.normal(0, best_count / 4)
            index = best_count + -1*abs(index)
            self.__start_index = int(np.clip(index, 0, best_count-1))
        elif AstraRD.distribution == AstraRD.lower_normal_dis:
            index = np.random.normal(0, best_count / 4)
            index = abs(index)
            self.__start_index = int(np.clip(index, 0, best_count-1))
        else:
            self.__start_index = best_count - 1
        self.__input_core_history = [copy.deepcopy(self.__input_core_best[self.__start_index])]
        self.__train_set_history = [copy.deepcopy(self.__train_set_best[self.__start_index])]
        self.__crosspower_set_history = [copy.deepcopy(self.__crosspower_set_best[self.__start_index])]

    def get_oct_shuffle_space(self, space_action):
        return randint(0, self.__max_position-1)

    def set_working_directory(self, working):
        self.__working_directory = working

    def get_initial_state(self):
        return self.__train_set_history[0].state

    def step_shuffle(self, shuffle_position, reward_index):
        m_position1 = int(shuffle_position / self.__max_position) * AstraRD.n_move
        m_position2 = shuffle_position % self.__max_position
        output_core, reward_list, changed, info, succ_error, cross_set = self.change(int(m_position1), int(m_position2))
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(cross_set)
        return output_core, reward, changed, info,succ_error, satisfied, best, cross_set

    def step_shuffle_full(self, shuffle_position, reward_index):
        m_position1 = int(shuffle_position / self.__max_position) * AstraRD.n_move
        m_position2 = shuffle_position % self.__max_position
        output_core, reward_list, changed, info,succ_error, cross_set = self.change(int(m_position1), int(m_position2))
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(cross_set)
        return output_core, reward, reward_list, changed, info, satisfied, best, succ_error,cross_set

    def step_rotate(self, rotate_position, reward_index):
        m_position1 = int(rotate_position / AstraRD.n_rotate) + AstraRD.rotate1 + rotate_position % AstraRD.n_rotate
        output_core, reward_list, changed, info, succ_error,cross_set = self.change(int(m_position1), None)
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(cross_set)
        return output_core, reward, changed, info,succ_error, satisfied, best, cross_set

    def step_bp(self, bp_position, reward_index):
        m_position1 = int(bp_position / AstraRD.n_bp) + AstraRD.bp_in + bp_position % AstraRD.n_bp
        output_core, reward_list, changed, info,succ_error, cross_set = self.change(int(m_position1), None)
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(cross_set)
        return output_core, reward, changed, info, succ_error, satisfied, best, cross_set

    def change(self,  m_position1, position2):
        position1 = int(m_position1 / AstraRD.n_move)

        next_core = copy.deepcopy(self.__input_core_history[-1])
        shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]
        shuffle_block.core = next_core

        changed = False
        if m_position1 % AstraRD.n_move == AstraRD.shuffle and position2:
            changed = shuffle_block.core.shuffle(self.__position[position1], self.__position[position2])
        elif m_position1 % AstraRD.n_move == AstraRD.rotate1:
            changed = shuffle_block.core.rotate(self.__position[position1])
        elif m_position1 % AstraRD.n_move == AstraRD.rotate2:
            shuffle_block.core.rotate(self.__position[position1])
            changed = shuffle_block.core.rotate(self.__position[position1])
        elif m_position1 % AstraRD.n_move == AstraRD.rotate3:
            shuffle_block.core.rotate(self.__position[position1])
            shuffle_block.core.rotate(self.__position[position1])
            changed = shuffle_block.core.rotate(self.__position[position1])
        elif m_position1 % AstraRD.n_move == AstraRD.bp_in:
            changed = shuffle_block.core.poison(self.__position[position1], True)
        elif m_position1 % AstraRD.n_move == AstraRD.bp_de:
            changed = shuffle_block.core.poison(self.__position[position1], False)

        changed_block = shuffle_block.print_block()
        include_lps = changed_block in self.found_lps

        if not include_lps:
            if not changed:
                self.__input_core_history.append(next_core)
                copy_train_set = copy.deepcopy(self.__train_set_history[-1])
                self.__train_set_history.append(copy.deepcopy(self.__train_set_history[-1]))
                self.__crosspower_set_history.append(copy.deepcopy(self.__crosspower_set_history[-1]))
                return copy_train_set.input, 0.0, changed, True, True, copy.deepcopy(self.__crosspower_set_history[-1])

            self.__astra_input_reader.blocks[AstraInputReader.shuff_block] = shuffle_block

            self.__input_core_history.append(next_core)
            # output_core, reward_list, changed, successful = self.run_process_astra_3D(shuffle_block)
            output_core, reward_list, changed, successful, succ_error, cross_set = self.run_process_astra(shuffle_block)

            self.__train_set_history.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_history.append(cross_set)
            #self.found_lps[changed_block] = (output_core, reward_list, changed, successful, succ_error, cross_set)
        else:
            #print("Already",len(keys))

            output_core, reward_list, _, successful, succ_error, cross_set = self.found_lps[changed_block]
            if not changed:
                self.__input_core_history.append(next_core)
                self.__train_set_history.append(copy.deepcopy(self.__train_set_history[-1]))
                self.__crosspower_set_history.append(copy.deepcopy(self.__crosspower_set_history[-1]))
            else:
                self.__astra_input_reader.blocks[AstraInputReader.shuff_block] = shuffle_block
                self.__input_core_history.append(next_core)
                self.__train_set_history.append(AstraTrainSet(output_core, None, reward_list, False, None))
                self.__crosspower_set_history.append(cross_set)

        return output_core, reward_list, changed, successful, succ_error, cross_set

    def get_cross_set(self):
        return self.__crosspower_set_history[-1]

    def step_back(self):
        self.__crosspower_set_history.remove(self.__crosspower_set_history[-1])
        self.__input_core_history.remove(self.__input_core_history[-1])
        self.__train_set_history.remove(self.__train_set_history[-1])


    def get_last_core(self):
        return self.__input_core_history[-1]

    def step(self):

        batch_block = self.__astra_input_reader.blocks[AstraInputReader.batch_block]

        # batch_id is only one string
        batch_id = batch_block.batches[0][0]

        core = Core("LPD_SHF")
        isNotPass = True

        fr_ax_count = 0
        fr_di_count = 0
        fr_oc_count = 0
        while(isNotPass):

            ci = np.random.randint(0, len(fr_pl_ch) )
            ax, mi = fr_pl_ch[ci]

            fr_ax_co = 0

            # Fresh Fuel Axial Placements
            axial = [True]*ax+[False]*(sum(fr_ax_ac)+sum(fr_di_ac)-ax)
            np.random.shuffle(axial)
            for i, isPut in enumerate(fr_ax_ac):
                if isPut:
                    isFresh, axial = axial[-1], axial[:-1]
                    if isFresh:
                        batch_index = np.random.choice(fr_ax_bc[i])
                        batch_name = batch_id+str(batch_index)
                        fresh = FreshAssembly()
                        fresh.set_batch(batch_name)
                        fresh.set_number(0)
                        core.set_shuffle_at(fresh, ab_ax[i][1]-9, ab_ax[i][2]-9)
                        core.set_shuffle_at(fresh, ab_ax[i][2]-9, ab_ax[i][1]-9)
                        fr_ax_co+=1

            fr_di_co = 0
            for i, isPut in enumerate(fr_di_ac):
                if isPut:
                    isFresh, axial = axial[-1], axial[:-1]
                    if isFresh:
                        batch_index = np.random.choice(fr_di_bc[i])
                        batch_name = batch_id + str(batch_index)
                        fresh = FreshAssembly()
                        fresh.set_batch(batch_name)
                        fresh.set_number(0)
                        core.set_shuffle_at(fresh, ab_di[i][1]-9, ab_di[i][2]-9)
                        fr_di_co += 1

            # Fresh Fuel Octant CWhitesore Placement
            octant = [True] * mi + [False] * (sum(fr_oc_ac) - mi)
            np.random.shuffle(octant)
            for i, isPut in enumerate(fr_oc_ac):
                if isPut:
                    isFresh, octant = octant[-1], octant[:-1]
                    if isFresh:
                        batch_index = np.random.choice(fr_oc_bc[i])
                        batch_name = batch_id + str(batch_index)
                        fresh = FreshAssembly()
                        fresh.set_batch(batch_name)
                        fresh.set_number(0)
                        core.set_shuffle_at(fresh, ab_oc[i][1]-9, ab_oc[i][2]-9)
                        core.set_shuffle_at(fresh, ab_oc[i][2]-9, ab_oc[i][1]-9)

            count2 = core.count_in_range(0, 0, 1, 2, 2)
            count3 = core.count_in_range(0, 0, 1, 3, 3)
            count4 = core.count_in_range(0, 0, 1, 4, 4)
            count5 = core.count_in_range(0, 0, 1, 5, 5)
            count6 = core.count_in_range(0, 0, 1, 6, 6)

            two_by_two = False
            for row in range(core.max_row-1):
                for col in range(core.max_col-1):
                    if 4 == core.count_in_range(0, row, col, row+2,col+2):
                        two_by_two = True
                        break
                if two_by_two == True:
                    break

            one_by_four = False
            for row in range(core.max_row):
                for col in range(core.max_col-3):
                    if 4 == core.count_in_range(0, row, col, row+1, col + 4):
                        one_by_four = True
                        break
                if one_by_four == True:
                    break

            if count2 <= 1 and \
                            count3 <= 5 and \
                            count4 <= 8 and \
                            count5 <= 10 and \
                            count6 <= 13 and \
                            two_by_two == False and \
                            one_by_four == False:
                fr_ax_count = fr_ax_co
                fr_di_count = fr_di_co
                fr_oc_count = mi
                isNotPass=False
            else:
                core = Core("LPD_SHF")

        # Find Once Burnt

        on_axial = []
        on_diagonal = []
        on_octant = []
        axial_collect = []
        diagonal_collect = []
        octant_collect = []

        for i  in range(len(self.pre_shuff)):
            values = self.pre_shuff[i]
            col = col_index.index(values[1])
            row = row_index.index(values[2])
            batch_id = values[3][2:]
            if self.once_batch_id in batch_id or (self.twice_batch_id+'0') in batch_id:
                if col != 0:
                    if row == 0:
                        on_axial.append([row, col])
                    elif row == col:
                        on_diagonal.append([row, col])
                    elif [col, row] not in on_octant:
                        on_octant.append([row, col])

        np.random.shuffle(on_octant)

        on_ax = sum(on_ax_ac) - fr_ax_count
        if len(on_axial) > on_ax:
            axial_collect = axial_collect + on_axial[on_ax:]
            on_axial = on_axial[:on_ax]

        on_di = sum(on_di_ac) - fr_di_count
        if len(on_diagonal) > on_di:
            diagonal_collect = diagonal_collect + on_diagonal[on_di:]
            on_diagonal = on_diagonal[:on_di]

        on_oc = sum(on_oc_ac) - fr_oc_count
        if len(on_octant) > on_oc:
            octant_collect = octant_collect + on_octant[on_oc:]
            on_octant = on_octant[:on_oc]

        on_axial = on_axial + diagonal_collect
        diagonal_collect = on_axial[on_ax:]
        on_axial = on_axial[:on_ax]

        on_diagonal = on_diagonal + axial_collect
        axial_collect = on_diagonal[on_di:]
        on_diagonal = on_diagonal[:on_di]

        if len(diagonal_collect)>0  or len(axial_collect) >0:
            raise InputError("Axial dia Left", len(diagonal_collect)+", "+len(axial_collect))

        octant_sym_collect = []
        for octant in octant_collect:
            octant_sym_collect.append(octant)
            octant_sym_collect.append([octant[1],octant[0]])

        np.random.shuffle(octant_sym_collect)

        on_axial = on_axial + octant_sym_collect
        octant_sym_collect = on_axial[on_ax:]
        on_axial = on_axial[:on_ax]

        on_diagonal = on_diagonal + octant_sym_collect
        octant_sym_collect = on_diagonal[on_di:]
        on_diagonal = on_diagonal[:on_di]
        """
        if len(octant_sym_collect) > 0:
            raise InputError("Octant Left", len(octant_sym_collect))
        """
        np.random.shuffle(on_axial)
        for i, isPut in enumerate(on_ax_ac):
            row = ab_ax[i][1]-9
            col = ab_ax[i][2]-9

            if isPut and FreshAssembly != type(core.assemblies[row][col]):
                shuffle_pos, on_axial = on_axial[-1], on_axial[:-1]
                shuff = ShuffleAssembly()
                shuff.set_rotation(1)
                shuff.set_cycle(1)
                shuff.set_row(row_index[shuffle_pos[0]])
                shuff.set_col(col_index[shuffle_pos[1]])

                shuff1 = ShuffleAssembly()
                shuff1.set_rotation(0)
                shuff1.set_cycle(1)
                shuff1.set_row(row_index[shuffle_pos[0]])
                shuff1.set_col(col_index[shuffle_pos[1]])

                core.set_shuffle_at(shuff, row, col)
                core.set_shuffle_at(shuff1, col, row)

        np.random.shuffle(on_diagonal)
        for i, isPut in enumerate(on_di_ac):
            row = ab_di[i][1]-9
            col = ab_di[i][2]-9

            if isPut and FreshAssembly != type(core.assemblies[row][col]):
                shuffle_pos, on_diagonal = on_diagonal[-1], on_diagonal[:-1]
                shuff = ShuffleAssembly()
                shuff.set_rotation(1)
                shuff.set_cycle(1)
                shuff.set_row(row_index[shuffle_pos[0]])
                shuff.set_col(col_index[shuffle_pos[1]])

                core.set_shuffle_at(shuff, row, col)

        np.random.shuffle(on_octant)
        for i, isPut in enumerate(on_oc_ac):
            row = ab_oc[i][1]-9
            col = ab_oc[i][2]-9

            if isPut and FreshAssembly != type(core.assemblies[row][col]):
                shuffle_pos, on_octant = on_octant[-1], on_octant[:-1]
                shuff = ShuffleAssembly()
                rot_value = np.random.randint(4)
                shuff.set_rotation(rot_value)
                shuff.set_cycle(1)
                shuff.set_row(row_index[shuffle_pos[0]])
                shuff.set_col(col_index[shuffle_pos[1]])

                shuff1 = ShuffleAssembly()
                shuff1.set_rotation(rot_sym[rot_value])
                shuff1.set_cycle(1)
                shuff1.set_row(row_index[shuffle_pos[1]])
                shuff1.set_col(col_index[shuffle_pos[0]])

                core.set_shuffle_at(shuff, row, col)
                core.set_shuffle_at(shuff1, col, row)


        shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]

        for value in ab_pre:
            core.set_shuffle_at(shuffle_block.core.assemblies[value[1]-9][value[2]-9], value[1]-9, value[2]-9)

        shuffle_block.core = core
        #print(shuffle_block.print_block())
        self.__astra_input_reader.blocks[AstraInputReader.shuff_block] = shuffle_block

        self.__input_core_history.append(core)
        # output_core, reward_list, changed, successful = self.run_process_astra_3D(shuffle_block)
        output_core, reward_list, changed, successful, succ_error, cross_set = self.run_process_astra(shuffle_block)

        self.__train_set_history.append(AstraTrainSet(output_core, None, reward_list, False, None))
        self.__crosspower_set_history.append(cross_set)

        return output_core, reward_list, changed, successful, succ_error, cross_set

    def run_process_astra(self, shuffle_block=None):

        if not shuffle_block:
            shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]
        output_string = self.run_astra(shuffle_block)

        if output_string:

            self.__reading_out = AstraOutputReader(output_string=output_string)

            output_core, reward_list, successful = self.__reading_out.process_astra()

            cross_power_density_list, successful, succ_error = self.__reading_out.process_astra_cross_power()

            depletions = []
            try:
                if cross_power_density_list:
                    for list in cross_power_density_list:
                        depletions.append(CrossPowerSetN(list[0],
                                                         list[1],
                                                         list[2],
                                                         list[3],
                                                         list[4],
                                                         list[5],
                                                         list[6],
                                                         list[7],
                                                         list[8],
                                                         list[9],
                                                         list[10],
                                                         list[11],
                                                         list[12],
                                                         list[13],
                                                         list[14],
                                                         list[15],
                                                         list[16],
                                                         list[17],
                                                         list[18],
                                                         list[19],
                                                         list[20],
                                                         list[21],
                                                         ))
            except TypeError as e:
                print(e)
                return output_core, reward_list, True, False, False, depletions,
            return output_core, reward_list, True, successful, succ_error, depletions,

        return None, None, True, False, False, []

    def run_astra(self, shuffle_block):

        replaced = self.__astra_input_reader.replace_block([shuffle_block])
        time.sleep(1)

        try:
            p = subprocess.Popen(['/home/youndukn/bin/astra.1.4.2_jylee'],
                                 stdout=subprocess.PIPE,
                                 stdin=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 shell=True,
                                 cwd=self.__working_directory)
            output, err = (p.communicate(input=replaced.encode('utf-8')))
            return output.decode('utf-8')
        except subprocess.CalledProcessError:
            print("Error in running astra")
            return None

    def get_oct_move_action(self):

        shuffle_prob = 6
        batch_prob = 3

        move = randint(0, 10)
        if shuffle_prob > move:
            next_move = 0
        elif batch_prob > move:
            next_move = randint(1, 3)
        else:
            next_move = randint(4, 5)

        position = randint(0, self.__max_position-1)
        position *= self.n_move
        position += next_move

        return position

    def __calculate_reward(self, reward_list, reward_index = [0, 1, 2, 3, 4, 5]):
        if reward_list == 0.0 or not reward_list:
            return 0.0, False, False

        reward_list_init = self.__train_set_history[0].reward
        reward_list_best= self.__train_set_best[-1].reward

        total_reward = 0
        satisfied = True
        s_c = 0

        for j in reward_index:
            if j == 0:
                if min(self.__reward_list_target[j], reward_list_best[j]) < reward_list[j]:
                    s_c += 1

                if self.__reward_list_target[j] > reward_list[j]:
                    satisfied = False
                total_reward += \
                    (max(self.__reward_list_target[j], reward_list[j]) - max(self.__reward_list_target[j], reward_list_init[j]) )\
                    / self.__reward_list_target[j]
            else:
                if max(self.__reward_list_target[j], reward_list_best[j]) > reward_list[j]:
                    s_c += 1

                if self.__reward_list_target[j] < reward_list[j]:
                    satisfied = False
                total_reward += \
                    (min(self.__reward_list_target[j], reward_list_init[j]) - min(self.__reward_list_target[j], reward_list[j]))\
                    / self.__reward_list_target[j]

        best = False
        if s_c >= len(reward_index):
            best = True
            with open("./Best.txt", "a") as myfile:
                for key in self.__reading_out.blocks[AstraOutputReader.input_block].dictionary:
                    myfile.write("Best Reward " + str(reward_list) + self.__working_directory)
                    myfile.write(self.__reading_out.blocks[AstraOutputReader.input_block].dictionary[key])

        if satisfied:
            with open("./Satisfied.txt", "a") as myfile:
                print("Satisfied")
                for key in self.__reading_out.blocks[AstraOutputReader.input_block].dictionary:
                    myfile.write(self.__reading_out.blocks[AstraOutputReader.input_block].dictionary[key])

        return total_reward/len(reward_index), best, satisfied


    @staticmethod
    def set_oct_position(max_col):
        position = []
        for row in range(max_col):
            for col in range(row, max_col):
                position.append([row, col])
        return position

if __name__ == "__main__":
    astra = AstraRD("01_boc.inp", thread_id=-1)

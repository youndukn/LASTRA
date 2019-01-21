#System Module
from subprocess import Popen, PIPE
from random import randint
import subprocess
import copy
import os
from data.astra_train_set import AstraTrainSet
from data.cross_power_set import CrossPowerSetI,CrossPowerSetN, CrossPowerSet3D
from astra_io.astra_input_reader import AstraInputReader
from astra_io.astra_output_reader import AstraOutputReader
import random
import numpy as np
from error import InputError
import pickle
import time

def my_outputs(cross_set):
    my_cl = cross_set[-1].summary_tensor[0]
    my_fxy = 0
    for burnup_point in cross_set:
        if my_fxy < burnup_point.summary_tensor[5]:
            my_fxy = burnup_point.summary_tensor[5]
    return my_cl, my_fxy

class Astra():

    uniform_dis = 0
    normal_dis = 1
    greedy_dis = 2
    upper_normal_dis = 3
    lower_normal_dis = 4

    distribution = lower_normal_dis

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
                 main_directory=".{}".format(os.path.sep),
                 working_directory=".{}".format(os.path.sep),
                 shuff_string = ""
                 ):

        self.input_name = input_name

        self.__astra_input_reader = AstraInputReader(self.input_name, main_directory)

        for astra_block in self.__astra_input_reader.blocks:
            self.__astra_input_reader.parse_block_content(astra_block.block_name)

        if len(shuff_string) > 0:
            self.__astra_input_reader.input_string = self.__astra_input_reader.replace_block(
                            [self.__astra_input_reader.blocks[AstraInputReader.shuff_block]],
                            [shuff_string])

            for astra_block in self.__astra_input_reader.blocks:
                self.__astra_input_reader.parse_block_content(astra_block.block_name)

        # Core information
        max_row = self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core.max_row
        max_col = self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core.max_col

        self.__max_position = max_row * (max_col - 1) / 2 + max_row

        self.__position = self.set_oct_position(max_col)

        # Working Directory
        self.__working_directory = working_directory

        cross_set, successful, succ_error = \
            self.run_process_astra(self.__astra_input_reader.blocks[AstraInputReader.shuff_block])

        if not succ_error:
            raise InputError("Initial Run", "Initial LP has an error")

        self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core. \
            set_cross_section(copy.deepcopy(cross_set[0].iinputn_tensor_full))


        self.original_core = copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core)
        self.original_cross_set = copy.deepcopy(cross_set)
        self.original_output = my_outputs(cross_set)

        self.pre_core = copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core)
        self.pre_cross_set = copy.deepcopy(cross_set)
        self.pre_output = my_outputs(cross_set)

        self.changed_core = copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core)
        self.changed_cross_set = copy.deepcopy(cross_set)
        self.changed_output = my_outputs(cross_set)

    def reset(self):
        self.pre_core = self.original_core
        self.pre_cross_set = self.original_cross_set
        self.pre_output = self.original_output

        self.changed_core = self.original_core
        self.changed_cross_set = self.original_cross_set
        self.changed_output = self.original_output

    def get_oct_shuffle_space(self, space_action):
        return randint(0, self.__max_position-1)

    def set_working_directory(self, working):
        self.__working_directory = working

    def step_shuffle(self, shuffle_position):
        m_position1 = int(shuffle_position / self.__max_position) * Astra.n_move
        m_position2 = shuffle_position % self.__max_position
        return self.change(int(m_position1), int(m_position2))

    def get_shuffle(self, shuffle_position):
        m_position1 = int(shuffle_position / self.__max_position) * Astra.n_move
        m_position2 = shuffle_position % self.__max_position
        return self.temp_change(int(m_position1), int(m_position2))

    def step_rotate(self, rotate_position):
        m_position1 = int(rotate_position / Astra.n_rotate) * Astra.n_move + Astra.rotate1 + rotate_position % Astra.n_rotate
        return self.change(int(m_position1), None)

    def step_bp(self, bp_position):
        m_position1 = int(bp_position / Astra.n_bp) * Astra.n_move + Astra.bp_in + bp_position % Astra.n_bp
        return self.change(int(m_position1), None)

    def change_lp(self, core, m_position1, position1, position2):

        changed = False
        if m_position1 % Astra.n_move == Astra.shuffle and position2:
            changed = core.shuffle(self.__position[position1], self.__position[position2])
            if changed:
                core.shuffle_cross(self.__position[position1], self.__position[position2])
        elif m_position1 % Astra.n_move == Astra.rotate1:
            changed = core.rotate(self.__position[position1])
            if changed:
                core.rotate_cross(self.__position[position1])
        elif m_position1 % Astra.n_move == Astra.rotate2:
            core.rotate(self.__position[position1])
            changed = core.rotate(self.__position[position1])
            if changed:
                core.rotate_cross(self.__position[position1])
                core.rotate_cross(self.__position[position1])
        elif m_position1 % Astra.n_move == Astra.rotate3:
            core.rotate(self.__position[position1])
            core.rotate(self.__position[position1])
            changed = core.rotate(self.__position[position1])
            if changed:
                core.rotate_cross(self.__position[position1])
                core.rotate_cross(self.__position[position1])
                core.rotate_cross(self.__position[position1])
        elif m_position1 % Astra.n_move == Astra.bp_in:
            changed = core.poison(self.__position[position1], True)
            if changed:
                core.poison_cross(self.__position[position1])
        elif m_position1 % Astra.n_move == Astra.bp_de:
            changed = core.poison(self.__position[position1], False)
            if changed:
                core.poison_cross(self.__position[position1])

        return changed

    def change(self,  m_position1, position2):
        position1 = int(m_position1 / Astra.n_move)

        next_core = copy.deepcopy(self.changed_core)
        shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]
        shuffle_block.core = next_core
        changed = self.change_lp(next_core, m_position1, position1, position2)
        if not changed:
            return next_core, self.changed_cross_set, changed, True, True,
        cross_set, successful, succ_error = self.run_process_astra(shuffle_block)
        if len(cross_set)>0:
            checked = self.check_cross_set(cross_set[0].iinputn_tensor_full, next_core.cross_section)
            if not checked:
                print(m_position1, position1, position2)

        return next_core, cross_set, changed, successful, succ_error

    def check_cross_set(self, f_cross_power, s_cross_power):
        checked = True
        for row, row_assembly in enumerate(f_cross_power):
            for col, col_assembly in enumerate(row_assembly):
                for type, type_value in enumerate(col_assembly):
                    if type_value > 0:
                        value1 = s_cross_power[row][col][type]
                        if abs(value1 - type_value)/value1> 0.01:
                            checked = False
                            print(type_value, row, col, type)
        return checked

    def temp_change(self,  m_position1, position2):
        position1 = int(m_position1 / Astra.n_move)
        next_core = copy.deepcopy(self.changed_core)
        changed = self.change_lp(next_core, m_position1, position1, position2)
        return next_core, changed

    def print_shuffle(self, core):
        shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]
        shuffle_block.core = core
        return shuffle_block.print_block()

    def step(self, m_position1, position2):
        output_core, reward_list, changed, info = self.change(m_position1, position2)
        return output_core, self.calculate_reward(reward_list), info, changed

    def run_process_astra(self, shuffle_block=None):

        if not shuffle_block:
            shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]

        output_string = self.run_astra(shuffle_block)

        if output_string:

            self.__reading_out = AstraOutputReader(output_string=output_string)

            cross_power_density_list, successful, succ_error = self.__reading_out.process_astra_cross_power()
            depletions = []
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
                                                    list[18]
                                                     ))

            return depletions, successful, succ_error

        return [], False, False,

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

    @staticmethod
    def set_oct_position(max_col):
        position = []
        for row in range(max_col):
            for col in range(row, max_col):
                position.append([row, col])
        return position

#System Module
from subprocess import Popen, PIPE
from random import randint
import subprocess
import copy
import os
from data.astra_train_set import AstraTrainSet
from data.cross_power_set import CrossPowerSet, CrossPowerSet3D
from astra_io.astra_input_reader import AstraInputReader
from astra_io.astra_output_reader import AstraOutputReader
import random
import numpy as np
from error import InputError
import pickle
import time

class Astra():

    uniform_dis = 0
    normal_dis = 1
    greedy_dis = 2
    upper_normal_dis = 3

    distribution = greedy_dis

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
                 ):
        self.action_space = (3300)
        self.observation_space = (10, 10, 28)
        #Input Reader
        self.cross_set = None

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

        #Input
        #output_core, reward_list, _, successful = \
        #    self.run_process_astra_3D(self.__astra_input_reader.blocks[AstraInputReader.shuff_block])

        output_core, reward_list, _, successful = \
            self.run_process_astra(self.__astra_input_reader.blocks[AstraInputReader.shuff_block])

        if not successful:
            raise InputError("Initial Run", "Initial LP has an error")

        self.__input_core_best = [copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core)]
        self.__train_set_best = [AstraTrainSet(output_core, None, reward_list, False, None)]
        self.__crosspower_set_best = [copy.deepcopy(self.cross_set)]
        self.__start_index = 0

        self.__input_core_history = [copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block].core)]
        self.__train_set_history = [AstraTrainSet(output_core, None, reward_list, False, None)]
        self.__crosspower_set_history = [copy.deepcopy(self.cross_set)]

        self.__reward_list_target = reward_list_target #target_rewards

    def reset(self):
        best_count = len(self.__input_core_best)
        if Astra.distribution == Astra.uniform_dis:
            self.__start_index = random.randint(0, best_count - 1)
        elif Astra.distribution == Astra.normal_dis:
            index = np.clip(np.random.normal(0, best_count / 4), -best_count/2, best_count/2)
            if index<0:
                index = best_count + index
            self.__start_index = int(index)
        elif Astra.distribution == Astra.upper_normal_dis:
            index = np.random.normal(0, best_count / 4)
            index = best_count + -1*abs(index)
            self.__start_index = int(np.clip(index, 0, best_count-1))
        else:
            self.__start_index = best_count - 1
        self.__input_core_history = [copy.deepcopy(self.__input_core_best[self.__start_index])]
        self.__train_set_history = [copy.deepcopy(self.__train_set_best[self.__start_index])]
        self.__crosspower_set_history = [copy.deepcopy(self.__crosspower_set_best[self.__start_index])]
        self.cross_set = copy.deepcopy(self.__crosspower_set_best[self.__start_index])

    def get_oct_shuffle_space(self, space_action):
        return randint(0, self.__max_position-1)

    def set_working_directory(self, working):
        self.__working_directory = working

    def get_initial_state(self):
        return self.__train_set_history[0].state

    def step_shuffle(self, shuffle_position, reward_index):
        m_position1 = int(shuffle_position / self.__max_position) * Astra.n_move
        m_position2 = shuffle_position % self.__max_position
        output_core, reward_list, changed, info = self.change(int(m_position1), int(m_position2))
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(self.cross_set)
        return output_core, reward, changed, info, satisfied

    def step_shuffle_full(self, shuffle_position, reward_index):
        m_position1 = int(shuffle_position / self.__max_position) * Astra.n_move
        m_position2 = shuffle_position % self.__max_position
        output_core, reward_list, changed, info = self.change(int(m_position1), int(m_position2))
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(self.cross_set)
        return output_core, reward, reward_list, changed, info, satisfied

    def step_rotate(self, rotate_position, reward_index):
        m_position1 = int(rotate_position / Astra.n_rotate) + Astra.rotate1 + rotate_position % Astra.n_rotate
        output_core, reward_list, changed, info = self.change(int(m_position1), None)
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(self.cross_set)
        return output_core, reward, changed, info, satisfied

    def step_bp(self, bp_position, reward_index):
        m_position1 = int(bp_position / Astra.n_bp) + Astra.bp_in + bp_position % Astra.n_bp
        output_core, reward_list, changed, info = self.change(int(m_position1), None)
        reward, best, satisfied = self.__calculate_reward(reward_list, reward_index)
        if best:
            self.__input_core_best.append(copy.deepcopy(self.__astra_input_reader.blocks[AstraInputReader.shuff_block]).core)
            self.__train_set_best.append(AstraTrainSet(output_core, None, reward_list, False, None))
            self.__crosspower_set_best.append(self.cross_set)
        return output_core, reward, changed, info, satisfied

    def change(self,  m_position1, position2):
        position1 = int(m_position1 / Astra.n_move)

        next_core = copy.deepcopy(self.__input_core_history[-1])
        shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]
        shuffle_block.core = next_core

        changed = False
        if m_position1 % Astra.n_move == Astra.shuffle and position2:
            changed = shuffle_block.core.shuffle(self.__position[position1], self.__position[position2])
        elif m_position1 % Astra.n_move == Astra.rotate1:
            changed = shuffle_block.core.rotate(self.__position[position1])
        elif m_position1 % Astra.n_move == Astra.rotate2:
            shuffle_block.core.rotate(self.__position[position1])
            changed = shuffle_block.core.rotate(self.__position[position1])
        elif m_position1 % Astra.n_move == Astra.rotate3:
            shuffle_block.core.rotate(self.__position[position1])
            shuffle_block.core.rotate(self.__position[position1])
            changed = shuffle_block.core.rotate(self.__position[position1])
        elif m_position1 % Astra.n_move == Astra.bp_in:
            changed = shuffle_block.core.poison(self.__position[position1], True)
        elif m_position1 % Astra.n_move == Astra.bp_de:
            changed = shuffle_block.core.poison(self.__position[position1], False)

        if not changed:
            self.__input_core_history.append(next_core)
            copy_train_set = copy.deepcopy(self.__train_set_history[-1])
            self.__train_set_history.append(copy.deepcopy(self.__train_set_history[-1]))
            self.__crosspower_set_history.append(copy.deepcopy(self.__crosspower_set_history[-1]))
            return copy_train_set.input, 0.0, changed, True

        self.__astra_input_reader.blocks[AstraInputReader.shuff_block] = shuffle_block

        self.__input_core_history.append(next_core)
        #output_core, reward_list, changed, successful = self.run_process_astra_3D(shuffle_block)
        output_core, reward_list, changed, successful = self.run_process_astra(shuffle_block)

        self.__train_set_history.append(AstraTrainSet(output_core, None, reward_list, False, None))
        self.__crosspower_set_history.append(copy.deepcopy(self.cross_set))

        return output_core, reward_list, changed, successful

    def step(self, m_position1, position2):
        output_core, reward_list, changed, info = self.change(m_position1, position2)
        return output_core, self.calculate_reward(reward_list), info, changed

    def run_process_astra(self, shuffle_block=None):

        if not shuffle_block:
            shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]

        output_string = self.run_astra(shuffle_block)

        if output_string:

            self.__reading_out = AstraOutputReader(output_string=output_string)

            output_core, reward_list, successful = self.__reading_out.process_astra()

            cross_power_density_list, successful = self.__reading_out.process_astra_cross_power()
            if cross_power_density_list:
                depletions = []
                for list in cross_power_density_list:
                    depletions.append(CrossPowerSet(list[0],list[1], list[2], list[3], list[4], list[5], list[6],list[7]))
                self.cross_set = depletions

            return output_core, reward_list, True, successful

        return None, None, True, False

    def run_process_astra_3D(self, shuffle_block=None):

        if not shuffle_block:
            shuffle_block = self.__astra_input_reader.blocks[AstraInputReader.shuff_block]

        output_string = self.run_astra(shuffle_block)

        if output_string:

            self.__reading_out = AstraOutputReader(output_string=output_string)

            output_core, reward_list, successful = self.__reading_out.process_astra()

            cross_power_density_list, successful = self.__reading_out.process_astra_cross_power_3D()
            if successful:
                depletions = []
                for list in cross_power_density_list:
                    depletions.append(CrossPowerSet3D(list[0],list[1], list[2], list[3], list[4], list[5], list[6],list[7]))
                self.cross_set = depletions
            else:
                self.cross_set = None

            return output_core, reward_list, True, successful

        return None, None, True, False

    def run_astra(self, shuffle_block):

        replaced = self.__astra_input_reader.replace_block([shuffle_block])
        time.sleep(1)

        try:
            p = subprocess.Popen(['/home/youndukn/bin/astra.1.4.2'],
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

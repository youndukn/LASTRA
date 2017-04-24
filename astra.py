#System Module
from subprocess import Popen, PIPE
from random import randint
import subprocess
import copy
import os

from astra_io.astra_output_reader import AstraOutputReader

class Astra():
    def __init__(self, astra_input_reader):

        self.astra_input_reader = astra_input_reader
        self.original_core = copy.deepcopy(self.astra_input_reader.blocks[0].core)
        print(self.astra_input_reader.blocks[0].print_block())
        self.n_move = 6

        self.shuffle = 0
        self.rotate1 = 1
        self.rotate2 = 2
        self.rotate3 = 3
        self.bp_in = 4
        self.bp_de = 5

        self.max_row = 10
        self.max_col = 10
        self.position = []
        self.set_oct_position()
        self.max_position = self.max_row * (self.max_col - 1) / 2 + self.max_row
        self.absolute_reward = 0
        self.original_reward = 0
        self.working_directory = ".{}".format(os.path.sep)

    def make(self):
        output_string = self.run_astra(self.astra_input_reader.blocks[0])
        if output_string:
            reading_out = AstraOutputReader(output_string=output_string)
            reading_out.parse_block_contents()
            self.calculate_reward(reading_out.blocks[3].lists())
            self.original_reward = self.absolute_reward
            return True
        else:
            return False

    def reset(self):
        self.astra_input_reader.blocks[0].core = copy.deepcopy(self.original_core)
        self.absolute_reward = self.original_reward
        return

    def change(self,  m_position1, position2):
        position1 = int(m_position1 / self.n_move)

        shuffle_block = self.astra_input_reader.blocks[0]
        changed = False
        if m_position1 % self.n_move == self.shuffle and position2:
            changed = shuffle_block.core.shuffle(self.position[position1], self.position[position2])
        elif m_position1 % self.n_move == self.rotate1:
            changed = shuffle_block.core.rotate(self.position[position1])
        elif m_position1 % self.n_move == self.rotate2:
            shuffle_block.core.rotate(self.position[position1])
            changed = shuffle_block.core.rotate(self.position[position1])
        elif m_position1 % self.n_move == self.rotate3:
            shuffle_block.core.rotate(self.position[position1])
            shuffle_block.core.rotate(self.position[position1])
            changed = shuffle_block.core.rotate(self.position[position1])
        elif m_position1 % self.n_move == self.bp_in:
            changed = shuffle_block.core.poison(self.position[position1], True)
        elif m_position1 % self.n_move == self.bp_de:
            changed = shuffle_block.core.poison(self.position[position1], False)
        self.astra_input_reader.blocks[0] = shuffle_block

        if not changed:
            return shuffle_block.core, 0.0, changed, False

        return self.run_process_astra(shuffle_block)

    def run_process_astra(self, shuffle_block=None):

        if not shuffle_block:
            shuffle_block = self.astra_input_reader.blocks[0]

        output_string = self.run_astra(shuffle_block)

        if output_string:

            reading_out = AstraOutputReader(output_string=output_string)
            reading_out.parse_block_contents()

            core, lists, successful = reading_out.process_astra()

            if not successful:
                return shuffle_block.core, None, True, successful
            else:
                return shuffle_block.core, lists, True, successful

        return shuffle_block.core, None, True, False

    def step(self, m_position1, position2):
        core, lists, changed, info = self.change(m_position1, position2)
        return core, self.calculate_reward(lists), info, changed

    def run_astra(self, shuffle_block):

        replaced = self.astra_input_reader.replace_block([shuffle_block])

        try:
            p = subprocess.Popen(['astra'],
                                 stdout=subprocess.PIPE,
                                 stdin=subprocess.PIPE,
                                 shell=True,
                                 cwd=self.working_directory)
            output = (p.communicate(input=replaced.encode('utf-8'))[0]).decode('utf-8')
            return output
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

        position = randint(0, self.max_position-1)
        position *= self.n_move
        position += next_move

        return position

    def get_oct_shuffle_space(self, space_action):
        return randint(0, self.max_position-1)

    def set_oct_position(self):
        for row in range(self.max_col):
            for col in range(row, self.max_col):
                self.position.append([row, col])

    def calculate_reward(self, lists):
        burnup = 0
        cbc = 0
        fr = 0
        fz = 0
        fxy = 0
        fq = 0

        for a_list in lists:
            burnup = a_list[1] if burnup < a_list[1] else burnup
            cbc = a_list[3] if cbc < a_list[3] else cbc
            fr = a_list[7] if fr < a_list[7] else fr
            fz = a_list[8] if fz < a_list[8] else fz
            fxy = a_list[9] if fxy < a_list[9] else fxy
            fq = a_list[10] if fq < a_list[10] else fq

        reward = burnup / 100000 + cbc / 3000 + fr / 3 + fz / 3 + fxy / 3
        absolute_reward = self.absolute_reward
        self.absolute_reward = reward
        reward -= absolute_reward

        return reward

#System Module
from subprocess import Popen, PIPE
from random import randint
import subprocess
import copy
import os
from data.astra_train_set import AstraTrainSet
from astra_io.astra_input_reader import AstraInputReader
from astra_io.astra_output_reader import AstraOutputReader
from convolutional import Convolutional

class Astra():

    n_move = 10

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

    def __init__(self, astra_input_reader):

        #Astra Input Reader
        self.astra_input_reader = astra_input_reader

        #Core information
        self.original_core = copy.deepcopy(self.astra_input_reader.blocks[AstraInputReader.shuff_block].core)
        self.max_row = self.original_core.max_row
        self.max_col = self.original_core.max_col

        self.max_position = self.max_row * (self.max_col - 1) / 2 + self.max_row

        self.position = self.set_oct_position()

        self.absolute_reward = 0
        self.original_reward = 0
        self.working_directory = ".{}".format(os.path.sep)

        self.train_set = AstraTrainSet(None, None, None)
        self.change_data = [self.train_set]


    def reset(self):
        self.astra_input_reader.blocks[AstraInputReader.shuff_block].core = copy.deepcopy(self.original_core)
        self.absolute_reward = self.original_reward
        self.change_data = [self.train_set]
        return

    def change(self,  m_position1, position2):
        position1 = int(m_position1 / Astra.n_move)

        shuffle_block = self.astra_input_reader.blocks[AstraInputReader.shuff_block]
        changed = False
        if m_position1 % Astra.n_move == Astra.shuffle and position2:
            changed = shuffle_block.core.shuffle(self.position[position1], self.position[position2])
        elif m_position1 % Astra.n_move == Astra.rotate1:
            changed = shuffle_block.core.rotate(self.position[position1])
        elif m_position1 % Astra.n_move == Astra.rotate2:
            shuffle_block.core.rotate(self.position[position1])
            changed = shuffle_block.core.rotate(self.position[position1])
        elif m_position1 % Astra.n_move == Astra.rotate3:
            shuffle_block.core.rotate(self.position[position1])
            shuffle_block.core.rotate(self.position[position1])
            changed = shuffle_block.core.rotate(self.position[position1])
        elif m_position1 % Astra.n_move == Astra.bp_in:
            changed = shuffle_block.core.poison(self.position[position1], True)
        elif m_position1 % Astra.n_move == Astra.bp_de:
            changed = shuffle_block.core.poison(self.position[position1], False)
        self.astra_input_reader.blocks[AstraInputReader.shuff_block] = shuffle_block

        if not changed:
            return shuffle_block.core, 0.0, changed, False

        return self.run_process_astra(shuffle_block)

    def run_process_astra(self, shuffle_block=None):

        if not shuffle_block:
            shuffle_block = self.astra_input_reader.blocks[AstraInputReader.shuff_block]

        output_string = self.run_astra(shuffle_block)

        if output_string:

            reading_out = AstraOutputReader(output_string=output_string)
            reading_out.parse_block_contents()

            core, lists, successful = reading_out.process_astra()

            if not successful:
                return core, None, True, successful
            else:
                return core, lists, True, successful

        return None, None, True, False

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
        position = []
        for row in range(self.max_col):
            for col in range(row, self.max_col):
                position.append([row, col])
        return position


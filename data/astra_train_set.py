from data.train_set import TrainSet
import numpy as np

class AstraTrainSet(TrainSet):

    def __init__(self, input, output, reward, total_reward):
        super(AstraTrainSet, self).__init__(input, output, reward, total_reward)
        self.process_input()
        self.process_output()

    def process_input(self):
        if self.input:

            input_array = np.zeros([3, 20, 20], dtype=np.int)
            nodewise = [self.input.get_value_matrix(0), self.input.get_value_matrix(1), self.input.get_value_matrix(2), self.input.get_value_matrix(3)]

            concentration = self.input.get_value_matrix(5)
            poison = self.input.get_value_matrix(6)

            for i in range(20):
                for j in range(20):
                    j_swap = j % 2
                    i_swap = i % 2
                    j_index = int(j / 2)
                    i_index = int(i / 2)
                    input_array[0, i, j] = nodewise[i_swap + j_swap][i_index][j_index]


    def process_output(self):
        self.output = self.output
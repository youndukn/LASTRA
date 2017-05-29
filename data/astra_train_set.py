from data.train_set import TrainSet
import numpy as np

class AstraTrainSet(TrainSet):

    def __init__(self, input, output, reward, done, total_reward):
        super(AstraTrainSet, self).__init__(input, output, reward, done, total_reward)
        self.state, self.state2 = self.process_input()
        self.first_move, self.second_move = self.process_output()

    def process_input(self):
        if self.input:

            input_array = np.zeros([20, 20, 3], dtype=np.int)

            nodewise = [self.input.get_value_matrix(0), self.input.get_value_matrix(1), self.input.get_value_matrix(2), self.input.get_value_matrix(3)]

            concentration = self.input.get_value_matrix(4)
            poison = self.input.get_value_matrix(5)

            for i in range(20):
                for j in range(20):
                    j_swap = j % 2
                    i_swap = i % 2
                    j_index = int(j / 2)
                    i_index = int(i / 2)
                    input_array[i, j, 0] = int(int(nodewise[(2*i_swap) + j_swap][i_index][j_index])/100)
                    input_array[i, j, 1] = int(int(concentration[i_index][j_index])*100)
                    input_array[i, j, 2] = int(int(poison[i_index][j_index]) / 100000) * \
                                           (int(int(poison[i_index][j_index]) / 1000) - \
                                            int(int(poison[i_index][j_index]) / 100000)*100)

            input_array2 = np.zeros([3, 20, 20], dtype=np.int)

            for i in range(20):
                for j in range(20):
                    for k in range(3):
                        input_array2[k, i, j] = input_array[i, j, k]

            return input_array, input_array2
        return None, None

    def process_output(self):
        if self.output:
            a = np.zeros(550)
            np.put(a, self.output[0], 1)

            if self.output[1]:
                b = np.zeros(55)
                np.put(b, self.output[1], 1)
                return a, b
            return a, None
        return None, None

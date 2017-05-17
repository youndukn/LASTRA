from data.train_set import TrainSet
import numpy as np

class AstraTrainSet(TrainSet):

    def __init__(self, input, output, reward, total_reward):
        super(AstraTrainSet, self).__init__(input, output, reward, total_reward)
        process_input()

    def process_input(self):

        input_array = np.zeros([3, 20, 20], dtype=np.int)

        nodewise = self.input[0]
        concentration = self.input[1]
        poison = self.input[2]

        input_array


    def process_output(self):
        self.
        np.zeros([size, num_actions])
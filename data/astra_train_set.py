from data.train_set import TrainSet
import numpy as np

class AstraTrainSet(TrainSet):

    def __init__(self, input, output, reward, total_reward):
        super(AstraTrainSet, self).__init__(input, output, reward, total_reward)
        self.processed_input = self.process_input()
        self.processed_output = self.process_output()

    def process_input(self):
        if self.input:

            input_array = np.zeros([3, 20, 20], dtype=np.int)
            nodewise = [self.input.get_value_matrix(0), self.input.get_value_matrix(1), self.input.get_value_matrix(2), self.input.get_value_matrix(3)]

            poison = self.input.get_value_matrix(5)
            concentration = self.input.get_value_matrix(6)

            for i in range(20):
                for j in range(20):
                    j_swap = j % 2
                    i_swap = i % 2
                    j_index = int(j / 2)
                    i_index = int(i / 2)
                    input_array[0, i, j] = nodewise[(2*i_swap) + j_swap][i_index][j_index]
                    input_array[1, i, j] = int(int(concentration[i_index][j_index])*100)
                    input_array[2, i, j] = int(int(poison[i_index][j_index]) / 100000) * \
                                           (int(int(poison[i_index][j_index]) / 1000) - \
                                            int(int(poison[i_index][j_index]) / 100000)*100)

            return input_array

    def process_output(self):
        if self.output:
            a = np.zeros(10)
            np.put(a, self.output, 1)
            return a
        return None

    def convertToOneHot(vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        if num_classes is None:
            num_classes = np.max(vector) + 1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)
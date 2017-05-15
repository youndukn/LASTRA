from data.train_set import TrainSet

class AstraTrainSet(TrainSet):

    def __init__(self, input, output, reward):
        super(AstraTrainSet, self).__init__(input, output, reward, total_reward)
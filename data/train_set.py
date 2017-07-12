class TrainSet():

    def __init__(self, input, output, reward, done=False, total_reward=0.0):
        self.input = input
        self.output = output
        self.reward = reward
        self.total_reward = total_reward
        self.done = done
        if not done:
            self.done = False

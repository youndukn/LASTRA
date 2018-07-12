import random


class Space:
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return random.randrange(self.shape[0])

class Enviroment:
    def __init__(self):
        self.action_space = Space((3300,))
        self.observation_space = Space((19, 19, 28))
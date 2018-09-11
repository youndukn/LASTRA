import random
import numpy

class Space:
    def __init__(self, shape, shapes=[1],  high = 1, low = -1):
        self.shape = shape
        self.shapes = shapes
        sum = 0
        for shape_value in self.shapes:
            sum += shape_value
        self.shape_sum = sum
        self.high = high
        self.low = low

    def sample(self, ratios = [1]):

        base = random.randrange(self.shape[0])

        if len(ratios) > 1 and len(self.shapes)>1:
            value = numpy.random.f()
            a_ratio = 0
            ratio_index = 0
            shape_index_sum = 0
            for index, ratio in enumerate(ratios):
                a_ratio += ratio
                if value < a_ratio:
                    break
                shape_index_sum += self.shapes[index]
                ratio_index += 1
            the_index = random.randrange(self.shapes[ratio_index])
            base = random.randrange(self.shape[0]/self.shape_sum)
            base = base+shape_index_sum+the_index

        zeros = numpy.zeros(self.shape)
        zeros[base] = 1
        return zeros

class Environment:
    def __init__(self,
                 action_main_shape = (3300,),
                 action_sub_shapes = (55, 3, 2),
                 observation_shape = (19, 19, 28)):

        self.action_space = Space(action_main_shape, shapes=action_sub_shapes, high =  100, low = -100)
        self.observation_space = Space(observation_shape)
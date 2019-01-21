
class AstraPreShuffleReader():

    def __init__(self, file_name="input_shuff"):
        self.pre_shuff = []
        with open(file_name) as file:
            for line in file:
                values = line.split()
                self.pre_shuff.append(values[1:4])
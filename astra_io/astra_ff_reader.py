import numpy

class AstraFFReader:
    def __init__(self, ff_name=None, main_directory=None):

        f = open(main_directory+ff_name, "r")
        counter = 0
        self.ff = {}
        self.depltion = []

        while True:
            line = f.readline()
            if not line: break

            type = counter%(3+16*4+1)

            if type == 1:
                split = line.split()
                name = split[0]

            if type == 2:
                split = line.split()
                depletion = int(float(split[0]))
                counter += 1
                self.depltion.append(depletion)
                first = numpy.zeros((1, 4, 16, 16), dtype=numpy.float16)
                for ff_type in range(4):
                    for row in range(16):
                        line = f.readline()
                        counter += 1
                        split = [line[i:i + 6] for i in range(0, len(line), 6)]
                        for col, value in enumerate(split[:-1]):
                            first[0][ff_type][row][col] = numpy.float16(value)
                if name in self.ff:
                    self.ff[name] = numpy.concatenate((self.ff[name], first))
                else:
                    self.ff[name] = first

            else:
                counter += 1

"""
read = AstraFFReader("24MPLUS7_REFL_XSE.FF", "../")
print(read.ff)
"""
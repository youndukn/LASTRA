from data.assembly import Assembly
from data.assembly import FreshAssembly
from data.assembly import ShuffleAssembly
import copy

class Core:
    def __init__(self, block_name):
        self.block_name = block_name
        self.max_row = 10
        self.max_col = 10
        self.assemblies = [[Assembly() for x in range(self.max_col)] for y in range(self.max_row)]
        self.batches = []


    def set_batches(self, batches):
        """
        Set deep copcy of batch types and sort them
        :param batches: batches to be used
        :return:
        """
        self.batches = sorted(copy.deepcopy(batches))

    def set_shuffle_at(self, assembly, row, col):
        """
        Set shuffling information at row and col
        :param assembly: assembly to be copied
        :param row: row
        :param col: col
        :return: success
        """
        if row >= self.max_row or col >= self.max_row:
            raise IndexError("Maximum row {}, : col {}".format(self.max_row, self.max_col))
        if not issubclass(type(assembly), Assembly):
            return False

        self.assemblies[row][col] = copy.deepcopy(assembly)
        return True

    def shuffle(self, position1, position2):
        """
        Shuffle assemblies at position1 and position2
        :param position1: row and col of first point
        :param position2: row and col of second point
        :return: Success
        """
        if type(self.assemblies[position1[0]][position1[1]]) is Assembly or \
                        type(self.assemblies[position2[0]][position2[1]]) is Assembly:
            return False

        if position1[0] == position2[0] and position1[1] == position2[1]:
            return False

        if position1[0] == 0 and position1[1] == 0:
            return False

        if position2[0] == 0 and position2[1] == 0:
            return False

        sym_posi1 = [position1[1], position1[0]]
        sym_posi2 = [position2[1], position2[0]]

        q1 = self.get_quadrant(position1)
        q2 = self.get_quadrant(position2)

        if q1 == 0 and q2 == 0:
            self.__swap_assemblies(position1, position2)
            return True
        elif q1 == q2:
            self.__swap_assemblies(position1, position2)
            self.__swap_assemblies(sym_posi1, sym_posi2)
            return True
        elif q1 == 0 and q2 != 3:
            self.__swap_assemblies(position1, position2)
            self.copy_assembly(position2, sym_posi2)
            if type(self.assemblies[sym_posi2[0]][sym_posi2[1]]) is FreshAssembly:
                return True
            elif q2 == 1:
                self.assemblies[sym_posi2[0]][sym_posi2[1]].rotate()
                self.assemblies[sym_posi2[0]][sym_posi2[1]].rotate()
                self.assemblies[sym_posi2[0]][sym_posi2[1]].rotate()
                return True
            elif q2 == 2:
                self.assemblies[sym_posi2[0]][sym_posi2[1]].rotate()
                return True
        elif q2 == 0 and q1 != 3:
            self.__swap_assemblies(position1, position2)
            self.copy_assembly(position1, sym_posi1)
            if type(self.assemblies[sym_posi1[0]][sym_posi1[1]]) is FreshAssembly:
                return True
            elif q1 == 1:
                self.assemblies[sym_posi1[0]][sym_posi1[1]].rotate()
                self.assemblies[sym_posi1[0]][sym_posi1[1]].rotate()
                self.assemblies[sym_posi1[0]][sym_posi1[1]].rotate()
                return True
            elif q1 == 2:
                self.assemblies[sym_posi1[0]][sym_posi1[1]].rotate()
            return True

        return False

    def rotate(self, position):
        """
        Rotate assembly at position
        :param position:
        :return: Success
        """
        if type(self.assemblies[position[0]][position[1]]) is not ShuffleAssembly:
            return False

        self.assemblies[position[0]][position[1]].rotate()
        if position[0] != position[1]:
            self.assemblies[position[1]][position[0]].rotate()

        return True

    def poison(self, position, increase):
        """
        Poison Rod Batch Increase
        :param position:
        :param increase:
        :return:
        """
        if len(self.batches) == 0:
            raise IOError("Batches not found in core".format(self.max_row, self.max_col))

        assembly = self.assemblies[position[0]][position[1]]

        if type(assembly) is not FreshAssembly:
            return False

        batch = assembly.get_batch()
        batch_found = None
        if increase:
            for x in range(len(self.batches) - 1):
                if batch == self.batches[x]:
                    batch_found = self.batches[x + 1]
        else:
            for x in range(1, len(self.batches)):
                if batch == self.batches[x]:
                    batch_found = self.batches[x - 1]

        if batch_found:
            self.assemblies[position[0]][position[1]].set_batch(batch_found)
            self.assemblies[position[1]][position[0]].set_batch(batch_found)
            return True

        return False

    def concetration(self, position, increase):
        return

    def copy_assembly(self, from_position, to_position):

        if type(self.assemblies[to_position[0]][to_position[1]]) is Assembly:
            return False

        self.assemblies[to_position[0]][to_position[1]] = \
            copy.deepcopy(self.assemblies[from_position[0]][from_position[1]])

        return True

    def get_value_matrix(self, index):
        a_matrix = []
        for row_assemblies in self.assemblies:
            a_array = []
            for assembly in row_assemblies:
                if len(assembly.get_values()) > index:
                    a_array.append(assembly.get_values()[index])
                else:
                    a_array.append(0)
            a_matrix.append(a_array)
        return a_matrix

    def get_batch_matrix(self):
        a_matrix = []
        for row_assemblies in self.assemblies:
            a_array = []
            for assembly in row_assemblies:
                a_array.append(assembly.get_batch())
            a_matrix.append(a_array)
        return a_matrix

    def __swap_assemblies(self, position1, position2):
        assembly_temp = self.assemblies[position1[0]][position1[1]]
        self.assemblies[position1[0]][position1[1]] = self.assemblies[position2[0]][position2[1]]
        self.assemblies[position2[0]][position2[1]] = assembly_temp


    @staticmethod
    def sym(position):
        return [position[1], position[0]]

    @staticmethod
    def get_quadrant(position):
        if position[0] == position[1]:
            return 0
        elif position[0] == 0:
            return 1
        elif position[1] == 0:
            return 2
        else:
            return 3


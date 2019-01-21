from data.assembly import Assembly
from data.assembly import FreshAssembly
from data.assembly import ShuffleAssembly
import copy

fresh_assembly = 0
shuffle_assembly = 1

fr_fb = [
    [True,  True,  True,  True,  True,  True,  True,  False, False, False],
    [True,  True,  True,  True,  True,  True,  True,  False, False, False],
    [True,  True,  True,  True,  True,  True,  True,  False, False, False],
    [True,  True,  True,  True,  True,  True,  True,  False, False, False],
    [True,  True,  True,  True,  True,  True,  False, False, False, False],
    [True,  True,  True,  True,  True,  False, False, False, False, False],
    [True,  True,  True,  True,  False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False, False],
]
class Core:
    def __init__(self, block_name):
        self.block_name = block_name
        self.max_row = 10
        self.max_col = 10
        self.assemblies = [[Assembly() for x in range(self.max_col)] for y in range(self.max_row)]
        self.batches = []
        self.cross_section = None
        self.batches_cross = {}

    def set_cross_section(self, cross_section):
        self.cross_section = cross_section

        for row, row_assemblies in enumerate(self.assemblies):
            for col, col_assembly in enumerate(row_assemblies):
                if type(col_assembly) is FreshAssembly:
                    r_row = (row+9)*2
                    r_col = (col+9)*2

                    self.batches_cross[col_assembly.get_batch()] = \
                        copy.deepcopy(self.cross_section[r_row:r_row+1][r_col:r_col+1])

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

        if (type(self.assemblies[position1[0]][position1[1]]) is FreshAssembly and
                not fr_fb[position2[0]][position2[1]]):
            return False

        if (type(self.assemblies[position2[0]][position2[1]]) is FreshAssembly and
                not fr_fb[position1[0]][position1[1]]):
            return False

        sym_posi1 = [position1[1], position1[0]]
        sym_posi2 = [position2[1], position2[0]]

        q1 = self.get_quadrant(position1)
        q2 = self.get_quadrant(position2)

        if q1 == 2 or q2 == 2:
            return False

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

    def shuffle_cross(self, position1, position2):

        sym_posi1 = [position1[1], position1[0]]
        sym_posi2 = [position2[1], position2[0]]

        q1 = self.get_quadrant(position1)
        q2 = self.get_quadrant(position2)

        is_sym = False

        cross_position1 = [position1[0] + 9, position1[1] + 9]
        cross_position1_1 = [position1[1] + 9, -1 * position1[0] + 9]
        cross_position1_2 = [-1 * position1[0] + 9, -1 * position1[1] + 9]
        cross_position1_3 = [-1 * position1[1] + 9, position1[0] + 9]

        cross_position2 = [position2[0] + 9, position2[1] + 9]
        cross_position2_1 = [position2[1] + 9, -1 * position2[0] + 9]
        cross_position2_2 = [-1 * position2[0] + 9, -1 * position2[1] + 9]
        cross_position2_3 = [-1 * position2[1] + 9, position2[0] + 9]

        if q1 == 3:
            is_sym = True
            sym_cross_position1 = [sym_posi1[0] + 9, sym_posi1[1] + 9]
            sym_cross_position1_1 = [sym_posi1[1] + 9, -1 * sym_posi1[0] + 9]
            sym_cross_position1_2 = [-1 * sym_posi1[0] + 9, -1 * sym_posi1[1] + 9]
            sym_cross_position1_3 = [-1 * sym_posi1[1] + 9, sym_posi1[0] + 9]

        if q2 == 3:
            sym_cross_position2 = [sym_posi2[0] + 9, sym_posi2[1] + 9]
            sym_cross_position2_1 = [sym_posi2[1] + 9, -1 * sym_posi2[0] + 9]
            sym_cross_position2_2 = [-1 * sym_posi2[0] + 9, -1 * sym_posi2[1] + 9]
            sym_cross_position2_3 = [-1 * sym_posi2[1] + 9, sym_posi2[0] + 9]

        self.__swap_node_cross(cross_position1, cross_position2)
        self.__swap_node_cross(cross_position1_1, cross_position2_1)
        self.__swap_node_cross(cross_position1_2, cross_position2_2)
        self.__swap_node_cross(cross_position1_3, cross_position2_3)

        if is_sym:
            self.__swap_node_cross(sym_cross_position1, sym_cross_position2)
            self.__swap_node_cross(sym_cross_position1_1, sym_cross_position2_1)
            self.__swap_node_cross(sym_cross_position1_2, sym_cross_position2_2)
            self.__swap_node_cross(sym_cross_position1_3, sym_cross_position2_3)

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

    def rotate_cross(self, position1):

        sym_posi1 = [position1[1], position1[0]]

        q1 = self.get_quadrant(position1)

        is_sym = False

        cross_position1 = [position1[0] + 9, position1[1] + 9]
        cross_position1_1 = [position1[1] + 9, -1 * position1[0] + 9]
        cross_position1_2 = [-1 * position1[0] + 9, -1 * position1[1] + 9]
        cross_position1_3 = [-1 * position1[1] + 9, position1[0] + 9]

        if q1 == 3:
            is_sym = True
            sym_cross_position1 = [sym_posi1[0] + 9, sym_posi1[1] + 9]
            sym_cross_position1_1 = [sym_posi1[1] + 9, -1 * sym_posi1[0] + 9]
            sym_cross_position1_2 = [-1 * sym_posi1[0] + 9, -1 * sym_posi1[1] + 9]
            sym_cross_position1_3 = [-1 * sym_posi1[1] + 9, sym_posi1[0] + 9]

        self.__rotate_node_cross(cross_position1)
        self.__rotate_node_cross(cross_position1_1)
        self.__rotate_node_cross(cross_position1_2)
        self.__rotate_node_cross(cross_position1_3)

        if is_sym:
            self.__rotate_node_cross(sym_cross_position1)
            self.__rotate_node_cross(sym_cross_position1_1)
            self.__rotate_node_cross(sym_cross_position1_2)
            self.__rotate_node_cross(sym_cross_position1_3)

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

    def poison_cross(self, position1):

        sym_posi1 = [position1[1], position1[0]]

        q1 = self.get_quadrant(position1)

        is_sym = False

        assembly = self.assemblies[position1[0]][position1[1]]
        batch = assembly.get_batch()

        cross_position1 = [position1[0] + 9, position1[1] + 9]
        cross_position1_1 = [position1[1] + 9, -1 * position1[0] + 9]
        cross_position1_2 = [-1 * position1[0] + 9, -1 * position1[1] + 9]
        cross_position1_3 = [-1 * position1[1] + 9, position1[0] + 9]

        if q1 == 3:
            is_sym = True
            sym_cross_position1 = [sym_posi1[0] + 9, sym_posi1[1] + 9]
            sym_cross_position1_1 = [sym_posi1[1] + 9, -1 * sym_posi1[0] + 9]
            sym_cross_position1_2 = [-1 * sym_posi1[0] + 9, -1 * sym_posi1[1] + 9]
            sym_cross_position1_3 = [-1 * sym_posi1[1] + 9, sym_posi1[0] + 9]

        self.__poison_node_cross(cross_position1, batch)
        self.__poison_node_cross(cross_position1_1, batch)
        self.__poison_node_cross(cross_position1_2, batch)
        self.__poison_node_cross(cross_position1_3, batch)

        if is_sym:
            self.__poison_node_cross(sym_cross_position1, batch)
            self.__poison_node_cross(sym_cross_position1_1, batch)
            self.__poison_node_cross(sym_cross_position1_2, batch)
            self.__poison_node_cross(sym_cross_position1_3, batch)


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

    def get_node_matrix(self, index, e_numb=2):
        a_matrix = []

        for row, row_assemblies in enumerate(self.assemblies):
            f_array = []
            s_array = []
            for col, assembly in enumerate(row_assemblies):

                if col == 0 and row == 0:
                    s_array.append(assembly.get_values()[index])
                elif row == 0:
                    if len(assembly.get_values()) > 0:
                        s_array.append(assembly.get_values()[index*e_numb])
                        s_array.append(assembly.get_values()[index*e_numb+1])
                    else:
                        s_array.append(0)
                        s_array.append(0)
                elif col == 0:
                    if len(assembly.get_values()) > 0:
                        f_array.append(assembly.get_values()[index])
                        s_array.append(assembly.get_values()[index+e_numb])
                    else:
                        f_array.append(0)
                        s_array.append(0)
                else:
                    if len(assembly.get_values()) > 0:
                        f_array.append(assembly.get_values()[index * e_numb])
                        f_array.append(assembly.get_values()[index * e_numb + 1])
                        s_array.append(assembly.get_values()[index * e_numb + 2*e_numb])
                        s_array.append(assembly.get_values()[index * e_numb + 2*e_numb+1])
                    else:
                        f_array.append(0)
                        f_array.append(0)
                        s_array.append(0)
                        s_array.append(0)
            if len(f_array) > 0:
                a_matrix.append(f_array)
            a_matrix.append(s_array)

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

    def __swap_node_cross(self, position1, position2):
        r_position1 = [position1[0] * 2,   position1[1] * 2]
        r_position2 = [position2[0] * 2,   position2[1] * 2]
        r_position3 = [position1[0] * 2,   position1[1] * 2+1]
        r_position4 = [position2[0] * 2,   position2[1] * 2+1]
        r_position5 = [position1[0] * 2+1, position1[1] * 2]
        r_position6 = [position2[0] * 2+1, position2[1] * 2]
        r_position7 = [position1[0] * 2+1, position1[1] * 2+1]
        r_position8 = [position2[0] * 2+1, position2[1] * 2+1]

        self.__swap_cross(r_position1, r_position2)
        self.__swap_cross(r_position3, r_position4)
        self.__swap_cross(r_position5, r_position6)
        self.__swap_cross(r_position7, r_position8)

    def __rotate_node_cross(self, position1):
        r_position1 = [position1[0] * 2,   position1[1] * 2]
        r_position2 = [position1[0] * 2,   position1[1] * 2+1]
        r_position3 = [position1[0] * 2+1, position1[1] * 2]
        r_position4 = [position1[0] * 2+1, position1[1] * 2+1]

        cross_section_temp1 = copy.deepcopy(self.cross_section[r_position1[0]][r_position1[1]])
        cross_section_temp2 = copy.deepcopy(self.cross_section[r_position2[0]][r_position2[1]])
        cross_section_temp3 = copy.deepcopy(self.cross_section[r_position3[0]][r_position3[1]])
        cross_section_temp4 = copy.deepcopy(self.cross_section[r_position4[0]][r_position4[1]])

        self.cross_section[r_position1[0]][r_position1[1]] = cross_section_temp2
        self.cross_section[r_position2[0]][r_position2[1]] = cross_section_temp4
        self.cross_section[r_position3[0]][r_position3[1]] = cross_section_temp1
        self.cross_section[r_position4[0]][r_position4[1]] = cross_section_temp3

    def __poison_node_cross(self, position, batch):

        r_position = [position[0] * 2,   position[1] * 2]
        self.cross_section[r_position[0]:r_position[0]+1][r_position[1]:r_position[1]+1] \
            = copy.deepcopy(self.batches_cross[batch])

    def __swap_cross(self, position1, position2):
        cross_section_temp = copy.deepcopy(self.cross_section[position1[0]][position1[1]])
        self.cross_section[position1[0]][position1[1]] = self.cross_section[position2[0]][position2[1]]
        self.cross_section[position2[0]][position2[1]] = cross_section_temp

    def count_in_range(self, assembly_type, rl, cl, rh, ch):
        count = 0
        rh = min(self.max_row, rh)
        ch = min(self.max_col, ch)
        for row in range(rl, rh):
            for col in range(cl, ch):
                if assembly_type == fresh_assembly and type(self.assemblies[row][col]) == FreshAssembly:
                    count+=1
                if assembly_type == shuffle_assembly and type(self.assemblies[row][col]) == ShuffleAssembly:
                    count += 1
        return count
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


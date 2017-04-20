import re

from data.astra_block import AstraBlock
from data.core import Core


class AstraOutputBlock(AstraBlock):
    def __init__(self, block_name=None, key_names=None):
        self.tag = "Astra_Output_Block"
        self.delimiter = r'-'
        self.block_name = block_name
        self.key_names = []
        self.dictionary = {}
        self.separators = ' '
        if key_names:
            for key_name in key_names:
                self.key_names.append(key_name)

    def increment_key_for_value(self, key, value):
        max_value = 0
        for key_in_dic in self.dictionary.keys():
            if key in key_in_dic:
                row = re.findall(r'(?<=\()\d+(?:\.\d+)?(?=\))', key_in_dic)
                if row and int(row[0]) > max_value:
                    max_value = int(row[0])

        self.dictionary[key + "({})".format(max_value + 1)] = value

    def finalize(self):
        return


class AstraOutputCoreBlock(AstraOutputBlock):
    def __init__(self, block_name=None, key_names=None):
        super(AstraOutputCoreBlock, self).__init__(block_name, key_names)
        self.cores = []

    def finalize(self):
        for key_in_dic in self.dictionary.keys():
            if "Y/X" in key_in_dic:
                self.cores.append(self.__get_core_data(self.dictionary[key_in_dic]))

    def __get_core_data(self, value):

        core = Core(self.block_name)

        value_row = 0
        row = -1
        col_len = 0

        for line in str.splitlines(value):
            splitted = line.split()
            length = len(splitted)
            if value_row == 0:
                col_len = length
            elif re.search(r'[a-z]+', line, re.I):
                # Start value is row number so 1
                row += 1
                col_len = length - 1
                for col in range(1, length - 1):
                    core.assemblies[row][col - 1].set_batch(splitted[col])
            elif length == col_len * 2:
                # True node wise
                for col in range(0, length, 2):
                    core.assemblies[row][int(col / 2)].add_value(splitted[col])
                    core.assemblies[row][int(col / 2)].add_value(splitted[col + 1])
            elif length == col_len * 2 - 1:
                # Node wise but no left symmetry
                core.assemblies[row][0].add_value(splitted[0])
                for col in range(1, length, 2):
                    core.assemblies[row][int((col + 1) / 2)].add_value(splitted[col])
                    core.assemblies[row][int((col + 1) / 2)].add_value(splitted[col + 1])
            else:
                # Start value is value so 0
                for col in range(0, length):
                    core.assemblies[row][col].add_value(splitted[col])

            value_row += 1
        return core

    def print_block(self):
        string = " "
        for core in self.cores:
            string += super().print_block()
            for row in range(0, core.max_row):
                for col in range(0, core.max_col):
                    if core.assemblies[row][col].print_assembly():
                        string += core.assemblies[row][col].print_assembly() + ' '
                if core.assemblies[row][0].print_assembly():
                    string += "\n"
        return string


class AstraOutputListBlock(AstraOutputBlock):
    def __init__(self, block_name=None, key_names=None):
        super(AstraOutputListBlock, self).__init__(block_name, key_names)
        self.lists = []

    def finalize(self):
        for key_in_dic in self.dictionary.keys():
            if "---" in key_in_dic:
                splitted = self.get_list_data(self.dictionary[key_in_dic])
                if len(splitted) > 0:
                    self.lists.append(splitted)

    @staticmethod
    def get_list_data(value):
        value_list = []
        for line in str.splitlines(value):
            splitted = line.split()
            if len(splitted) > 2:
                value_list.append(splitted)
        return value_list

    def print_block(self):
        a_string = ""
        for a_list in self.lists:
            a_string += super().print_block()
            for lines in a_list:
                for values in lines:
                    a_string += str(values) + " "
                a_string += "\n"
        return a_string

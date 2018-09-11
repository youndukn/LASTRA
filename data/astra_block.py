import re
import os

from data.assembly import Assembly
from data.assembly import FreshAssembly
from data.assembly import ShuffleAssembly
from data.core import Core
from error import InputError


class AstraBlock:

    def __init__(self, block_name = None):
        self.tag = "Astra_Block"
        self.delimiter = '%'
        self.block_name = block_name
        self.key_names = []
        self.dictionary = {}
        self.separator = ' '

    def add_key_for_value(self, key, value):

        if not key or not value:
            return

        match_key = False
        for preset_key in self.key_names:
            if preset_key in key:
                match_key = True
        if not match_key:
            raise InputError("Key:", key + "do not match")
            return

        value_temp = ""
        for key_in_dic in self.dictionary.keys():
            if key_in_dic is key:
                value_temp = self.dictionary[key]

        value_temp += " " + value

        key = key.lstrip()
        key = key.rstrip()
        value_temp = value_temp.lstrip()
        value_temp = value_temp.rstrip()

        self.dictionary[key] = value_temp

    def set_value_separator(self, separator):
        self.separator = separator

    def finalize(self):
        return

    def print_block(self):
        return self.delimiter + self.block_name + "\n"


class AstraShuffleBlock(AstraBlock):
    def __init__(self):
        super(AstraShuffleBlock, self).__init__("LPD_SHF")
        self.key_names.append("SHUFFL")
        self.core = Core(self.block_name)

    def finalize(self):
        for key in self.dictionary.keys():
            row = re.findall(r'(?<=\()\d+(?:\.\d+)?(?=\))', key)
            if row is not None:
                values = self.dictionary[key].split(',')
                col = 0
                for assembly_string in values:
                    assembly_values = assembly_string.split()
                    if assembly_values:
                        if assembly_values[0] is "F":
                            assembly = FreshAssembly()
                            assembly.set_batch(assembly_values[1])
                            assembly.set_number(assembly_values[2])
                        else:
                            assembly = ShuffleAssembly()
                            assembly.set_cycle(assembly_values[0])
                            assembly.set_col(assembly_values[1])
                            assembly.set_row(assembly_values[2])
                            assembly.set_rotation(assembly_values[3])
                        self.core.set_shuffle_at(assembly, int(row[0])-1, col)
                        col += 1

    def print_block(self):
        string = super().print_block()
        for row in range(0, self.core.max_row):
            if type(self.core.assemblies[row][0]) is Assembly: break
            string += "\t{}({}) = ".format(self.key_names[0], row+1)
            for col in range(0, self.core.max_col):
                if type(self.core.assemblies[row][col]) is Assembly: break
                string += self.core.assemblies[row][col].print_assembly() + ' '
            string += "\n"
        return string


class AstraBatchBlock(AstraBlock):
    def __init__(self):
        super(AstraBatchBlock, self).__init__("LPD_B&C")
        self.key_names.append("FUEL_DB")
        self.batches = []

    def finalize(self):
        for key in self.dictionary.keys():
            if "FUEL_DB" in key:
                batch = re.findall(r'(?<=\()([A-Za-z0-9_]+)(?:\.\d+)?(?=\))', key)
                self.batches.append(batch[0])

        sorted(self.batches)

    def print_block(self):
        string = super().print_block()
        for key in self.dictionary.keys():
            string += "\t{} = {}\n".format(key, self.dictionary[key])
        return string

class AstraSingleBlock(AstraBlock):

    def __init__(self, block_name, key_names):
        super(AstraSingleBlock, self).__init__(block_name)
        self.key_names = key_names

    def finalize(self):
        return

    def print_block(self):
        string = super().print_block()
        for key in self.dictionary.keys():
            string += "\t{} = {}\n".format(key, self.dictionary[key])
        return string

class AstraDirectoryBlock(AstraSingleBlock):

    def __init__(self, block_name, key_names, dir_key_names, main_directory):
        super(AstraDirectoryBlock, self).__init__(block_name, key_names)
        self.directory_keys = dir_key_names
        self.main_directory = main_directory

    def finalize(self):
        for key in self.dictionary.keys():
            value = self.dictionary[key]
            if key in self.directory_keys:
                self.dictionary[key] = os.path.abspath("{}{}".format(self.main_directory, value))
        return


class AstraJobBlock(AstraDirectoryBlock):

    def __init__(self, main_directory):
        keywords_in_order = ("CYCLE",
                             "PLANT",
                             "TABLE_SET",
                             "FORM_FUNCTION",
                             "GEOMETRY_FILE",
                             "RESTART_FILE",
                             "RESTART_STEP",
                             "GEOMETRY_FILE(1)",
                             "RESTART_FILE(1)",
                             "RESTART_STEP(1)",
                             "GEOMETRY_FILE(2)",
                             "RESTART_FILE(2)",
                             "RESTART_STEP(2)",
                             "GEOMETRY_FILE(3)",
                             "RESTART_FILE(3)",
                             "RESTART_STEP(3)",
                             "DATABASE_FILE",
                             "DATABASE_FUEL",
                             "TITLE")

        super(AstraJobBlock, self).__init__("JOB_TYP"
                                            ,
                                            keywords_in_order
                                            ,
                                            ("TABLE_SET",
                                             "FORM_FUNCTION",
                                             "GEOMETRY_FILE",
                                             "RESTART_FILE",
                                             "GEOMETRY_FILE(1)",
                                             "RESTART_FILE(1)",
                                             "GEOMETRY_FILE(2)",
                                             "RESTART_FILE(2)",
                                             "GEOMETRY_FILE(3)",
                                             "RESTART_FILE(3)",
                                             "DATABASE_FILE",)
                                            ,
                                            main_directory
                                            )
        self.print_order = keywords_in_order

    def print_block(self):
        string = self.delimiter + self.block_name + "\n"
        for key in self.print_order:
            try:
                string += "\t{} = {}\n".format(key, self.dictionary[key])
            except:
                pass
        return string


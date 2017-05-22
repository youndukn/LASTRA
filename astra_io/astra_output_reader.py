import re

from data.astra_output_block import AstraOutputBlock, AstraOutputCoreBlock, AstraOutputListBlock, AstraOutputNodeCoreBlock
from data.core import Core
from error import InputError
import numpy as np

class AstraOutputReader:

    b2dn_block = 0
    b2d_block = 1
    gadm_block = 2
    sum_block = 3
    happy_block = 4
    error_block = 5
    warn_block = 6
    enrc_block =7
    cbat_block = 8

    def __init__(self, output_name=None, output_string=None):
        """
        Define all blocks to be used in blocks
        :param output_name: output to parse
        :param output_string: output string to parse
        """
        # output string setting
        self.set_output_string(output_name, output_string)

        # output block to parse
        self.blocks = [
            AstraOutputNodeCoreBlock("B2DN", ["Y/X"]),
            AstraOutputCoreBlock("B2D", ["Y/X"]),
            AstraOutputCoreBlock("GADM", ["Y/X"]),
            AstraOutputListBlock("SUMMARY", ["---"]),
            AstraOutputBlock("HAPPY"),
            AstraOutputBlock("ERROR"),
            AstraOutputBlock("WARN"),
            AstraOutputListBlock("ENRC", ["---"]),
            AstraOutputListBlock("CBAT", ["---"])
        ]

    def set_output_string(self, output_name=None, output_string=None):
        """
        Replace output string with value
        :param block_name: block name to find
        :return: block string that was found
        """
        # Read Output and
        self.output_string = output_string
        if output_name:
            self.output_string = open(output_name, "r").read()

    def get_block_content(self, block_name):
        """
        Remove comments and find the block from ASTRA Input file
        :param block_name: block name to find
        :return: block string that was found
        """
        astra_block = self.find_astra_block(block_name)
        try:
            content = self.find_block_contents([astra_block], self.output_string)
        except InputError:
            return
        return content

    def parse_block_contents(self, block_name=None):
        """
        Parse the block from content into astra block and finalize
        :param block_content:content to parse from
        :param block_name: name to
        :return:astra_block with finalized data
        """
        # get c

        if block_name:
            block_contents = self.find_block_contents([self.find_astra_block(block_name)], self.output_string)
        else:
            block_contents = self.find_block_contents(self.blocks, self.output_string)

        for block_content in block_contents:
            astra_block = None
            for block in self.blocks:
                if re.match("\s*-" + format(block.block_name) + r'(\s+|:)', block_content):
                    astra_block = block
                    break

            # Separator String
            if len(astra_block.key_names) > 0:

                key_name_string = r'('
                for x in range(0, len(astra_block.key_names) - 1):
                    key_name_string += astra_block.key_names[x] + r'|'
                key_name_string += astra_block.key_names[len(astra_block.key_names) - 1]
                key_name_string += r')'

                splits = re.split(key_name_string, block_content)
                for x in range(0, len(splits) - 1):
                    is_key = False
                    for key_name in astra_block.key_names:
                        if key_name == splits[x]:
                            is_key = True
                            break

                    if is_key:
                        astra_block.increment_key_for_value(splits[x], splits[x + 1])
            else:
                astra_block.increment_key_for_value(astra_block.block_name, block_content)

        for block in self.blocks:
            block.finalize()

        return

    def find_block_contents(self, blocks, output_string):
        """
        Finds the last block from data with delimiter and block_name
        :param output_string:
        :param blocks:
        :param data: ASTRA Input file
        :param delimiter:
        :return: block_desire last block string
        """

        block_desire = ""
        start_block = False
        delimiter = "-"

        block_strings = []
        for line in str.splitlines(output_string):

            if re.match(r'\s' + delimiter + r'\w+', line) and start_block:
                block_strings.append(block_desire)
                block_desire = ""
                start_block = False

            if not start_block:
                for block in blocks:
                    if re.match(r'\s*' + delimiter + "{}".format(block.block_name) + r'(\s+|:)', line):
                        start_block = True
                        break

            if start_block:
                block_desire += line + "\n"

        return block_strings

    def find_astra_block(self, block_name):
        """
        Find AstraBlock from predefined
        :param block_name: name of the block
        :return:astra_block in self.blocks
        """
        astra_block = None
        for astra_block_i in self.blocks:
            if astra_block_i.block_name == block_name:
                astra_block = astra_block_i
                break
        if not astra_block:
            raise InputError(block_name + "Not Found")
        return astra_block

    def process_astra(self):

        self.parse_block_contents()

        if len(self.blocks[AstraOutputReader.sum_block].dictionary) == 0:
            return None, None, False

        if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
            return None, None, False

        if len(self.blocks[AstraOutputReader.warn_block].dictionary) > 0:
            return self.get_input_parameters(), self.get_output_parameters(), False

        return self.get_input_parameters(), self.get_output_parameters(), True

    def get_input_parameters(self):
        a_list = self.blocks[AstraOutputReader.cbat_block].lists[0]
        input_core = self.blocks[AstraOutputReader.b2dn_block].cores[0]
        for assemblies in input_core.assemblies:
            for assembly in assemblies:
                if 'B0' == assembly.get_batch():
                    assembly.add_value(3.049)
                if 'B1' == assembly.get_batch():
                    assembly.add_value(2.899)
                if 'B2' == assembly.get_batch():
                    assembly.add_value(2.805)
                if 'B3' == assembly.get_batch():
                    assembly.add_value(2.883)
                if 'C0' == assembly.get_batch():
                    assembly.add_value(3.407)
                if 'C1' == assembly.get_batch():
                    assembly.add_value(3.338)
                if 'C2' == assembly.get_batch():
                    assembly.add_value(3.314)
                if 'C3' == assembly.get_batch():
                    assembly.add_value(3.220)
                if 'D0' == assembly.get_batch():
                    assembly.add_value(4.342)
                if 'D1' == assembly.get_batch():
                    assembly.add_value(4.238)
                if 'D2' == assembly.get_batch():
                    assembly.add_value(4.203)

                for values in a_list:
                    if values[0] == assembly.get_batch():
                        assembly.add_value(values[3])

        return input_core

    def get_output_parameters(self):

        temp_list = []

        for a_list in self.blocks[AstraOutputReader.sum_block].lists:
            for values in a_list:
                if len(values) > 0:

                    for index, value in enumerate(values):
                        try:
                            float(value)
                            values[index]
                        except ValueError:
                            values[index] = 0

                    temp_list.append(values[0:14])

        a = np.array(temp_list, dtype=np.float64)

        return [a.max(axis=0)[1],
                a.max(axis=0)[3],
                a.max(axis=0)[7],
                a.max(axis=0)[8],
                a.max(axis=0)[9],
                a.max(axis=0)[10]]
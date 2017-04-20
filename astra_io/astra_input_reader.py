"""
1. Should Include the checker for the input error when it is first created
2. EDT_OPT to change to renovate
"""

import re

from data.astra_block import AstraBatchBlock, AstraShuffleBlock, AstraJobBlock
from error import InputError


class AstraInputReader:

    def __init__(self, input_name=None):
        """
        Define all blocks to be used in blocks
        :param input_name: ASTRA input name, cannot be None
        """

        self.blocks = [AstraShuffleBlock(),
                       AstraBatchBlock(),
                       AstraJobBlock()]

        self.input = input_name
        file = open(self.input, "r")
        self.input_string = file.read()

        file.close()

    def get_block_content(self, block_name):
        """
        Remove comments and find the block content from ASTRA Input file
        :param block_name: block name to find
        :return: block string that was found
        """
        astra_block = self.find_astra_block(block_name)
        content = self.remove_comments(self.input_string)
        try:
            content = self.find_block(content,
                                      astra_block)
        except InputError:
            return

        return content

    def parse_block_content(self, block_name):
        """
        Parse the block from content into astra block and finalize
        :param block_content:content to parse from
        :param block_name: name to
        :return:astra_block with finalized data
        """
        #get c
        block_content = self.get_block_content(block_name)

        astra_block = self.find_astra_block(block_name)
        key = None
        for line in str.splitlines(block_content):
            if astra_block.block_name is not line:
                split_line = line.split('=')
                length = len(split_line)
                if length > 0:
                    value = split_line[0]
                    if length == 2:
                        value = split_line[1]
                        key = split_line[0]
                    elif length == 1:
                        value = split_line[0]
                    elif length > 2:
                        raise InputError(
                            "Multi Keyword in one line not supported")
                    astra_block.add_key_for_value(key, value)

        astra_block.finalize()

        self.handle_relation(astra_block)

        return astra_block

    def replace_block(self, astra_blocks):
        """
        Replace first block with same block name
        :param astra_block:
        :return:
        """

        string = ""
        blocks = re.split(astra_blocks[0].delimiter, self.input_string)
        block_found = False
        for block in blocks:
            for astra_block in astra_blocks:
                if astra_block.block_name in block:
                    block_found = True
                    break

            if block_found:
                string += astra_block.print_block()
                block_found = False
            elif len(block) > 0:
                string += "%" + block

        return string

    def replace_block_to_name(self, astra_blocks, name):
        """
        Replace first block with same block name
        :param astra_block:
        :return:
        """
        file_writer = open(name, "w")
        file_writer.write(self.replace_block(astra_blocks))
        file_writer.close()

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

    def handle_relation(self, astra_block):
        """
        Handle all relationship between astra blocks
        :param astra_block: ASTRA block to set relation
        """
        if type(astra_block) == AstraBatchBlock:
            for astra_block_temp in self.blocks:
                if type(astra_block_temp) == AstraShuffleBlock:
                    astra_block_temp.core.set_batches(astra_block.batches)

    @staticmethod
    def find_block(data, astra_block):
        """
        Finds the last block from data with delimiter and block_name
        :param data: ASTRA Input file
        :param delimiter:
        :return: block_desire last block string
        """
        blocks = re.split(astra_block.delimiter, data)
        block_desire = None
        for block in blocks:
            if re.match(astra_block.block_name, block):
                block_desire = block

        if block_desire is None:
            raise InputError(block.block_name + ": Not found")

        return block_desire

    @staticmethod
    def remove_comments(data):
        """
        Remove all comments that starts with #
        :param data: content that has comments
        :return: content without comments
        """
        data = re.sub(r'#.*', "", data)
        return data

    @staticmethod
    def remove_branches(data):
        """
        Remove all comments that starts with #
        :param data: content that has comments
        :return: content without comments
        """
        a_string = ""
        for line in str.splitlines(data):
            if re.match("\s*/", line):
                break
            a_string += line
        return a_string

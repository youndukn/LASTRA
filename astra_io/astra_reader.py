import re
from data.astra_output_block import AstraOutputBlock, AstraOutputCoreBlock, AstraOutputListBlock, AstraOutputNodeCoreBlock
from data.core import Core
from error import InputError
import numpy as np

class AstraReader:



    def __init__(self, file_name="", file_content=""):
        """
        Define all blocks to be used in blocks
        :param file_name: file name to open
        :param file_content: output string to parse
        """

        # output string setting
        self._file_name = file_name
        self._file_content = file_content


    @property
    def file_name(self):
        return self._file_name

    @property
    def file_content(self):
        if not self._file_content or len(self._file_content) == 0:
            return open(self._file_name, "r").read()
        return self._file_content

    @file_name.setter
    def file_name(self, value):
        if not open(value, "r"):
            raise IOError("File {} does not exists".format(value))

        self._file_name = value
        file = open(value, "r")
        self._file_content = file.read()
        file.close()

    @file_content.setter
    def file_content(self, value):

        if self._file_name:
            raise IOError("Cannot set content if file_name exists")

        self._file_content = value


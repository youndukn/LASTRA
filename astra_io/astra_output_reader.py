import re

from data.astra_output_block import AstraOutputValueBlock, AstraOutputBlock, AstraOutputCoreBlock, AstraOutputListBlock, AstraOutputNodeCoreBlock
from data.core import Core
from error import InputError
import numpy as np

class AstraOutputReader:

    isParsed = False

    b2dn_block = 0
    b2d_block = 1
    gadm_block = 2
    sum_block = 3
    happy_block = 4
    error_block = 5
    warn_block = 6
    enrc_block =7
    cbat_block = 8
    input_block = 9
    abs_block = 10
    nfi_block = 11
    xss_block = 12
    p2d_block = 13
    abs_o_block = 14
    nfi_o_block = 15
    xss_o_block = 16
    f2d_block = 17
    f2d_o_block = 18
    iabs_block = 19
    infi_block = 20
    ixss_block = 21
    iabs_o_block = 22
    infi_o_block = 23
    ixss_o_block = 24
    iabsn_block = 25
    infin_block = 26
    ixssn_block = 27
    iabsn_o_block = 28
    infin_o_block = 29
    ixssn_o_block = 30
    peak_block = 31
    p2dn_block = 32
    p3dn_block = 33
    p3d_block = 34
    i3absn_block = 35
    i3nfin_block = 36
    i3xssn_block = 37
    i3absn_o_block = 38
    i3nfin_o_block = 39
    i3xssn_o_block = 40
    """
    isotope_start_block = 41
    isotope_end_block = isotope_start_block+20
    """
    def __init__(self, output_name=None, output_string=None):
        """
        Define all blocks to be used in blocks
        :param output_name: output to parse
        :param output_string: output string to parse
        """
        # output string setting
        self.set_output_string(output_name, output_string)

        # output block to parse 2D
        self.blocks = [
            AstraOutputNodeCoreBlock("B2DN", ["Y/X"]),
            AstraOutputCoreBlock("B2D", ["Y/X"]),
            AstraOutputCoreBlock("GADM", ["Y/X"]),
            AstraOutputListBlock("SUMMARY", ["---"]),
            AstraOutputBlock("HAPPY"),
            AstraOutputBlock("ERROR"),
            AstraOutputBlock("WARN"),
            AstraOutputListBlock("ENRC", ["---"]),
            AstraOutputListBlock("CBAT", ["---"]),
            AstraOutputBlock("INPUT"),
            AstraOutputCoreBlock("XSAB2D", ["Y/X"]),
            AstraOutputCoreBlock("XSNF2D", ["Y/X"]),
            AstraOutputCoreBlock("XSSC2D", ["Y/X"]),
            AstraOutputCoreBlock("P2D", ["Y/X"]),
            AstraOutputValueBlock("XSAB2D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("XSNF2D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("XSSC2D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputCoreBlock("F2D", ["Y/X"]),
            AstraOutputValueBlock("F2D", ["LINE  1: GROUP  1", "LINE  2: GROUP  2"]),
            AstraOutputCoreBlock("IXSAB2D", ["Y/X"]),
            AstraOutputCoreBlock("IXSNF2D", ["Y/X"]),
            AstraOutputCoreBlock("IXSSC2D", ["Y/X"]),
            AstraOutputValueBlock("IXSAB2D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("IXSNF2D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("IXSSC2D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputCoreBlock("IXSAB2DN", ["Y/X"]),
            AstraOutputCoreBlock("IXSNF2DN", ["Y/X"]),
            AstraOutputCoreBlock("IXSSC2DN", ["Y/X"]),
            AstraOutputValueBlock("IXSAB2DN", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("IXSNF2DN", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("IXSSC2DN", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputCoreBlock("PEAK", ["Y/X"]),
            AstraOutputCoreBlock("P2DN", ["Y/X"]),
            AstraOutputCoreBlock("P3DN", ["Y/X"]),
            AstraOutputCoreBlock("P3D", ["Y/X"]),
            AstraOutputCoreBlock("IXSAB3DN", ["Y/X"]),
            AstraOutputCoreBlock("IXSNF3DN", ["Y/X"]),
            AstraOutputCoreBlock("IXSSC3DN", ["Y/X"]),
            AstraOutputValueBlock("IXSAB3DN", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("IXSNF3DN", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("IXSSC3DN", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
        ]
        """
        AstraOutputCoreBlock("U2342D", ["Y/X"]),
        AstraOutputCoreBlock("U2352D", ["Y/X"]),
        AstraOutputCoreBlock("U2362D", ["Y/X"]),
        AstraOutputCoreBlock("NP372D", ["Y/X"]),
        AstraOutputCoreBlock("U2382D", ["Y/X"]),
        AstraOutputCoreBlock("NP392D", ["Y/X"]),
        AstraOutputCoreBlock("PU402D", ["Y/X"]),
        AstraOutputCoreBlock("PU412D", ["Y/X"]),
        AstraOutputCoreBlock("PU422D", ["Y/X"]),
        AstraOutputCoreBlock("AM432D", ["Y/X"]),
        AstraOutputCoreBlock("PM472D", ["Y/X"]),
        AstraOutputCoreBlock("PS482D", ["Y/X"]),
        AstraOutputCoreBlock("PM482D", ["Y/X"]),
        AstraOutputCoreBlock("PM492D", ["Y/X"]),
        AstraOutputCoreBlock("I1352D", ["Y/X"]),
        AstraOutputCoreBlock("XE452D", ["Y/X"]),
        AstraOutputCoreBlock("FP.12D", ["Y/X"]),
        AstraOutputCoreBlock("B-102D", ["Y/X"]),
        AstraOutputCoreBlock("H2O2D", ["Y/X"]),
        AstraOutputCoreBlock("DETE2D", ["Y/X"]),
        AstraOutputValueBlock("U2342D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("U2352D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("U2362D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("NP372D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("U2382D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("NP392D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("PU402D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("PU412D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("PU422D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("AM432D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("PM472D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("PS482D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("PM482D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("PM492D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("I1352D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("XE452D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("FP.12D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("B-102D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("H2O2D", ["DISTRIBUTION"]),
        AstraOutputValueBlock("DETE2D", ["DISTRIBUTION"])
        """
        """
        # output block to parse 3D
        self.blocks = [
            AstraOutputNodeCoreBlock("B2DN", ["Y/X"]),
            AstraOutputCoreBlock("B2D", ["Y/X"]),
            AstraOutputCoreBlock("GADM", ["Y/X"]),
            AstraOutputListBlock("SUMMARY", ["---"]),
            AstraOutputBlock("HAPPY"),
            AstraOutputBlock("ERROR"),
            AstraOutputBlock("WARN"),
            AstraOutputListBlock("ENRC", ["---"]),
            AstraOutputListBlock("CBAT", ["---"]),
            AstraOutputBlock("INPUT"),
            AstraOutputCoreBlock("XSAB3D", ["Y/X"]),
            AstraOutputCoreBlock("XSNF3D", ["Y/X"]),
            AstraOutputCoreBlock("XSSC3D", ["Y/X"]),
            AstraOutputCoreBlock("P3D", ["Y/X"]),
            AstraOutputValueBlock("XSAB3D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("XSNF3D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputValueBlock("XSSC3D", ["FIRST  LINE: GROUP 1", "SECOND LINE: GROUP 2"]),
            AstraOutputCoreBlock("F2D", ["Y/X"]),
            AstraOutputValueBlock("F2D", ["LINE  1: GROUP  1", "LINE  2: GROUP  2"]),
            AstraOutputCoreBlock("U2342D", ["Y/X"]),
            AstraOutputCoreBlock("U2352D", ["Y/X"]),
            AstraOutputCoreBlock("U2362D", ["Y/X"]),
            AstraOutputCoreBlock("NP372D", ["Y/X"]),
            AstraOutputCoreBlock("U2382D", ["Y/X"]),
            AstraOutputCoreBlock("NP392D", ["Y/X"]),
            AstraOutputCoreBlock("PU402D", ["Y/X"]),
            AstraOutputCoreBlock("PU412D", ["Y/X"]),
            AstraOutputCoreBlock("PU422D", ["Y/X"]),
            AstraOutputCoreBlock("AM432D", ["Y/X"]),
            AstraOutputCoreBlock("PM472D", ["Y/X"]),
            AstraOutputCoreBlock("PS482D", ["Y/X"]),
            AstraOutputCoreBlock("PM482D", ["Y/X"]),
            AstraOutputCoreBlock("PM492D", ["Y/X"]),
            AstraOutputCoreBlock("I1352D", ["Y/X"]),
            AstraOutputCoreBlock("XE452D", ["Y/X"]),
            AstraOutputCoreBlock("FP.12D", ["Y/X"]),
            AstraOutputCoreBlock("B-102D", ["Y/X"]),
            AstraOutputCoreBlock("H2O2D", ["Y/X"]),
            AstraOutputCoreBlock("DETE2D", ["Y/X"]),
            AstraOutputValueBlock("U2342D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("U2352D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("U2362D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("NP372D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("U2382D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("NP392D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("PU402D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("PU412D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("PU422D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("AM432D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("PM472D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("PS482D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("PM482D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("PM492D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("I1352D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("XE452D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("FP.12D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("B-102D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("H2O2D", ["DISTRIBUTION"]),
            AstraOutputValueBlock("DETE2D", ["DISTRIBUTION"])
        ]"""

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

        if self.isParsed:
            return True

        if block_name:
            block_contents = self.find_block_contents([self.find_astra_block(block_name)], self.output_string)
        else:
            block_contents = self.find_block_contents(self.blocks, self.output_string)

        boolean_summary = False

        for block_content in block_contents:

            astra_blocks = []

            for block in self.blocks:
                if re.match("\s*-" + format(block.block_name) + r'(\s+|:)', block_content):
                    astra_blocks.append(block)

            for astra_block in astra_blocks:
                if re.match("\s*-" + format("SUMMARY") + r'(\s+|:)', block_content):
                    boolean_summary = True
                    self.isParsed = True

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

        return boolean_summary

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

        summary_bool = self.parse_block_contents()

        if not summary_bool:
            if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
                #print(self.blocks[AstraOutputReader.warn_block].dictionary)
                return None, None, False
            return None, None, False

        if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
            #print(self.blocks[AstraOutputReader.warn_block].dictionary)
            return None, None, False

        if len(self.blocks[AstraOutputReader.warn_block].dictionary) > 0:
            #print(self.blocks[AstraOutputReader.warn_block].dictionary)
            return self.get_input_parameters(), self.get_output_parameters(), False

        return self.get_input_parameters(), self.get_output_parameters(), True

    def process_astra_cross_power(self):

        summary_bool = self.parse_block_contents()

        if not summary_bool:
            if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
                if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
                    value = str(self.blocks[AstraOutputReader.error_block].dictionary)
                    print(value[:100])
                if len(self.blocks[AstraOutputReader.warn_block].dictionary) > 0:
                    value = str(self.blocks[AstraOutputReader.warn_block].dictionary)
                    print(value[:100])
                return None, False, False
            return None, False, False

        if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
            value = str(self.blocks[AstraOutputReader.error_block].dictionary)
            print(value[:100])
            return None, False, False

        if len(self.blocks[AstraOutputReader.warn_block].dictionary) > 0:
            #value = str(self.blocks[AstraOutputReader.warn_block].dictionary)
            #print(value[:100])
            return self.get_cross_power_density_parameters(), False, True

        return self.get_cross_power_density_parameters(), True, True

    def process_astra_cross_power_3D(self):

        summary_bool = self.parse_block_contents()

        if not summary_bool:
            if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
                #print(self.blocks[AstraOutputReader.warn_block].dictionary)
                return None, False
            return None, False

        if len(self.blocks[AstraOutputReader.error_block].dictionary) > 0:
            value = str(self.blocks[AstraOutputReader.error_block].dictionary)
            print(value[:100])
            return None, False

        if len(self.blocks[AstraOutputReader.warn_block].dictionary) > 0:
            #print(self.blocks[AstraOutputReader.warn_block].dictionary)
            return self.get_cross_power_3D_parameters(), False

        return self.get_cross_power_3D_parameters(),True

    def get_input_parameters(self):

        return None

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

    def get_cross_power_density_parameters(self):

        summary = self.get_summary()
        summary = np.array(summary)

        burnup_len = len(self.blocks[AstraOutputReader.abs_block].cores)
        final_xs = []

        numb_plane = 26
        p_numb_plane = 28


        for i in range(burnup_len):
            densities = []
            """
            for j in range(self.isotope_start_block, self.isotope_end_block):
                density = self.blocks[j].cores[i]
                density_o1 = float(self.blocks[j+20].value_dict["DISTRIBUTION"][i].replace("(","").replace( " #/BARN-CM)", ""))
                setattr(density, "o1", density_o1)
                densities.append(density)
            """
            b2d_core = self.blocks[AstraOutputReader.b2d_block].cores[i]
            f2d_core = self.blocks[AstraOutputReader.f2d_block].cores[i]
            p2d_core = self.blocks[AstraOutputReader.p2d_block].cores[i]
            peak_core = self.blocks[AstraOutputReader.peak_block].cores[i]
            p2dn_core = self.blocks[AstraOutputReader.p2dn_block].cores[i]
            abs_core = self.blocks[AstraOutputReader.abs_block].cores[i]
            nfi_core = self.blocks[AstraOutputReader.nfi_block].cores[i]
            sc1_core = self.blocks[AstraOutputReader.xss_block].cores[i*2]
            sc2_core = self.blocks[AstraOutputReader.xss_block].cores[i*2+1]
            
            f2d_o1 = float(self.blocks[AstraOutputReader.f2d_o_block].
                           value_dict["LINE  1: GROUP  1"][i].replace("(","").replace( ")", ""))
            f2d_o2 = float(self.blocks[AstraOutputReader.f2d_o_block].
                           value_dict["LINE  2: GROUP  2"][i].replace("(", "").replace( ")", ""))
            abs_o1 = float(self.blocks[AstraOutputReader.abs_o_block].
                           value_dict["FIRST  LINE: GROUP 1"][i].replace("(", "").replace(")", ""))
            abs_o2 = float(self.blocks[AstraOutputReader.abs_o_block].
                           value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))
            nfi_o1 = float(self.blocks[AstraOutputReader.nfi_o_block].
                           value_dict["FIRST  LINE: GROUP 1"][i].replace("(", "").replace(")", ""))
            nfi_o2 = float(self.blocks[AstraOutputReader.nfi_o_block].
                           value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))
            sc1_o1 = float(self.blocks[AstraOutputReader.xss_o_block].
                           value_dict["FIRST  LINE: GROUP 1"][i*2].replace("(", "").replace(")", ""))
            sc1_o2 = float(self.blocks[AstraOutputReader.xss_o_block].
                           value_dict["SECOND LINE: GROUP 2"][i*2].replace("(", "").replace(")", ""))
            sc2_o1 = float(self.blocks[AstraOutputReader.xss_o_block].
                           value_dict["FIRST  LINE: GROUP 1"][i*2+1].replace("(", "").replace(")", ""))
            sc2_o2 = float(self.blocks[AstraOutputReader.xss_o_block].
                           value_dict["SECOND LINE: GROUP 2"][i*2+1].replace("(", "").replace(")", ""))

            setattr(f2d_core, "g1", f2d_o1)
            setattr(f2d_core, "g2", f2d_o2)
            setattr(abs_core, "g1", abs_o1)
            setattr(abs_core, "g2", abs_o2)
            setattr(nfi_core, "g1", nfi_o1)
            setattr(nfi_core, "g2", nfi_o2)
            setattr(sc1_core, "g1", sc1_o1)
            setattr(sc1_core, "g2", sc1_o2)
            setattr(sc2_core, "g1", sc2_o1)
            setattr(sc2_core, "g2", sc2_o2)

            iabs_core = self.blocks[AstraOutputReader.iabs_block].cores[i]
            infi_core = self.blocks[AstraOutputReader.infi_block].cores[i]
            isc1_core = self.blocks[AstraOutputReader.ixss_block].cores[i]

            iabs_o1 = float(self.blocks[AstraOutputReader.iabs_o_block].
                value_dict["FIRST  LINE: GROUP 1"][i].replace("(", "").replace(")", ""))
            iabs_o2 = float(self.blocks[AstraOutputReader.iabs_o_block].
                value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))
            infi_o1 = float(self.blocks[AstraOutputReader.infi_o_block].
                value_dict["FIRST  LINE: GROUP 1"][i].replace("(","").replace( ")", ""))
            infi_o2 = float(self.blocks[AstraOutputReader.infi_o_block].
                value_dict["SECOND LINE: GROUP 2"][i].replace("(","").replace(")", ""))
            isc1_o1 = float(self.blocks[AstraOutputReader.ixss_o_block].
                value_dict["FIRST  LINE: GROUP 1"][i].replace("(","").replace(")", ""))
            isc1_o2 = float(self.blocks[AstraOutputReader.ixss_o_block].
                value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))

            setattr(iabs_core, "g1", iabs_o1)
            setattr(iabs_core, "g2", iabs_o2)
            setattr(infi_core, "g1", infi_o1)
            setattr(infi_core, "g2", infi_o2)
            setattr(isc1_core, "g1", isc1_o1)
            setattr(isc1_core, "g2", isc1_o2)


            iabsn_core = self.blocks[AstraOutputReader.iabsn_block].cores[i]
            infin_core = self.blocks[AstraOutputReader.infin_block].cores[i]
            isc1n_core = self.blocks[AstraOutputReader.ixssn_block].cores[i]

            iabsn_o1 = float(self.blocks[AstraOutputReader.iabsn_o_block].
                value_dict["FIRST  LINE: GROUP 1"][i].replace("(", "").replace(")", ""))
            iabsn_o2 = float(self.blocks[AstraOutputReader.iabsn_o_block].
                value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))
            infin_o1 = float(self.blocks[AstraOutputReader.infin_o_block].
                value_dict["FIRST  LINE: GROUP 1"][i].replace("(","").replace( ")", ""))
            infin_o2 = float(self.blocks[AstraOutputReader.infin_o_block].
                value_dict["SECOND LINE: GROUP 2"][i].replace("(","").replace(")", ""))
            isc1n_o1 = float(self.blocks[AstraOutputReader.ixssn_o_block].
                value_dict["FIRST  LINE: GROUP 1"][i].replace("(","").replace(")", ""))
            isc1n_o2 = float(self.blocks[AstraOutputReader.ixssn_o_block].
                value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))

            setattr(iabsn_core, "g1", iabsn_o1)
            setattr(iabsn_core, "g2", iabsn_o2)
            setattr(infin_core, "g1", infin_o1)
            setattr(infin_core, "g2", infin_o2)
            setattr(isc1n_core, "g1", isc1n_o1)
            setattr(isc1n_core, "g2", isc1n_o2)

            p3dn_cores = []

            for j in range(numb_plane):
                p_index = i * numb_plane + j

                p3dn_core = self.blocks[AstraOutputReader.p3dn_block].cores[p_index]
                p3dn_cores.append(p3dn_core)

            p3d_cores = []

            for j in range(numb_plane):
                p_index = i * numb_plane + j

                p3d_core = self.blocks[AstraOutputReader.p3d_block].cores[p_index]
                p3d_cores.append(p3d_core)

            i3absn_cores = []
            i3nfin_cores = []
            i3sc1n_cores = []

            i3absn_o1 = float(self.blocks[AstraOutputReader.i3absn_o_block].
                              value_dict["FIRST  LINE: GROUP 1"][i].replace("(", "").replace(")", ""))
            i3absn_o2 = float(self.blocks[AstraOutputReader.i3absn_o_block].
                              value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))
            i3nfin_o1 = float(self.blocks[AstraOutputReader.i3nfin_o_block].
                              value_dict["FIRST  LINE: GROUP 1"][i].replace("(", "").replace(")", ""))
            i3nfin_o2 = float(self.blocks[AstraOutputReader.i3nfin_o_block].
                              value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))
            i3sc1n_o1 = float(self.blocks[AstraOutputReader.i3xssn_o_block].
                              value_dict["FIRST  LINE: GROUP 1"][i].replace("(", "").replace(")", ""))
            i3sc1n_o2 = float(self.blocks[AstraOutputReader.i3xssn_o_block].
                              value_dict["SECOND LINE: GROUP 2"][i].replace("(", "").replace(")", ""))

            for j in range(p_numb_plane):
                p_index = i * p_numb_plane + j
                i3absn_core = self.blocks[AstraOutputReader.i3absn_block].cores[p_index]
                i3nfin_core = self.blocks[AstraOutputReader.i3nfin_block].cores[p_index]
                i3sc1n_core = self.blocks[AstraOutputReader.i3xssn_block].cores[p_index]

                setattr(i3absn_core, "g1", i3absn_o1)
                setattr(i3absn_core, "g2", i3absn_o2)
                setattr(i3nfin_core, "g1", i3nfin_o1)
                setattr(i3nfin_core, "g2", i3nfin_o2)
                setattr(i3sc1n_core, "g1", i3sc1n_o1)
                setattr(i3sc1n_core, "g2", i3sc1n_o2)

                i3absn_cores.append(i3absn_core)
                i3nfin_cores.append(i3nfin_core)
                i3sc1n_cores.append(i3sc1n_core)

            final_xs.append(([summary[i, 1],
                              summary[i, 3],
                              summary[i, 6],
                              summary[i, 8],
                              summary[i, 9],
                              summary[i, 10],
                              summary[i, 11]],
                             abs_core,
                             nfi_core,
                             sc1_core,
                             sc2_core,
                             p2d_core,
                             f2d_core,
                             densities,
                             iabs_core,
                             infi_core,
                             isc1_core,
                             iabsn_core,
                             infin_core,
                             isc1n_core,
                             b2d_core,
                             peak_core,
                             p2dn_core,
                             p3dn_cores,
                             p3d_cores,
                             i3absn_cores,
                             i3nfin_cores,
                             i3sc1n_cores,
                             ))

        return final_xs


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
                a.max(axis=0)[8],
                a.max(axis=0)[9],
                a.max(axis=0)[10],
                a.max(axis=0)[11]]


    def get_summary(self):

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

        return a

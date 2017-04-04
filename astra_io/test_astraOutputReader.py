from unittest import TestCase
from astra_io.astra_output_reader import AstraOutputReader

class TestAstraOutputReader(TestCase):
    def test_get_block_content(self):
        output_reader = AstraOutputReader("outfile.out")
        #print(output_reader.get_block_content("PEAK")[1])
        #for block in output_reader.blocks:
        #    print(output_reader.get_block_content(block.block_name))
        self.assertTrue(True)

    def test_parse_block_contents(self):
        output_reader = AstraOutputReader("outfile.out")
        output_reader.parse_block_contents()
        print(output_reader.blocks[0].print_block())
        print("block",output_reader.blocks[3].print_block())
        self.assertTrue(True)

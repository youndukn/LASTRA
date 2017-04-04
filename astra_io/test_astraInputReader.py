from unittest import TestCase

from astra_io.astra_input_reader import AstraInputReader


class TestAstraInputReader(TestCase):
    input_reader = AstraInputReader("AST_INP")

    def test_get_block_content(self):
        #print(self.input_reader.get_block_content("LPD_SHF"))
        self.assertTrue(True)

    def test_parse_block_content(self):
        #print(self.input_reader.parse_block_content("LPD_SHF").print_block())
        self.assertTrue(True)

    def test_replace_block_to_name(self):

        from data.assembly import ShuffleAssembly, FreshAssembly

        for astra_block in self.input_reader.blocks:
            self.input_reader.parse_block_content(astra_block.block_name)

        #rotation shuffle assembly
        shuffle_assembly = ShuffleAssembly().copy(self.input_reader.blocks[0].core.assemblies[0][0])
        rotation = self.input_reader.blocks[0].core.assemblies[0][0].get_rotation()
        rotation += 1
        rotation %= 4
        self.input_reader.blocks[0].core.rotate([0, 0])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[0][0].get_rotation() == rotation)
        self.assertFalse(self.input_reader.blocks[0].core.assemblies[0][0] == shuffle_assembly)

        #rotation fresh assembly
        self.assertFalse(self.input_reader.blocks[0].core.rotate([0, 2]))

        #rotation assembly with symmetry
        shuffle_assembly = ShuffleAssembly()
        shuffle_assembly.copy(self.input_reader.blocks[0].core.assemblies[2][3])
        rotation1 = self.input_reader.blocks[0].core.assemblies[2][3].get_rotation()
        rotation2 = self.input_reader.blocks[0].core.assemblies[3][2].get_rotation()

        rotation1 += 1
        rotation1 %= 4

        rotation2 += 1
        rotation2 %= 4

        self.input_reader.blocks[0].core.rotate([2, 3])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][3].get_rotation() == rotation1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[3][2].get_rotation() == rotation2)

        # shuffling assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[2][2]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[3][3]

        self.input_reader.blocks[0].core.shuffle([2, 2], [3, 3])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][2] == fresh_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[3][3] == shuffle_assembly1)

        #shuffling assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[0][1]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[0][2]
        shuffle_assembly2 = self.input_reader.blocks[0].core.assemblies[1][0]
        fresh_assembly2 = self.input_reader.blocks[0].core.assemblies[2][0]

        self.input_reader.blocks[0].core.shuffle([0, 1], [0, 2])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[0][1] == fresh_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[0][2] == shuffle_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[1][0] == fresh_assembly2)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][0] == shuffle_assembly2)

        # shuffling assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[2][3]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[2][4]
        shuffle_assembly2 = self.input_reader.blocks[0].core.assemblies[3][2]
        fresh_assembly2 = self.input_reader.blocks[0].core.assemblies[4][2]

        self.input_reader.blocks[0].core.shuffle([2, 3], [2, 4])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][3] == fresh_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][4] == shuffle_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[3][2] == fresh_assembly2)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[4][2] == shuffle_assembly2)

        # shuffling assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[0][4]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[4][4]

        self.input_reader.blocks[0].core.shuffle([0, 4], [4, 4])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[0][4] == fresh_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[4][4] == shuffle_assembly1)

        self.assertFalse(self.input_reader.blocks[0].core.assemblies[4][0].get_rotation() == (shuffle_assembly1.get_rotation()+1)%4)

        # shuffling assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[2][2]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[2][0]

        self.input_reader.blocks[0].core.shuffle([2, 2], [2, 0])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][2] == fresh_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][0] == shuffle_assembly1)

        self.assertFalse(self.input_reader.blocks[0].core.assemblies[2][0].get_rotation() == (shuffle_assembly1.get_rotation()+1)%4)

        # shuffling to non assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[2][2]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[9][0]

        self.input_reader.blocks[0].core.shuffle([2, 2], [9, 0])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][2] == shuffle_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[9][0] == fresh_assembly1)

        # poison
        self.input_reader.blocks[0].core.poison([1, 1], False)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[1][1].get_batch() == "D0")
        self.input_reader.blocks[0].core.poison([1, 1], False)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[1][1].get_batch() == "D0")
        self.input_reader.blocks[0].core.poison([1, 1], True)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[1][1].get_batch() == "D1")
        self.input_reader.blocks[0].core.poison([1, 1], True)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[1][1].get_batch() == "D2")
        self.input_reader.blocks[0].core.poison([1, 1], True)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[1][1].get_batch() == "D2")

        # shuffling to non assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[2][1]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[3][2]
        changed = self.input_reader.blocks[0].core.shuffle([2, 1], [3, 2])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[2][1] == fresh_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[3][2] == shuffle_assembly1)

        # shuffling to non assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[0][2]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[0][0]
        changed = self.input_reader.blocks[0].core.shuffle([0, 2], [0, 0])
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[0][2] == fresh_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[0][0] == shuffle_assembly1)

        # shuffling to non assembly
        shuffle_assembly1 = self.input_reader.blocks[0].core.assemblies[3][3]
        fresh_assembly1 = self.input_reader.blocks[0].core.assemblies[3][1]
        changed = self.input_reader.blocks[0].core.shuffle([3, 3], [3, 1])
        print(self.input_reader.blocks[0].print_block())
        self.assertFalse(changed)
        print(self.input_reader.blocks[0].print_block())
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[3][3] == shuffle_assembly1)
        self.assertTrue(self.input_reader.blocks[0].core.assemblies[3][1] == fresh_assembly1)

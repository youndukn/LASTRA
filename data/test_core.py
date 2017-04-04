from unittest import TestCase


class TestCore(TestCase):
    def test_set_assembly_shuffle_at(self):

        from data.core import Core
        from data.assembly import Assembly

        assembly = Assembly()
        core = Core()
        core.set_shuffle_at(assembly, 1, 2)
        self.assertEqual(core.assemblies[1][2], assembly)

    def test_set_integer_shuffle_at(self):

        from data.core import Core
        from data.assembly import ShuffleAssembly

        core = Core()
        self.assertTrue(core.set_shuffle_at(ShuffleAssembly(), 1, 2))

    def test_set_out_of_bounds_shuffle_at(self):

        from data.core import Core
        from data.assembly import ShuffleAssembly

        core = Core()
        self.assertRaises(IndexError,  core.set_shuffle_at, ShuffleAssembly(), core.max_row+1, core.max_col+1)

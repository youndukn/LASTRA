from unittest import TestCase


class TestShuffleAssembly(TestCase):
    def test_rotate(self):
        from data.assembly import ShuffleAssembly

        shuffle = ShuffleAssembly()
        shuffle.rotation = 1
        shuffle.rotate()
        self.assertEqual(int(shuffle.rotation), 2)

    def test_string_none(self):
        from data.assembly import Assembly

        shuffle = Assembly()
        string_temp = ""
        if shuffle.print_assembly():
            string_temp += shuffle.print_assembly()
        self.assertEqual(string_temp, "")

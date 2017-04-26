from error import InputError


class Assembly:
    def __init__(self):
        self.__batch = ""
        self.__values = []

    def copy(self, assembly):
        self.__batch = assembly.get_batch()
        self.__values = assembly.get_values()
        return

    def set_batch(self, batch):
        self.__batch = batch

    def add_value(self, value):
        self.__values.append(value)

    def remove_all_values(self):
        self.__values.clear()

    def get_batch(self):
        return self.__batch

    def get_values(self):
        return self.__values

    def print_assembly(self):
        if self.__batch and self.__values:
            string = self.__batch + " ".join(" "+e+" " for e in self.__values)
            return string
        return None


class WaterAssembly(Assembly):
    def __init__(self):
        super(WaterAssembly, self).__init__()
        self.__type = "water"

    def print_assembly(self):
        return None


class FreshAssembly(Assembly):
    def __init__(self):
        super(FreshAssembly, self).__init__()
        self.__name = "F"
        self.__batch = ""
        self.__number = -1

    def copy(self, assembly):
        self.__batch = assembly.get_batch()
        self.__number = assembly.get_number()
        return

    def set_batch(self, batch):
        self.__batch = str(batch)

    def get_batch(self):
        return self.__batch

    def get_number(self):
        return self.__number

    def set_number(self, number):
        self.__number = int(number)

    def print_assembly(self):
        return \
            str(self.__name).ljust(3, ' ') + \
            str(self.__batch).ljust(6, ' ') + \
            str(self.__number).ljust(1, ' ') + \
            ','

class ShuffleAssembly(Assembly):
    def __init__(self):
        super(ShuffleAssembly, self).__init__()
        self.__rotation = -1
        self.__col = 'A'
        self.__row = -1
        self.__cycle = -1

    def copy(self, assembly):
        self.__rotation = assembly.get_rotation()
        self.__col = assembly.get_col()
        self.__row = assembly.get_row()
        self.__cycle = assembly.get_cycle()
        return

    def set_rotation(self, value):
        self.__rotation = int(value)

    def set_row(self, value):
        self.__row = int(value)

    def set_col(self, value):
        self.__col = value

    def set_cycle(self, value):
        self.__cycle = int(value)

    def get_rotation(self):
        return self.__rotation

    def get_row(self):
        return self.__row

    def get_col(self):
        return self.__col

    def get_cycle(self):
        return self.__cycle

    def rotate(self):
        if self.__rotation >= 0:
            self.__rotation = (self.__rotation + 1) % 4
        else:
            raise ValueError("Rotation undefined")

    def print_assembly(self):
        return \
            str(self.__cycle).ljust(3, ' ') + \
            str(self.__col).ljust(3, ' ') + \
            str(self.__row).ljust(3, ' ') + \
            str(self.__rotation).ljust(1, ' ') + \
            ','

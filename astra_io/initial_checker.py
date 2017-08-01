from astra import Astra
from astra_io.astra_input_reader import AstraInputReader
from astra_io.astra_output_reader import AstraOutputReader
from data.astra_train_set import AstraTrainSet
from error import InputError

class InitialChecker():
    def __init__(self, input_name, directory = "./"):
        self.input_name = input_name

        self.input_reader = AstraInputReader(self.input_name)

        for astra_block in self.input_reader.blocks:
            self.input_reader.parse_block_content(astra_block.block_name)

        self.astra = Astra(self.input_reader)

        self.astra.set_working_directory(directory)

    def get_proccessed_astra(self):

        self.astra.reset()

        output_string = self.astra.run_astra(self.input_reader.blocks[AstraInputReader.shuff_block])

        if not output_string:
            raise InputError("Initial Run", "Initial LP has an error")

        #Read output from running the initial input
        reading_out = AstraOutputReader(output_string=output_string)

        #Set default Starting Paramter
        core, lists, successful = reading_out.process_astra()

        if not successful:
            raise InputError("Initial Run", "Initial LP has an error")

        self.astra.set_initial_train_set(AstraTrainSet(core, None, lists, False, None))

        self.astra.reset()

        return self.astra

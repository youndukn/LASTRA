from astra import Astra
from astra_io.astra_input_reader import AstraInputReader
from astra_io.astra_output_reader import AstraOutputReader
from data.astra_train_set import AstraTrainSet

class InitialChecker():
    def __init__(self, input_name):
        self.input_name = input_name

        self.input_reader = AstraInputReader(self.input_name)

        for astra_block in self.input_reader.blocks:
            self.input_reader.parse_block_content(astra_block.block_name)

        self.astra = Astra(self.input_reader)

    def get_proccessed_astra(self):

        self.astra.reset()

        output_string = self.astra.run_astra(self.input_reader.blocks[AstraInputReader.shuff_block])

        if not output_string:
            return

        #Read output from running the initial input
        reading_out = AstraOutputReader(output_string=output_string)
        reading_out.parse_block_contents()

        # Set learning input parameter from output and save it as core in reader
        core = reading_out.get_input_parameters()
        self.input_reader.process_node_burnup_core(core)

        #Set default Starting Paramter
        core, lists, successful = reading_out.process_astra()
        self.astra.train_set = AstraTrainSet(core, [], list, None)

        self.astra.reset()

        return self.astra

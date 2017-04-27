import argparse
import sys
import pickle

from astra_io.astra_input_reader import AstraInputReader
from reward_calculator import RewardCalculator
from astra_io.astra_output_reader import AstraOutputReader
from astra import Astra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=0.0, help='helper')
    args = parser.parse_args()

    reading = AstraInputReader("01_s3c02p_nep_depl.job")

    for astra_block in reading.blocks:
        reading.parse_block_content(astra_block.block_name)

    astra = Astra(reading)

    astra.reset()

    output_string = astra.run_astra(reading.blocks[0])

    if not output_string:
        return

    reading_out = AstraOutputReader(output_string=output_string)
    reading_out.parse_block_contents()

    core = reading_out.get_input_parameters()
    reading.process_node_burnup_core(core)

    cal = RewardCalculator(reading)
    cal.calculate_rate()

    print(args.x)

if __name__ == '__main__':
    main()
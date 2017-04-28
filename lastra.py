import argparse

from astra_io.astra_input_reader import AstraInputReader
from reward_calculator import RewardCalculator
from astra_io.astra_output_reader import AstraOutputReader
from astra import Astra
from astra_io.initial_checker import InitialChecker
from reinforcement_learning import ReinforcementLearning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=0.0, help='helper')
    args = parser.parse_args()


    input_name = "01_s3c02p_nep_depl.job"
    thread_number = 12

    print("Reference Calculation")
    checker = InitialChecker(input_name)
    astra = checker.get_proccessed_astra()

    print("Reward Calculation")
    cal = RewardCalculator(thread_number, astra)
    dev = cal.calculate_rate()

    print("Reinforcement")
    learning = ReinforcementLearning(thread_number, astra, dev, None, None)
    learning.initial_population()

    print(args.x)

if __name__ == '__main__':
    main()
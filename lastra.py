import argparse

from reward_calculator import RewardCalculator
from astra_io.initial_checker import InitialChecker
from reinforcement_learning import ReinforcementLearning
from gui.main_window import MainWindow
import time

def main(input="test2.job", output="none", thread=13):


    #Main Window
    #main_window = MainWindow(thread)
    #main_window.mainloop()

    print("Reference Calculation")
    checker = InitialChecker(input)
    astra = checker.get_proccessed_astra()

    print("Reward Calculation")
    cal = RewardCalculator(thread, astra)
    dev = cal.dev_p
    maximum_rewards = cal.max
    #dev = cal.calculate_rate()

    print("Reinforcement")
    learning = ReinforcementLearning(thread, astra, maximum_rewards, None, None)
    learning.initial_population()

    print(args.i)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, default="01_s3c02p_nep_depl.job", help='helper')
    parser.add_argument('--o', type=str, default="none", help='helper')
    parser.add_argument('--t', type=float, default=12, help='helper')
    parser.add_argument('--p', type=float, default=0.0, help='helper')
    parser.add_argument('--d', type=float, default=0.0, help='helper')

    args = parser.parse_args()

    main()
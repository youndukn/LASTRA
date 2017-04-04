import argparse
import sys

from astra_io.astra_input_reader import AstraInputReader
from reward_calculator import RewardCalculator

reading = AstraInputReader("01_s3c02p_nep_depl.job")

astra_collector = []

#SHUFFLE
for astra_block in reading.blocks:
    reading.parse_block_content(astra_block.block_name)

cal = RewardCalculator(reading)
cal.calculate_rate()


#reading.replace_block_to_name(reading.blocks[0], "AST_INP")

#reading_out = AstraOutputReader("outfile.out")

#print(reading_out.get_block_content("P2DN"))

#reading_out.parse_block_content("P2DN")
#reading_out.parse_block_content("B2D")
#reading_out.parse_block_content("PEAK")

#reading_out.parse_block_contents()
#print(reading_out.blocks[0].print_block())
#print(reading_out.blocks[1].print_block())
#print(reading_out.blocks[2].print_block())

#input_string = open("01_s3c02p_nep_depl.job", 'rb')
#try :
#    output = subprocess.check_call(['astra.exe'], stdout= open("AST_OUT", 'wb'), stdin=open("AST_INP", 'rb'))
#except subprocess.CalledProcessError:
#    print("error")
#running_proc = Popen(['astra'], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
#file = open("outp", 'wb')
#print(running_proc.communicate(input=input_string.read())[0].decode('utf-8'))



#outfile = open("hello.out", "wb")
#outfile.write(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default=0.0, help='helper')
    args = parser.parse_args()
    print(args.x)

if __name__ == '__main__':
    main()
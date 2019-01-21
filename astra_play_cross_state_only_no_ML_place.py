import numpy as np
import threading, queue
import time

from astra_rd import AstraRD
import glob
import os
import enviroments
import pickle
import copy

from data.assembly import FreshAssembly, ShuffleAssembly

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import multiprocessing as mp

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 50000
# 환경 생성

ab55 = [
    [0, 9, 9],
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
    [8, 9, 17],
    [9, 9, 18],
    [11, 10, 10],
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [18, 10, 17],
    [19, 10, 18],
    [22, 11, 11],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [28, 11, 17],
    [29, 11, 18],
    [33, 12, 12],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [37, 12, 16],
    [38, 12, 17],
    [39, 12, 18],
    [44, 13, 13],
    [45, 13, 14],
    [46, 13, 15],
    [47, 13, 16],
    [48, 13, 17],
    [49, 13, 18],
    [55, 14, 14],
    [56, 14, 15],
    [57, 14, 16],
    [58, 14, 17],
    [59, 14, 18],
    [66, 15, 15],
    [67, 15, 16],
    [68, 15, 17],
    [69, 15, 18],
    [77, 16, 16],
    [78, 16, 17],
    [79, 16, 18],
    [88, 17, 17],
    [89, 17, 18],
    [99, 18, 18],
]

ab_si = [
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
    [11, 10, 10],
    [22, 11, 11],
    [33, 12, 12],
    [44, 13, 13],
    [55, 14, 14],
]

ab_in = [
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [45, 13, 14],
    [46, 13, 15],
]

cl_base = 17000
fxy_base = 1.55


# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self):

        # 쓰레드의 갯수
        self.threads = 7

    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):

        # self.load_model("./save_model/astra_a3c")
        # 쓰레드 수만큼 Agent 클래스 생성
        the_directory = "{}{}{}{}{}depl{}".format("/home/youndukn/Plants/1.4.0/",
                                                  "ucn5",
                                                  os.path.sep,
                                                  'c{0:02}'.format(13),
                                                  os.path.sep,
                                                  os.path.sep)

        agents = []
        for thread_id in range(self.threads):
            agents.append(Agent(the_directory, thread_id))

        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()
        """     
        fig, axs = plt.subplots(3, 5, figsize=(20, 15))

        lns = []
        for row in range(3):
            columns = []
            for column in range(2):
                columns.append(axs[row, column].plot([], [], 'bo'))
            lns.append(columns)
        plt.show(block=False)

        while True:
            item = my_queue.get()
            row = int(item[0] / 2) % 3
            axs[row, 0].imshow(np.array(item[2][:-2, :-2]))
            axs[row, 1].imshow(np.array(item[1][0][:-2, :-2]))
            axs[row, 1].set_title("{:3.2f} {:3.2f}".format(item[3], item[4]))
            fig.canvas.draw()
        """


# 액터러너 클래스(쓰레드)
class Agent(mp.Process):
    def __init__(self, input_directory, thread_id):
        mp.Process.__init__(self)

        self.thread_id = thread_id

        # create enviroment with directory
        self.input_directory = input_directory
        self.thread_id = thread_id
        directory = "{}{}{}".format(input_directory, os.path.sep, thread_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        input_name = glob.glob("{}01_*.inp".format(input_directory))

        self.file = open('/media/youndukn/lastra/running_data/{}_data_{}'.
                         format(input_name[0].replace(input_directory, ""), self.thread_id), 'wb')


    def my_input_output(self, values):
        burnup_boc = values[0]
        burnup_eoc = values[-1]

        selected_range = [4, 5, 6, 7]
        s_batch_init = np.array(burnup_boc.input_tensor_full)
        for index in selected_range:
            s_batch_init_selected = values[index].input_tensor_full
            s_batch_init_selected = np.array(s_batch_init_selected)
            s_batch_init = np.concatenate((s_batch_init, s_batch_init_selected), axis=2)

        my_state = s_batch_init[2:-2, 2:-2, :]

        my_output = np.array(burnup_boc.output_tensor)
        for index in selected_range:
            my_output = np.concatenate([my_output, values[index].output_tensor])

        o_batch_init = np.zeros((5, 10, 10))
        for indexes in ab55:
            o_batch_init[0][indexes[1] - 9][indexes[2] - 9] = burnup_boc.output_tensor[indexes[0]]
            o_batch_init[0][indexes[2] - 9][indexes[1] - 9] = burnup_boc.output_tensor[indexes[0]]

        for e_index, index in enumerate(selected_range):
            for indexes in ab55:
                o_batch_init[e_index + 1][indexes[1] - 9][indexes[2] - 9] = \
                    values[index].output_tensor[indexes[0]]
                o_batch_init[e_index + 1][indexes[2] - 9][indexes[1] - 9] = \
                    values[index].output_tensor[indexes[0]]

        my_cl = burnup_eoc.summary_tensor[0]

        my_fxy = 0

        for burnup_point in values:
            if my_fxy < burnup_point.summary_tensor[5]:
                my_fxy = burnup_point.summary_tensor[5]
        return my_state, my_output, my_cl, my_fxy, o_batch_init

    def convert_f_index(self, action):

        if action < len(ab_in) * len(ab_in):

            pre_pos = action // len(ab_in)
            next_pos = action % len(ab_in)
            position = ab_in[pre_pos][0]
            posibilities = ab_in[next_pos][0]

        else:
            changed_action = action - len(ab_in) * len(ab_in)

            pre_pos = changed_action // len(ab_si)
            next_pos = changed_action % len(ab_si)

            position = ab_si[pre_pos][0]
            posibilities = ab_si[next_pos][0]

        for index, value in enumerate(ab55):
            if value[0] == position:
                a_position = index
            if value[0] == posibilities:
                a_posibilities = index

        return a_position, a_posibilities

    def run(self):
        global episode

        step = 0
        self.T = 0

        directory = "{}{}{}".format(self.input_directory, os.path.sep, self.thread_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        input_name = glob.glob("{}01_*.inp".format(self.input_directory))
        self.env = AstraRD(input_name[0], reward_list_target=
        (16300, 0, 0, 0, 1.55, 0), main_directory=self.input_directory, working_directory=directory, thread_id = self.thread_id)

        while step < EPISODES:

            self.env.reset()

            s_t1, r_t, changed, info, succ_error, cross_set = self.env.step()

            if len(cross_set)>0:

                position_matrix = np.zeros((10, 10))

                core = self.env.get_last_core()
                for index1 in range(10):
                    for index2 in range(10):
                        if type(core.assemblies[index1][index2]) is FreshAssembly:
                            position_matrix[index1][index2] = 1.0
                        if type(core.assemblies[index1][index2]) is ShuffleAssembly:
                            position_matrix[index1][index2] = 0.5

                step += 1

                current_state, \
                current_output, \
                current_cl, \
                current_fxy, \
                current_output_matrix = \
                    self.my_input_output(cross_set)

                # 샘플을 저장

                dump_list = []
                for i, value in enumerate(cross_set):
                    a_list = [value.summary_tensor,
                              value.input_tensor_full,
                              value.output_tensor,
                              value.flux_tensor,
                              value.p2dn_tensor_full,
                              value.iinput_tensor_full,
                              value.iinputn_tensor_full,
                              value.binput_tensor_full,
                              value.batch_tensor_full,
                              value.peak_tensor_full,
                              value.fr_tensor_full,
                              value.p3dn_tensor_full,
                              value.p3d_tensor_full,
                              value.pp3d_tensor_full,
                              value.iinput3n_tensor_full]

                    dump_list.append(a_list)
                pickle.dump(dump_list, self.file, protocol=pickle.HIGHEST_PROTOCOL)

                print("|{:4d} |".format(self.thread_id),
                      "{:3.2f} |".format(current_cl),
                      "{:3.2f} |".format(current_fxy),
                      )

                    # 각 에피소드 당 학습 정보를 기록



if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()

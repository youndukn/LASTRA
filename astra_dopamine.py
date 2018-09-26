import tensorflow as tf
import numpy as np
import threading, queue
import time

from astra import Astra
import glob
import os
import enviroments
import pickle

from data.assembly import FreshAssembly

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

import multiprocessing as mp

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000
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
        # Enviroments f
        self.spaces = enviroments.Environment(action_main_shape=(400,), action_sub_shapes=(55,),
                                              observation_shape=(19, 19, 25))

        self.state_size = self.spaces.observation_space.shape
        self.action_size = self.spaces.action_space.shape[0]
        # A3C 하이퍼파라미터
        self.discount_factor = 0.70
        self.no_op_steps = 30
        self.actor_lr = 2.5e-5
        self.critic_lr = 2.5e-5
        # 쓰레드의 갯수
        self.threads = 6
        self.policies = []

        for index in range(9):
            policy_array = []
            for jnjex in range(self.spaces.action_space.shape[0]):
                policy_array.append(0)
            self.policies.append(policy_array)

        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic, self.n_critic = self.build_model()

        self.icm, self.icm2, self.r_in, self.icm_optimizer = self.build_icm_model((500,),
                                                                                  self.spaces.action_space.shape)

        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer(), self.icm_optimizer,
                          self.n_critic_optimizer()]

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()

        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/astra_a3c', self.sess.graph)

    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        my_queue = queue.Queue()

        self.load_model("./save_model/astra_a3c")
        # 쓰레드 수만큼 Agent 클래스 생성
        the_directory = "{}{}{}{}{}depl{}".format("/home/youndukn/Plants/1.4.0/",
                                                  "ygn3",
                                                  os.path.sep,
                                                  'c{0:02}'.format(10),
                                                  os.path.sep,
                                                  os.path.sep)

        agents = []
        for thread_id in range(self.threads):
            agents.append(Agent(self.action_size, self.state_size,
                                [self.actor, self.critic, self.n_critic], self.sess,
                                self.optimizer, self.discount_factor,
                                [self.summary_op, self.summary_placeholders,
                                 self.update_ops, self.summary_writer], the_directory, thread_id, self.icm, self.icm2,
                                self.r_in, my_queue))

        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()

        fig, axs = plt.subplots(3, 3, figsize=(20, 15))

        lns = []
        for row in range(3):
            columns = []
            for column in range(3):
                columns.append(axs[row, column].plot([], [], 'bo'))
            lns.append(columns)
        plt.show(block=False)

        last_summary_time = time.time()
        summary_interval = 1000

        while True:
            item = my_queue.get()
            row = int(item[0] / 2) % 3
            axs[row, 0].set_title("thread{:d}".format(item[0]))
            axs[row, 0].imshow(item[1][:8][:8])
            # axs[row, 1].imshow(item[10])
            # lns[row][2][0].set_xdata(range(0, self.spaces.action_space.shape[0]))
            # lns[row][2][0].set_ydata(item[2])
            # axs[row, 2].relim()
            # axs[row, 2].autoscale_view(True, True, True)

            # axs[row, 2].imshow(item[2])
            axs[row, 1].imshow(np.array(item[5][:8][:8]))
            # axs[row, 4].imshow(np.array(item[3][0]))
            # axs[row, 4].set_title("pre cl fxy {:3.2f} {:3.2f}".format(item[6], item[7]))
            axs[row, 2].imshow(np.array(item[4][0][:8][:8]))
            axs[row, 2].set_title("{:3.2f} {:3.2f}->{:3.2f} {:3.2f}".format(item[6], item[7], item[8], item[9]))
            fig.canvas.draw()
            now = time.time()
            if now - last_summary_time > summary_interval:
                self.save_model("./save_model/astra_a3c")
                last_summary_time = now



# 액터러너 클래스(쓰레드)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops, input_directory, thread_id, icm, icm2, r_in, my_queue):
        threading.Thread.__init__(self)

        # Action Space
        self.spaces = enviroments.Environment(action_main_shape=(400,), action_sub_shapes=(55,),
                                              observation_shape=(19, 19, 25))

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic, self.n_critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops
        self.icm = icm
        self.icm2 = icm2
        self.r_in = r_in
        self.thread_id = thread_id
        self.queue = my_queue

        self.target = (16300, 1.55)

        self.epsilon = 0.0
        self.epsilon_decay = 0.999

        self.pre_actions = np.zeros(self.action_size)

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards, self.next_states, self.outputs, self.next_outputs = [], [], [], [], [], []
        self.n_states, self.n_next_states, self.n_actions, self.n_rewards, self.n_outputs, self.n_next_outputs = [], [], [], [], [], []
        self.policies = []
        self.n_policies = []
        self.next_policies = []
        self.n_next_policies = []

        # 로컬 모델 생성
        self.local_actor = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # 모델 업데이트 주기
        self.t_max = 30
        self.t = 0

        self.T = 0

        # create enviroment with directory
        directory = "{}{}{}".format(input_directory, os.path.sep, thread_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        input_name = glob.glob("{}01_*.inp".format(input_directory))
        self.env = Astra(input_name[0], reward_list_target=
        (16300, 0, 0, 0, 1.55, 0), main_directory=input_directory, working_directory=directory)

        self.file = open('/media/youndukn/lastra/plants_data6/{}_data_{}'.
                         format(input_name[0].replace(input_directory, ""), self.thread_id), 'wb')

        self.file_trainable = open('/media/youndukn/lastra/trainable_data6/{}_data_{}'.
                                   format(input_name[0].replace(input_directory, ""), self.thread_id), 'wb')

    def my_input_output(self, values):
        burnup_boc = values[0]
        burnup_eoc = values[-1]
        """
        s_batch_init = burnup_boc.input_tensor_full
        s_batch_init_den = burnup_boc.density_tensor_full
        s_batch_init_den = np.array(s_batch_init_den)
        my_state = np.concatenate((s_batch_init, s_batch_init_den), axis=2)
        """
        selected_range = [4, 5, 6, 7]
        s_batch_init = np.array(burnup_boc.input_tensor_full)
        for index in selected_range:
            s_batch_init_selected = values[index].input_tensor_full
            s_batch_init_selected = np.array(s_batch_init_selected)
            s_batch_init = np.concatenate((s_batch_init, s_batch_init_selected), axis=2)

        my_state = s_batch_init

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

        while episode < EPISODES:
            done = False

            score = 0
            self.env.reset()
            self.pre_actions = np.zeros(self.action_size)

            start_state, start_output, start_cl, start_fxy, start_output_matrix = self.my_input_output(
                self.env.get_cross_set())

            history = np.reshape([start_state], (1,
                                                 self.spaces.observation_space.shape[0],
                                                 self.spaces.observation_space.shape[1],
                                                 self.spaces.observation_space.shape[2]))

            history_output = np.reshape([start_output], (1, 500))

            best_cl = start_cl
            best_fxy = start_fxy
            best_score = 0

            current_cl = start_cl
            current_fxy = start_fxy
            current_output = start_output
            current_output_matrix = start_output_matrix

            pre_cl = current_cl
            pre_fxy = current_fxy

            nonChanged = False

            while not done:

                step += 1

                action, policy, m_policy = self.get_action(history, history_output)

                policy_matrix = np.zeros((10, 10))
                m_policy_matrix = np.zeros((10, 10))
                max_policy_matrix = np.zeros((10, 10))
                next_max_policy_matrix = np.zeros((10, 10))

                max_value = 0

                for p_index, police in enumerate(policy):

                    position, posibilities = self.convert_f_index(p_index)

                    if posibilities == 1 or posibilities == 11:
                        max_value = 0

                    policy_matrix[ab55[position][1] - 9][ab55[position][2] - 9] += police
                    if ab55[position][2] != ab55[position][1]:
                        policy_matrix[ab55[position][2] - 9][ab55[position][1] - 9] += police

                    if police > max_value:
                        max_policy_matrix[ab55[position][1] - 9][ab55[position][2] - 9] = police
                        max_policy_matrix[ab55[position][2] - 9][ab55[position][1] - 9] = police
                        max_value = police

                max_value = 0

                for p_index, police in enumerate(m_policy):

                    position, posibilities = self.convert_f_index(p_index)

                    if posibilities == 0:
                        max_value = 0

                    if police > max_value:
                        m_policy_matrix[ab55[position][1] - 9][ab55[position][2] - 9] = police
                        m_policy_matrix[ab55[position][2] - 9][ab55[position][1] - 9] = police
                        max_value = police

                """
                max_policy_index = np.argmax(policy)

                max_policy_position = max_policy_index // (self.spaces.action_space.shapes[0])

                p_index = max_policy_position * (self.spaces.action_space.shapes[0])
                p_index_1 = p_index+(self.spaces.action_space.shapes[0])

                for n_dex, police in enumerate(m_policy[p_index:p_index_1]):
                    next_max_policy_matrix[ab55[n_dex][1] - 9][ab55[n_dex][2] - 9] = police
                    next_max_policy_matrix[ab55[n_dex][2] - 9][ab55[n_dex][1] - 9] = police
                """
                """
                posibilities = action % (self.spaces.action_space.shapes[0] + 3 + 2)
                position = action // (self.spaces.action_space.shapes[0] + 3 + 2)


                if posibilities < self.spaces.action_space.shapes[0]:
                    action_index = (position * self.spaces.action_space.shapes[0]) + posibilities
                    s_t1, r_t, changed, info, satisfied = self.env.step_shuffle(action_index, [0, 4])
                elif posibilities >= self.spaces.action_space.shapes[0] and posibilities < self.spaces.action_space.shapes[0] + 3:
                    action_index = (position * 3) + (posibilities - self.spaces.action_space.shapes[0])
                    s_t1, r_t, changed, info, satisfied = self.env.step_rotate(action_index, [0, 4])
                elif posibilities >= self.spaces.action_space.shapes[0] + 3 and posibilities < self.spaces.action_space.shapes[0] + 5:
                    action_index = (position * 2) + (posibilities - self.spaces.action_space.shapes[0] - 3)
                    s_t1, r_t, changed, info, satisfied = self.env.step_bp(action_index, [0, 4])
                """

                position, posibilities = self.convert_f_index(action)

                position_matrix = np.zeros((10, 10))

                core = self.env.get_last_core()
                for index1 in range(10):
                    for index2 in range(10):
                        if type(core.assemblies[index1][index2]) is FreshAssembly:
                            position_matrix[index1][index2] = 0.5

                if position_matrix[ab55[position][1] - 9][ab55[position][2] - 9] == 0.5:
                    position_matrix[ab55[position][1] - 9][ab55[position][2] - 9] = 1.5
                else:
                    position_matrix[ab55[position][1] - 9][ab55[position][2] - 9] = 1

                if position_matrix[ab55[posibilities][1] - 9][ab55[posibilities][2] - 9] == 0.5:
                    position_matrix[ab55[posibilities][1] - 9][ab55[posibilities][2] - 9] = 1.5
                else:
                    position_matrix[ab55[posibilities][1] - 9][ab55[posibilities][2] - 9] = 1

                action_index = (position * self.spaces.action_space.shapes[0]) + posibilities
                s_t1, r_t, changed, info, succ_error, satisfied, best, cross_set = self.env.step_shuffle(action_index,
                                                                                                         [0, 4])

                if not succ_error:
                    self.env.step_back()
                else:
                    done = not info

                    pre_cl = current_cl
                    pre_fxy = current_fxy

                    pre_output_matrix = current_output_matrix

                    current_state, \
                    current_output, \
                    current_cl, \
                    current_fxy, \
                    current_output_matrix = \
                        self.my_input_output(cross_set)

                    next_history = np.reshape([current_state], (1,
                                                                self.spaces.observation_space.shape[0],
                                                                self.spaces.observation_space.shape[1],
                                                                self.spaces.observation_space.shape[2]))

                    next_history_output = np.reshape([current_output], (1, 500))

                    reward = (10 * (min(self.target[0], current_cl) - min(self.target[0], pre_cl)) / cl_base + \
                              10 * (max(self.target[1], pre_fxy) - max(self.target[1], current_fxy)) / fxy_base) / 4
                    """
                    reward = (10 * (min(current_cl - self.target[0], 0)) / cl_base + \
                              10 * (min(self.target[1] - current_fxy, 0)) / fxy_base)/4
                    """
                    # r_in = get_reward_intrinsic(self.icm, self.r_in,  )
                    one_hot_action = np.zeros(self.spaces.action_space.shape)
                    one_hot_action[action] = 1
                    # r_in = self.icm2.predict([history_output, next_history_output, np.array([one_hot_action])])[0]
                    r_in = 0
                    # 정책의 최대값
                    self.avg_p_max += np.amax(self.actor.predict([history, history_output]))

                    non_clipped = reward

                    reward = reward + r_in

                    # score_addup
                    score += reward

                    reward = np.clip(reward, -1., 1.)

                    self.t += 1
                    if changed:

                        # 샘플을 저장
                        """if nonChanged:
                            self.append_sample(u_history,
                                               u_action,
                                               u_reward,
                                               u_next_history,
                                               u_history_output,
                                               u_next_history_output)
                        """

                        self.queue.put([self.thread_id,
                                        max_policy_matrix,
                                        policy,
                                        pre_output_matrix,
                                        current_output_matrix,
                                        position_matrix,
                                        pre_cl,
                                        pre_fxy,
                                        current_cl,
                                        current_fxy,
                                        policy_matrix])
                        self.T += 1

                        pickle.dump([history, action, reward, done], self.file_trainable,
                                    protocol=pickle.HIGHEST_PROTOCOL)

                        dump_list = []
                        for value in cross_set:
                            a_list = [value.summary_tensor,
                                      value.input_tensor_full,
                                      value.output_tensor,
                                      value.flux_tensor,
                                      value.density_tensor_full]
                            dump_list.append(a_list)

                        values3 = self.critic.predict([history, history_output])[0]
                        values4 = self.n_critic.predict([history, history_output])[0]

                        values = self.critic.predict([next_history, next_history_output])[0]
                        values2 = self.n_critic.predict([next_history, next_history_output])[0]

                        print("|{:4d} |".format(self.thread_id),
                              "{:4d} |".format(ab55[position][1] - 9),
                              "{:4d} |".format(ab55[position][2] - 9),
                              "{:4d} |".format(ab55[posibilities][1] - 9),
                              "{:4d} |".format(ab55[posibilities][2] - 9),
                              "{:3.2f} |".format(current_cl),
                              "{:3.2f} |".format(current_fxy),
                              "{:3.2f} |".format(non_clipped),
                              "{:3.2f} |".format(r_in),
                              "{:1.4f} |".format(values[0]),
                              "{:1.4f} |".format(values3[0]),
                              "{:1.4f} |".format(values2[0]),
                              "{:1.4f} |".format(values4[0]),
                              )

                        pickle.dump(dump_list, self.file, protocol=pickle.HIGHEST_PROTOCOL)
                        nonChanged = True

                    else:
                        nonChanged = False
                        u_history = history
                        u_action = action
                        u_reward = reward
                        u_next_history = next_history
                        u_history_output = history_output
                        u_next_history_output = next_history_output

                    if best_cl <= current_cl and best_fxy >= current_fxy:
                        next_policy = self.local_actor.predict([next_history, next_history_output])[0]

                        self.append_sample(history,
                                           action,
                                           reward,
                                           next_history,
                                           history_output,
                                           next_history_output,
                                           policy,
                                           next_policy)

                        best_cl = min(self.target[0], current_cl)
                        best_fxy = max(self.target[1], current_fxy)
                        best_score = score
                        history = next_history
                        history_output = next_history_output
                    else:

                        next_policy = self.local_actor.predict([next_history, next_history_output])[0]
                        self.append_sample(history,
                                           action,
                                           reward,
                                           next_history,
                                           history_output,
                                           next_history_output,
                                           policy,
                                           next_policy)
                        self.env.step_back()
                        current_cl = pre_cl
                        current_fxy = pre_fxy

                    # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                    if self.t >= self.t_max or done:
                        print("{}".format(self.thread_id), ' '.join('{:3d}'.format(np.argmax(k)) for k in self.actions))
                        print("{}".format(self.thread_id), ' '.join('{:3d}'.format(int(k * 100)) for k in self.rewards))
                        print("{}".format(self.thread_id),
                              ' '.join('{:3d}'.format(np.argmax(k)) for k in self.n_actions))
                        print("{}".format(self.thread_id),
                              ' '.join('{:3d}'.format(int(k * 100)) for k in self.n_rewards))
                        n_o1, n_o2, n_o3, n_o4, n_o5 = self.train_n_model(False)
                        if len(self.states) >= 1:
                            o1, o2, o3 = self.train_model(False)
                            self.optimizer[0](o1)
                            self.optimizer[1](o2)
                            self.optimizer[2](o3)

                        self.optimizer[0](n_o1)
                        self.optimizer[3](n_o2)
                        self.optimizer[0](n_o4)
                        self.optimizer[3](n_o5)
                        self.optimizer[2](n_o3)

                        self.n_states, self.n_actions, self.n_rewards, self.n_next_states, self.n_outputs, self.n_next_outputs = [], [], [], [], [], []
                        self.n_policies = []
                        self.n_next_policies = []

                        self.states, self.actions, self.rewards, self.next_states, self.outputs, self.next_outputs = [], [], [], [], [], []
                        self.policies = []
                        self.next_policies = []

                        self.update_local_model()
                        self.t = 0

                    if done:
                        # 각 에피소드 당 학습 정보를 기록
                        episode += 1

                        print(
                            "{:4d} |".format(self.thread_id),
                            "{:4d} |".format(self.T),
                            "{:4d} |".format(step),
                            "{:3.2f} |".format(start_cl),
                            "{:3.2f} |".format(start_fxy),
                            "{:3.2f} |".format(best_cl),
                            "{:3.2f} |".format(best_fxy),
                            "{:3.2f} |".format(best_score),
                            "{:3.2f} |".format(score),
                            "{:3.2f} |".format(self.epsilon),
                        )

                        stats = [score, self.avg_p_max / float(step), step]
                        for i in range(len(stats)):
                            self.sess.run(self.update_ops[i], feed_dict={
                                self.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = self.sess.run(self.summary_op)
                        self.summary_writer.add_summary(summary_str, episode + 1)
                        self.avg_p_max = 0
                        self.avg_loss = 0
                        step = 0
                        self.T = 0

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done, next_states, next_outputs, reverse=False):
        discounted_prediction = np.zeros_like(rewards)

        running_add = 0

        if not done:
            running_add = self.critic.predict(
                [np.float32(next_states[-1]),
                 np.float32(next_outputs[-1])]
            )[0]

        for t in reversed(range(0, len(rewards))):
            reward = rewards[t]
            if reverse:
                reward = reward * -1

            running_add = (running_add * self.discount_factor + reward)
            discounted_prediction[t] = running_add

        return discounted_prediction

    def n_discounted_prediction(self, rewards, done, next_states, next_outputs, reverse=False):
        discounted_prediction = np.zeros_like(rewards)

        for t in reversed(range(0, len(rewards))):

            running_add = self.critic.predict(
                [np.float32(next_states[t]),
                 np.float32(next_outputs[t])]
            )[0]

            reward = rewards[t]
            if reverse:
                reward = reward * -1

            running_add = (running_add * self.discount_factor + reward)
            discounted_prediction[t] = running_add

        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):

        states = np.zeros((len(self.states),
                           self.spaces.observation_space.shape[0],
                           self.spaces.observation_space.shape[1],
                           self.spaces.observation_space.shape[2]))

        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states)

        next_states = np.zeros((len(self.next_states),
                                self.spaces.observation_space.shape[0],
                                self.spaces.observation_space.shape[1],
                                self.spaces.observation_space.shape[2]))

        for i in range(len(self.next_states)):
            next_states[i] = self.next_states[i]

        next_states = np.float32(next_states)

        outputs = np.zeros((len(self.outputs), 500))

        for i in range(len(self.outputs)):
            outputs[i] = self.outputs[i]

        outputs = np.float32(outputs)

        next_outputs = np.zeros((len(self.next_outputs), 500))

        for i in range(len(self.next_outputs)):
            next_outputs[i] = self.next_outputs[i]

        next_outputs = np.float32(next_outputs)

        discounted_prediction = self.discounted_prediction(self.rewards, done, self.next_states, self.next_outputs)

        values = self.critic.predict([states, outputs])
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        print("{}".format(self.thread_id), "dis_r", ' '.join('{:1.2f}'.format(k) for k in discounted_prediction))
        print("{}".format(self.thread_id), "val_r", ' '.join('{:1.2f}'.format(k) for k in values))

        policy = self.actor.predict([states, outputs])
        old_policy = np.array(self.policies)
        action_prob = np.sum(np.array(self.actions) * policy, axis=1)
        old_action_prob = np.sum(np.array(self.actions) * old_policy, axis=1)
        cross_entropy = action_prob / (old_action_prob + 1e-10)
        log_cross_entropy = np.log(action_prob + 1e-10)
        entropy = np.sum(policy * np.log(policy / (old_policy + 1e-10)), axis=1)

        print("{}".format(self.thread_id), "x_ent    ", ' '.join('{:1.3f}'.format(k) for k in cross_entropy))
        print("{}".format(self.thread_id), "log_x_ent", ' '.join('{:1.3f}'.format(k) for k in log_cross_entropy))
        print("{}".format(self.thread_id), "ent      ", ' '.join('{:1.3f}'.format(k) for k in entropy))

        """
        if len(self.rewards) >2:
            reversed_discounted_prediction = self.discounted_prediction(self.rewards[:-1], done, self.states, self.outputs)

            reversed_values = self.critic.predict([next_states[1:], next_outputs[1:]])
            reversed_values = np.reshape(reversed_values, len(reversed_values))

            reversed_advantages = reversed_discounted_prediction - reversed_values

            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_discounted_prediction))
            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_values))

            self.optimizer[0]([next_states[1:], next_outputs[1:], np.array(self.actions[1:]), reversed_advantages])
            self.optimizer[1]([next_states[1:], next_outputs[1:], reversed_discounted_prediction])
        """
        """
        reversed_discounted_prediction = self.discounted_prediction(self.rewards, done, self.states, self.outputs, True)

        reversed_values = self.critic.predict([next_states, next_outputs])
        reversed_values = np.reshape(reversed_values, len(reversed_values))

        reversed_advantages = reversed_discounted_prediction - reversed_values

        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_discounted_prediction))
        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_values))
        self.optimizer[0]([next_states, next_outputs, np.array(self.actions), reversed_advantages])
        self.optimizer[1]([next_states, next_outputs, reversed_discounted_prediction])
        """

        """
        if len(states)>2:

            reversed_reward = []

            for reward in reversed(self.rewards[:-1]):
                reversed_reward.append(reward)

            reversed_discounted_prediction = self.discounted_prediction(reversed_reward, False,
                                                                        list(reversed(self.states[1:])),
                                                                        list(reversed(self.outputs[1:])))

            reversed_states = np.zeros((len(self.states) - 1,
                                        self.spaces.observation_space.shape[0],
                                        self.spaces.observation_space.shape[1],
                                        self.spaces.observation_space.shape[2]))

            for i, state in enumerate(list(reversed(self.states[1:]))):
                reversed_states[i] = state

            reversed_outputs = np.zeros((len(self.outputs)-1, 500))

            for i, output in enumerate(list(reversed(self.outputs[1:]))):
                reversed_outputs[i] = output

            reversed_outputs = np.float32(reversed_outputs)

            reversed_states = np.float32(reversed_states)

            reversed_values = self.critic.predict([reversed_states, reversed_outputs])
            reversed_values = np.reshape(reversed_values, len(reversed_values))

            reversed_advantages = reversed_discounted_prediction - reversed_values

            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_discounted_prediction))
            print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_values))

            self.optimizer[0]([reversed_states, reversed_outputs, np.array(list(reversed(self.actions[:-1]))), reversed_advantages])
            self.optimizer[1]([reversed_states, reversed_outputs, reversed_discounted_prediction])
        """

        # self.icm.train_on_batch([states, next_states, np.array(self.actions),dddd], np.zeros((length_state,)))

        return [states, outputs, np.array(self.actions), advantages, np.array(self.policies)], \
               [states, outputs, discounted_prediction], \
               [outputs, next_outputs, np.array(self.actions), np.array(discounted_prediction).reshape(-1, 1)]

    # 정책신경망과 가치신경망을 업데이트
    def train_n_model(self, done):

        states = np.zeros((len(self.n_states),
                           self.spaces.observation_space.shape[0],
                           self.spaces.observation_space.shape[1],
                           self.spaces.observation_space.shape[2]))

        for i in range(len(self.n_states)):
            states[i] = self.n_states[i]

        states = np.float32(states)

        next_states = np.zeros((len(self.n_next_states),
                                self.spaces.observation_space.shape[0],
                                self.spaces.observation_space.shape[1],
                                self.spaces.observation_space.shape[2]))

        for i in range(len(self.n_next_states)):
            next_states[i] = self.n_next_states[i]

        next_states = np.float32(next_states)

        outputs = np.zeros((len(self.n_outputs), 500))

        for i in range(len(self.n_outputs)):
            outputs[i] = self.n_outputs[i]

        outputs = np.float32(outputs)

        next_outputs = np.zeros((len(self.n_next_outputs), 500))

        for i in range(len(self.n_next_outputs)):
            next_outputs[i] = self.n_next_outputs[i]

        next_outputs = np.float32(next_outputs)

        discounted_prediction = self.n_discounted_prediction(self.n_rewards, done, self.n_next_states,
                                                             self.n_next_outputs)

        values = self.n_critic.predict([states, outputs])
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        print("{}".format(self.thread_id), "dis_n", ' '.join('{:1.2f}'.format(k) for k in discounted_prediction))
        print("{}".format(self.thread_id), "val_n", ' '.join('{:1.2f}'.format(k) for k in values))

        policy = self.actor.predict([states, outputs])
        old_policy = np.array(self.n_policies)
        action_prob = np.sum(np.array(self.n_actions) * policy, axis=1)
        old_action_prob = np.sum(np.array(self.n_actions) * old_policy, axis=1)
        cross_entropy = action_prob / (old_action_prob + 1e-10)
        log_cross_entropy = np.log(action_prob + 1e-10)
        entropy = np.sum(policy * np.log((policy / old_policy + 1e-10)), axis=1)

        print("{}".format(self.thread_id), "n_x_ent    ", ' '.join('{:1.3f}'.format(k) for k in cross_entropy))
        print("{}".format(self.thread_id), "n_log_x_ent", ' '.join('{:1.3f}'.format(k) for k in log_cross_entropy))
        print("{}".format(self.thread_id), "n_ent      ", ' '.join('{:1.3f}'.format(k) for k in entropy))

        reversed_discounted_prediction = self.n_discounted_prediction(self.n_rewards, done, self.n_states,
                                                                      self.n_outputs)

        reversed_values = self.n_critic.predict([next_states, next_outputs])
        reversed_values = np.reshape(reversed_values, len(reversed_values))

        reversed_advantages = reversed_discounted_prediction - reversed_values

        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_discounted_prediction))
        print("{}".format(self.thread_id), ' '.join('{:1.2f}'.format(k) for k in reversed_values))

        # self.icm.train_on_batch([states, next_states, np.array(self.actions),dddd], np.zeros((length_state,)))

        return [states, outputs, np.array(self.n_actions), advantages, np.array(self.n_policies)], \
               [states, outputs, discounted_prediction], \
               [outputs, next_outputs, np.array(self.n_actions), np.array(discounted_prediction).reshape(-1, 1)], \
               [next_states, next_outputs, np.array(self.n_actions), reversed_advantages,
                np.array(self.n_next_policies)], \
               [next_states, next_outputs, reversed_discounted_prediction]

    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, history, output):

        policy = self.local_actor.predict([history, output])[0]
        m_policy = policy

        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            action_index = np.random.choice(self.action_size, 1)[0]
        else:
            for action, action_number in enumerate(self.pre_actions):
                if 3 <= action_number:
                    value = m_policy[action]
                    distributed = value / self.action_size
                    m_policy[action] = 0
                    m_policy = np.add(m_policy, distributed)
                    print("V", self.thread_id, action)

            action_index = np.random.choice(self.action_size, 1, p=m_policy)[0]
            self.pre_actions[action_index] += 1
        return action_index, policy, m_policy

    # 샘플을 저장
    def append_sample(self, history, action, reward, next_history, output, next_output, policy, next_policy):

        self.n_states.append(history)
        self.n_next_states.append(next_history)

        self.n_outputs.append(output)
        self.n_next_outputs.append(next_output)

        self.n_policies.append(policy)
        self.n_next_policies.append(next_policy)

        act = np.zeros(self.action_size)
        act[action] = 1
        """
        posibilities = action % (self.spaces.action_space.shapes[0])
        position = action // (self.spaces.action_space.shapes[0])

        act[posibilities*self.spaces.action_space.shapes[0]+position] = 1
        """
        self.n_actions.append(act)
        self.n_rewards.append(reward)

        if reward >= 0:
            self.states.append(history)
            self.next_states.append(next_history)

            self.outputs.append(output)
            self.next_outputs.append(next_output)

            self.policies.append(policy)
            self.next_policies.append(next_policy)

            act = np.zeros(self.action_size)
            act[action] = 1
            """
            posibilities = action % (self.spaces.action_space.shapes[0])
            position = action // (self.spaces.action_space.shapes[0])

            act[posibilities*self.spaces.action_space.shapes[0]+position] = 1
            """
            self.actions.append(act)
            self.rewards.append(reward)


if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()

from keras.layers import Input, Add, Dense, BatchNormalization, Flatten, Conv2D,GlobalAveragePooling2D, \
                         Reshape, Multiply, LeakyReLU, Convolution2D, Concatenate, Lambda
from keras.models import Sequential, Model
from keras.layers.advanced_activations import ELU

from keras.initializers import glorot_uniform

from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np
import threading
import time

from astra import Astra
import glob
import os
import enviroments
import pickle

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000
# 환경 생성

cl_base = 17000
fxy_base = 1.55

def convolution_block_se_lk(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c', trainable=False)(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1', trainable=False)(X_shortcut)

    X = Add()([X_shortcut, X])
    X = LeakyReLU()(X)

    return X

def identity_block_se_lk(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2a', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2b', trainable=False)(X)
    X = LeakyReLU()(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis=3, name=bn_name_base + '2c', trainable=False)(X)

    se = GlobalAveragePooling2D(name='pool' + bn_name_base + '_gap')(X)
    se = Dense(F3 // 16, activation='relu', name = 'fc' + bn_name_base + '_sqz')(se)
    se = Dense(F3, activation='sigmoid', name = 'fc' + bn_name_base + '_exc')(se)
    se = Reshape([1, 1, F3])(se)
    X = Multiply(name='scale' + bn_name_base)([X, se])

    X = Add()([X, X_shortcut])
    X = LeakyReLU()(X)

    return X

def build_feature_map(input_shape, output_dim=288):
    input = Input(shape=input_shape)
    stage = 0;

    for i in range(5):
        stage_filters = [64, 64, 256]
        X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='fca_{}'.format(i),
                                    s=1)
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcb_{}'.format(i))
        X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='fcc_{}'.format(i))
        stage = stage + 1

    conv = Flatten()(X)
    fc = Dense(output_dim, name="feature", activation='relu')(conv)
    model = Model(inputs=input, outputs=fc)
    return model

def inverse_model(output_dim=3300):
    """
    s_t, s_t+1 -> a_t
    """
    def func(ft0, ft1):
        h = Concatenate()([ft0, ft1])
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='sigmoid')(h)
        return h
    return func

def forward_model(output_dim=288):
    """
    s_t, a_t -> s_t+1
    """
    def func(ft, at):
        h = Concatenate()([ft, at])
        h = Dense(256, activation='relu')(h)
        h = Dense(output_dim, activation='linear')(h)
        return h
    return func

def get_reward_intrinsic(model, intrinsic, x):
    return K.function([model.get_layer("state0").input,
                       model.get_layer("state1").input,
                       model.get_layer("action").input],
                      [intrinsic])(x)[0]

# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self):
        # 상태크기와 행동크기를 갖고옴
        self.spaces = enviroments.Environment(observation_shape=(19,19,40))

        self.state_size = self.spaces.observation_space.shape
        self.action_size = self.spaces.action_space.shape[0]
        # A3C 하이퍼파라미터
        self.discount_factor = 0.90
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        # 쓰레드의 갯수
        self.threads = 7

        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()

        self.icm, self.icm2, self.r_in, self.icm_optimizer= self.build_icm_model(self.spaces.observation_space.shape, self.spaces.action_space.shape)

        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer(), self.icm_optimizer]

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/astra_a3c', self.sess.graph)


    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):

        #self.load_model("./save_model/astra_a3c")
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
                  [self.actor, self.critic], self.sess,
                  self.optimizer, self.discount_factor,
                  [self.summary_op, self.summary_placeholders,
                   self.update_ops, self.summary_writer], the_directory, thread_id, self.icm,self.icm2, self.r_in))


        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()

        # 10분(600초)에 한번씩 모델을 저장
        while True:
            time.sleep(60 * 10)
            self.save_model("./save_model/astra_a3c")

    # 정책신경망과 가치신경망을 생성
    def build_model(self):
        input = Input(shape=self.state_size)
        stage = 0;

        for i in range(5):
            stage_filters = [64, 64, 256]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='ca_{}'.format(i),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cb_{}'.format(i))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}'.format(i))
            stage = stage + 1

        conv = Flatten()(X)
        fc = Dense(256, activation='relu')(conv)

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic


    def build_icm_model(self, state_shape, action_shape, lmd=1.0, beta=0.01):
        s_t0 = Input(shape=state_shape, name="state0")
        s_t1 = Input(shape=state_shape, name="state1")
        a_t = Input(shape=action_shape, name="action")
        fmap = build_feature_map(state_shape)
        f_t0 = fmap(s_t0)
        f_t1 = fmap(s_t1)
        act_hat = inverse_model()(f_t0, f_t1)
        f_t1_hat = forward_model()(f_t0, a_t)

        r_in = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1), name="intrinsic_reward")([f_t1, f_t1_hat])
        l_i = Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1))([a_t, act_hat])
        loss0 =Lambda(lambda x: beta * x[0] + (1.0 - beta) * x[1])([r_in, l_i])
        rwd = Input(shape=(1,))
        loss = Lambda(lambda x: (-lmd * x[0] + x[1]))([rwd, loss0])
        """
        r_in = merge([f_t1, f_t1_hat], mode=lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),
                     output_shape=(1,), name="reward_intrinsic")
        l_i = merge([a_t, act_hat], mode=lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1),
                    output_shape=(1,))
        loss0 = merge([r_in, l_i],
                      mode=lambda x: beta * x[0] + (1.0 - beta) * x[1],
                      output_shape=(1,))
        loss = merge([rwd, loss0],
                     mode=lambda x: (-lmd * x[0].T + x[1]).T,
                     output_shape=(1,))
        """
        model2 = Model([s_t0, s_t1, a_t], [r_in])
        model2._make_predict_function()
        model2.summary()

        model = Model([s_t0, s_t1, a_t, rwd], loss)
        model._make_predict_function()
        model.summary()

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(model.trainable_weights, [],loss)
        train = K.function([s_t0, s_t1, a_t, rwd],
                           [loss], updates=updates)

        return model, model2, r_in, train

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train


    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 액터러너 클래스(쓰레드)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops, input_directory, thread_id, icm,icm2, r_in):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops
        self.icm = icm
        self.icm2 = icm2
        self.r_in = r_in
        self.thread_id = thread_id

        self.target = (17100, 1.53)

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards, self.next_states = [], [], [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # 모델 업데이트 주기
        self.t_max = 20
        self.t = 0

        self.T = 0

        #Action Space
        self.spaces = enviroments.Environment(observation_shape=(19,19,40))

        #create enviroment with directory
        directory = "{}{}{}".format(input_directory, os.path.sep, thread_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        input_name = glob.glob("{}01_*.inp".format(input_directory))
        self.env = Astra(input_name[0], reward_list_target=
        (17100, 0, 0, 0, 1.53, 0), main_directory = input_directory, working_directory=directory)

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
        my_cl = burnup_eoc.summary_tensor[0]

        my_fxy = 0

        for burnup_point in values:
            if my_fxy < burnup_point.summary_tensor[5]:
                my_fxy = burnup_point.summary_tensor[5]
        return my_state, my_cl, my_fxy

    def run(self):
        global episode

        step = 0

        while episode < EPISODES:
            done = False

            score = 0
            self.env.reset()

            start_state, start_cl, start_fxy = self.my_input_output(self.env.cross_set)

            history = np.reshape([start_state], (1,
                                             self.spaces.observation_space.shape[0],
                                             self.spaces.observation_space.shape[1],
                                             self.spaces.observation_space.shape[2]))
            best_cl = start_cl
            best_fxy = start_fxy
            best_score = 0

            current_cl = start_cl
            current_fxy = start_fxy

            while not done:
                step += 1

                action, policy = self.get_action(history)

                posibilities = action % (self.spaces.action_space.shapes[0] + 3 + 2)
                position = int(action / (self.spaces.action_space.shapes[0] + 3 + 2))

                if posibilities < self.spaces.action_space.shapes[0]:
                    action_index = (position * self.spaces.action_space.shapes[0]) + posibilities
                    s_t1, r_t, changed, info, satisfied = self.env.step_shuffle(action_index, [0, 4])
                elif posibilities >= self.spaces.action_space.shapes[0] and posibilities < self.spaces.action_space.shapes[0] + 3:

                    action_index = (position * 3) + (posibilities - self.spaces.action_space.shapes[0])
                    s_t1, r_t, changed, info, satisfied = self.env.step_rotate(action_index, [0, 4])
                elif posibilities >= self.spaces.action_space.shapes[0] + 3 and posibilities < self.spaces.action_space.shapes[0] + 5:

                    action_index = (position * 2) + (posibilities - self.spaces.action_space.shapes[0] - 3)
                    s_t1, r_t, changed, info, satisfied = self.env.step_bp(action_index, [0, 4])

                done = not info

                pre_cl = current_cl
                pre_fxy = current_fxy

                current_state, current_cl, current_fxy = self.my_input_output(self.env.cross_set)

                next_history = np.reshape([current_state], (1,
                                                  self.spaces.observation_space.shape[0],
                                                  self.spaces.observation_space.shape[1],
                                                  self.spaces.observation_space.shape[2]))

                reward = 10 * (min(self.target[0], current_cl) - min(self.target[0], pre_cl)) / cl_base + \
                         10 * (max(self.target[1], pre_fxy) - max(self.target[1], current_fxy)) / fxy_base

                #r_in = get_reward_intrinsic(self.icm, self.r_in,  )
                one_hot_action = np.zeros(self.spaces.action_space.shape)
                one_hot_action[action_index] = 1
                r_in = self.icm2.predict([history, next_history, np.array([one_hot_action])])[0]
                # 정책의 최대값
                self.avg_p_max += np.amax(self.actor.predict(history))

                if done:
                    reward = -1

                reward = reward + r_in

                # score_addup
                score += reward

                if best_cl <= current_cl and best_fxy >= current_fxy:
                    best_cl = min(self.target[0], current_cl)
                    best_fxy = max(self.target[1], current_fxy)
                    best_score = score

                reward = np.clip(reward, -1., 1.)

                if changed:

                    self.t += 1
                    self.T += 1

                    # 샘플을 저장
                    self.append_sample(history, action, reward, next_history)

                    pickle.dump([history, action, reward, done], self.file_trainable, protocol=pickle.HIGHEST_PROTOCOL)

                    dump_list = []
                    for value in self.env.cross_set:
                        a_list = [value.summary_tensor,
                                  value.input_tensor_full,
                                  value.output_tensor,
                                  value.flux_tensor,
                                  value.density_tensor_full]
                        dump_list.append(a_list)
                    print("|{:4d} |".format(self.thread_id),
                          "{:4d} |".format(position),
                          "{:4d} |".format(posibilities),
                          "{:3.2f} |".format(current_cl),
                          "{:3.2f} |".format(current_fxy)
                          )
                    pickle.dump(dump_list, self.file, protocol=pickle.HIGHEST_PROTOCOL)


                history = next_history

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0

                if done:
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1
                    print("|{:>44d} |".format(episode),
                          "{:4d} |".format(self.T),
                          "{:4d} |".format(step),
                          "{:4d} |".format(self.thread_id),
                          "{:3.2f} |".format(start_cl),
                          "{:3.2f} |".format(start_fxy),
                          "{:3.2f} |".format(best_cl),
                          "{:3.2f} |".format(best_fxy),
                          "{:3.2f} |".format(best_score),
                          "{:3.2f} |".format(score),
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
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(
                self.states[-1]))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

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

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        self.optimizer[0]([states, np.array(self.actions), advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.optimizer[2]([states, next_states, np.array(self.actions),np.array(discounted_prediction).reshape(-1,1)])
        #self.icm.train_on_batch([states, next_states, np.array(self.actions),dddd], np.zeros((length_state,)))

        self.states, self.actions, self.rewards, self.next_states= [], [], [], []

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):
        input = Input(shape=self.state_size)
        stage = 0;

        for i in range(5):
            stage_filters = [64, 64, 256]
            X = convolution_block_se_lk(input, f=3, filters=stage_filters, stage=stage, block='ca_{}_{}'.format(i, self.thread_id),
                                        s=1)
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cb_{}_{}'.format(i, self.thread_id))
            X = identity_block_se_lk(X, 3, stage_filters, stage=stage, block='cc_{}_{}'.format(i, self.thread_id))
            stage = stage + 1

        conv = Flatten()(X)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic


    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, history):
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward, next_history):
        self.states.append(history)
        self.next_states.append(next_history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()

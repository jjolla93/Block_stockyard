import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from simulater import Environment as en
from view import ArrayView as av
import sys
import argparse
from collections import deque
import time
from scipy.interpolate import interp1d


class QCNN:
    def __init__(self, input_size, output_size, session, name):
        self.session=session
        self.input_size=input_size
        self.output_size=output_size
        self.net_name= name
        self._build_network()

    def _build_network(self, num_filter=16, h_size=128, l_rate=0.25e-3):
        with tf.variable_scope(self.net_name):
            self._X=tf.placeholder(tf.float32, [None, self.input_size[0]*self.input_size[1]], name="input_x")
            X=tf.reshape(self._X, [-1, self.input_size[0], self.input_size[1], 1])
            #dropout의 비율
            #keep_prob = tf.placeholder(tf.float32)

            with tf.name_scope("conv1") as scope:
                conv1 = tf.layers.conv2d(
                    inputs=X,
                    filters=num_filter,
                    kernel_size=[2, 2],
                    padding="same",
                    activation=tf.nn.relu)
                #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
                #pool1_flat = tf.reshape(conv1, [-1, self.input_size[0]*self.input_size[1]*num_filter])
                #dropout을 통하여 overfitting 감소
                #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            with tf.name_scope("conv2") as scope:
                num_filter2 = num_filter*2
                conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=num_filter2,
                    kernel_size=[2, 2],
                    padding="same",
                    activation=tf.nn.relu)
                #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
                flat_size = int(self.input_size[0] * self.input_size[1] * num_filter2)
                pool2_flat = tf.reshape(conv2, [-1, flat_size])

            with tf.name_scope("layer1") as scope:
                W1=tf.get_variable("W1", shape=[flat_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
                b1=tf.get_variable("b1", shape=[1, h_size], initializer=tf.contrib.layers.xavier_initializer())
                layer1=tf.nn.relu(tf.matmul(pool2_flat, W1)+b1)

            with tf.name_scope("layer1") as scope:
                W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.get_variable("b2", shape=[1, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
                self._Qpred = tf.matmul(layer1, W2)

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        self._loss = tf.reduce_sum(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size[0]*self.input_size[1]])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    return op_holder

def run_training():
    path = "../data/data.csv"
    env = en.Environment(path)

    prearrange = True
    reduced_action = False
    y = .99
    e = 0.005
    num_episodes = 1000000
    REPLAY_MEMORY = 100000
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    avg_rList = []
    avg_sList = []
    input_size=[env.width, env.height]
    output_size=env.size
    if reduced_action:
        output_size=int(output_size/4)
    replay_buffer = deque()
    with tf.Session() as sess:
        mainDQN = QCNN(input_size, output_size, sess, "main")
        targetDQN = QCNN(input_size, output_size, sess, "target")
        tf.global_variables_initializer().run()
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        t0 = time.time()
        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.reset()
            e = 1. / ((i / 50) + 1)
            rAll = 0
            d = False
            j = 0
            if i%1 == 0:
                prearranged = random.sample(range(input_size[0]*input_size[1]), 10)
            # The Q-Network
            while j < len(env.BLOCKS):
                if j< len(prearranged) and prearrange:
                    s, _, _ = env.step(prearranged[j])
                    j += 1
                    continue
                # state의 값은 -1~2 사이의 값이며 -1은 빈 공간 1은 투입블록의 잔여일
                s0=[item / env.BLOCKS[j].term if item!=-1.0 else item for item in s[0]]
                s0=[2.0 if item>2 else item for item in s0]
                Qs = mainDQN.predict(s0)

                # a0는 대략적인 배치 위치를 의미함
                a0 = np.argmax(Qs)
                # 임의 배치 위치는 빈 공간 중에서 선택함
                if np.random.rand(1) < e:
                    #a0 = random.randint(0, output_size - 1)

                    empties = []
                    for _i in range(len(s0)):
                        if s0[_i] == -1.0:
                            empties.append(_i)
                    ran = random.randint(0, len(empties)-1)
                    a0 = empties[ran]

                if reduced_action:
                    _i = int(a0/3)
                    _j = a0%3
                    a = [12*_i+2*_j, 12*_i+2*_j+1, 12*_i+2*_j+6, 12*_i+2*_j+7]
                    isfull = True
                    # 선택된 a0 공간내에서 실제 배치 위치를 결정함
                    for k in range(4):
                        if s0[0][a[k]] == 0:
                            a = a[k]
                            isfull = False
                            break
                    if isfull:
                        a=a[0]
                else:
                    a = a0
                #b = random.randint(0, 3)
                #a = a[b]
                # Get new state and reward from environment
                s1, r, d = env.step(a)
                # s1 = np.reshape(s1, (1, s1.size))
                if j < len(env.BLOCKS) - 1:
                    s2 = [item / env.BLOCKS[j+1].term if item!=-1.0 else item for item in s1[0]]
                    s2 = [2.0 if item > 2 else item for item in s2]

                # Save experience to buffer
                replay_buffer.append((s0, a0, r, s2, d))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()
                rAll += r
                s = s1
                j += 1
                if d == True:
                    # rAll=-1
                    break


            jList.append(j)
            rList.append(rAll)
            num_avg=1000
            if len(rList) > num_avg:
                avg_rList.append(sum(rList[(len(rList)-num_avg):]) / num_avg)
                avg_sList.append(sum(jList[(len(jList)-num_avg):]) / num_avg)
            print("Episode: {} steps: {} reward: {}".format(i, j, rAll))
            if i % 10 == 9:
                for _ in range(10):
                    # minibatch 생성
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                t1 = time.time()
                print("Seconds: %s" % (t1 - t0))
                t0 = t1
                sess.run(copy_ops)
        #bot_play(mainDQN, path)
    num_finish = 0
    for j in jList:
        if (j == len(env.BLOCKS)): num_finish += 1
    print("Percent of succesful episodes: " + str(100 * num_finish / num_episodes) + "%")
    # 결과 가시화
    #xList = list(range(len(rList)))
    #r2 = interp1d(xList, rList, kind='cubic')
    #j2 = interp1d(xList, jList, kind='cubic')
    #xnew = np.linspace(min(xList), max(xList), 500)
    #plt.plot(xnew, r2(xnew), label='reward')
    #plt.plot(xnew, j2(xnew), label='steps')
    plt.figure(figsize=(7, 4))
    plt.plot(avg_rList)
    plt.ylabel('Average reward')
    plt.xlabel('Episode')
    plt.savefig('../data/AvgR.png')
    plt.show()
    plt.figure(figsize=(7, 4))
    plt.plot(avg_sList)
    plt.ylabel('Average step')
    plt.xlabel('Episode')
    plt.savefig('../data/AvgS.png')
    plt.show()
    plt.figure(figsize=(7, 4))
    plt.plot(rList)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig('../data/R.png')
    plt.figure(figsize=(7, 4))
    plt.plot(jList)
    plt.ylabel('Step')
    plt.xlabel('Episode')
    plt.savefig('../data/S.png')
    if (len(env.LOGS) > 0):
        env.LOGS = sorted(env.LOGS, key=lambda log: log[-1])[::-1]
        av.visualize_log(env.LOGS, env.cumulate)


def bot_play(mainDQN, path):
    env = en.Environment(path)
    s=env.reset()
    reward_sum = 0
    while True:
        a = np.argmax(mainDQN.predict(s))
        s, reward, done= env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break


def replay_train(DQN, targetDQN, train_batch):
    x_stack=np.empty(0).reshape(0, DQN.input_size[0] * DQN.input_size[1])
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        dis=.99
        Q=DQN.predict(state)
        if(done):
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis*np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
    return DQN.update(x_stack, y_stack)

def simple_replay_train(DQN, train_batch):
    x_stack=np.empty(0).reshape(0, DQN.input_size[0] * DQN.input_size[1])
    y_stack = np.empty(0).reshape(0, DQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        dis=.99
        Q=DQN.predict(state)
        if(done):
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis*np.max(DQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
    return DQN.update(x_stack, y_stack)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run(main=main)
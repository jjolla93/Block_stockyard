import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from simulater import Environment as en
from view import ArrayView as av
import sys
import argparse
from collections import deque
import time

class QCNN:
    def __init__(self, input_size, output_size, session, name):
        self.session=session
        self.input_size=input_size
        self.output_size=output_size
        self.net_name= name
        self._build_network()

    def _build_network(self, h_size=64, l_rate=1e-3):
        with tf.variable_scope(self.net_name):
            self._X=tf.placeholder(tf.float32, [None, self.input_size[0]*self.input_size[1]], name="input_x")
            X=tf.reshape(self._X, [-1, self.input_size[0], self.input_size[1], 1])
            conv1 = slim.conv2d(
                inputs=X,
                num_outputs=32,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding="VALID")
            conv2 = slim.conv2d(
                inputs=conv1,
                num_outputs=h_size,
                kernel_size=[3, 3],
                stride=[1, 1],
                padding="VALID")

            self.streamAC, self.streamVC = tf.split(conv2, 2, 3)
            self.streamA = slim.flatten(self.streamAC)
            self.streamV = slim.flatten(self.streamVC)
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.AW = tf.Variable(xavier_init([h_size // 2, self.output_size]))
            self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
            #self.predictQ = tf.argmax(self.Qout, 1)

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self._Y = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, self.output_size, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        #self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        self.td_error = tf.square(self._Y - self.Q)
        self._loss = tf.reduce_mean(self.td_error)
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size[0]*self.input_size[1]])
        return self.session.run(self.Qout, feed_dict={self._X: x})

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
    path="C:\\Users\\BSKim\\PycharmProjects\\Stock_Reinforcement\\data\\space_test2.xml"
    env = en.Environment(path)

    y = .99
    e = 0.01
    num_episodes = 3000
    REPLAY_MEMORY = 100000
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    input_size=[env.width, env.height]
    output_size=env.size
    replay_buffer= deque()
    with tf.Session() as sess:
        mainDQN = QCNN(input_size, output_size, sess, "main")
        targetDQN = QCNN(input_size, output_size, sess, "target")

        tf.global_variables_initializer().run()
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        t0=time.time()
        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.reset()
            e=1./((i/50)+1)
            rAll = 0
            d = False
            j = 0
            # The Q-Network
            while j < len(env.BLOCKS):
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a = np.argmax(mainDQN.predict(s))
                if np.random.rand(1) <e:
                    a = random.randint(0, output_size-1)
                # Get new state and reward from environment
                s1, r, d = env.step(a)
                #Save experience to buffer
                replay_buffer.append((s, a, r, s1, d))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()
                rAll += r
                s = s1
                if d == True:
                    #rAll=-1
                    break
                j += 1

            jList.append(j)
            rList.append(rAll)
            print("Episode: {} steps: {} reward: {}".format(i, j, rAll))
            if i % 10 == 9:
                for _ in range(10):
                    # minibatch 생성
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                t1=time.time()
                print("Seconds: %s" %(t1-t0))
                t0=t1
                sess.run(copy_ops)
        bot_play(mainDQN, path)
    num_finish = 0
    for j in jList:
        if (j == len(env.BLOCKS)): num_finish += 1
    print("Percent of succesful episodes: " + str(100*num_finish / num_episodes) + "%")
    # av.visualize_space(env.LOGS[0])
    plt.plot(rList)
    plt.plot(jList)
    plt.show()
    if(len(env.LOGS)>=5):
        av.visualize_log(env.LOGS[-6:-1])


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
        dis=.9
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
        dis=.9
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
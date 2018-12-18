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

class QDNN:
    def __init__(self, input_size, output_size, session, name):
        self.session=session
        self.input_size=input_size
        self.output_size=output_size
        self.net_name= name
        self._build_network()
        logdir = "../logs"
        self.writer = tf.summary.FileWriter(logdir, session.graph)
        #self.writer.add_graph(session.graph)
        self.global_step=0

    def _build_network(self, h_size=256, l_rate=1e-3):
        with tf.variable_scope(self.net_name):
            self._X=tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            with tf.name_scope("layer1") as scope:
                W1=tf.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
                b1=tf.get_variable("b1", shape=[1, h_size], initializer=tf.contrib.layers.xavier_initializer())
                layer1=tf.nn.relu(tf.matmul(self._X, W1)+b1)
                #w1_hist = tf.summary.histogram("weights1", W1)
                #layer1_hist = tf.summary.histogram("layer1", layer1)

            with tf.name_scope("layer2") as scope:
                W2=tf.get_variable("W2", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
                b2=tf.get_variable("b2", shape=[1, h_size], initializer=tf.contrib.layers.xavier_initializer())
                layer2=tf.nn.relu(tf.matmul(layer1, W2)+b2)
                #w2_hist = tf.summary.histogram("weights2", W2)
                #layer2_hist = tf.summary.histogram("layer2", layer2)
            '''
            with tf.name_scope("layer3") as scope:
                W3=tf.get_variable("W3", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
                b3=tf.get_variable("b3", shape=[1, h_size], initializer=tf.contrib.layers.xavier_initializer())
                layer3=tf.nn.relu(tf.matmul(layer2, W3)+b3)
                #w3_hist = tf.summary.histogram("weights3", W3)
                #layer3_hist = tf.summary.histogram("layer3", layer3)
            
            with tf.name_scope("layer4") as scope:
                W4=tf.get_variable("W4", shape=[h_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
                b4=tf.get_variable("b4", shape=[1, h_size], initializer=tf.contrib.layers.xavier_initializer())
                layer4=tf.nn.relu(tf.matmul(layer3, W4)+b4)
                #w4_hist = tf.summary.histogram("weights4", W4)
                #layer4_hist = tf.summary.histogram("layer4", layer4)
            '''
            with tf.name_scope("layer5") as scope:
                W5 = tf.get_variable("W5", shape=[h_size, self.output_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b5 = tf.get_variable("b5", shape=[1, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
                self._Qpred = tf.matmul(layer2, W5)+b5
                #w5_hist = tf.summary.histogram("weights5", W5)
                #qpred_hist = tf.summary.histogram("qpred", self._Qpred)

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        #loss_sum = tf.summary.scalar("loss", self._loss)
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        #self.summary = tf.summary.merge_all()

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        loss, train = self.session.run([self._loss, self._train],
                                                feed_dict={self._X: x_stack, self._Y: y_stack})
        #loss, train, summary = self.session.run([self._loss, self._train, self.summary], feed_dict={self._X: x_stack, self._Y: y_stack})
        #self.writer.add_summary(summary, global_step=self.global_step)
        self.global_step+=1
        return loss, train

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    return op_holder

def run_training():
    path="../data/data.csv"

    env = en.Environment(path)

    y = .99
    e = 0.01
    num_episodes = 50000
    REPLAY_MEMORY = 100000
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    input_size=env.size
    output_size=env.size
    replay_buffer= deque()
    with tf.Session() as sess:
        mainDQN = QDNN(input_size, output_size, sess, "main")
        targetDQN = QDNN(input_size, output_size, sess, "target")
        tf.global_variables_initializer().run()
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        t0=time.time()
        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.reset()
            e=1./((i/50)+1)+0.007
            rAll = 0
            d = False
            j = 0
            # The Q-Network
            while j < len(env.BLOCKS):
                # Choose an action by greedily (with e chance of random action) from the Q-network
                s0 = [item/env.BLOCKS[j].term for item in s]
                Qs = mainDQN.predict(s0)
                #for k in range(len(s0[0])):
                    #if s0[0][k] != 0:
                        #Qs[0][k] = -100.0
                a = np.argmax(Qs)
                if np.random.rand(1) <e:
                    a = random.randint(0, input_size-1)
                # Get new state and reward from environment
                s1, r, d = env.step(a)
                #s1 = np.reshape(s1, (1, s1.size))
                if j<len(env.BLOCKS)-1:
                    s2 = [item/env.BLOCKS[j+1].term for item in s1]
                else:
                    s2 = s1
                #Save experience to buffer
                replay_buffer.append((s0, a, r, s2, d))
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
    plt.plot(rList, label='reward')
    plt.plot(jList, label='steps')
    plt.legend(bbox_to_anchor=(0.78, 0.98), loc=2, borderaxespad=0.)
    plt.show()
    if(len(env.LOGS)>0):
        env.LOGS=sorted(env.LOGS, key=lambda log: log[-1])[::-1]
        av.visualize_log(env.LOGS)

def bot_play(mainDQN, path):
    env = en.Environment(path)
    s=env.reset()
    reward_sum = 0
    j=0
    while True:
        a = np.argmax(mainDQN.predict(s))
        s, reward, done= env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break
        elif j==len(env.BLOCKS)-1:
            print("Total score: {}".format(reward_sum))
            break
        j+=1

def replay_train(DQN, targetDQN, train_batch):
    x_stack=np.empty(0).reshape(0, DQN.input_size)
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
    x_stack=np.empty(0).reshape(0, DQN.input_size)
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
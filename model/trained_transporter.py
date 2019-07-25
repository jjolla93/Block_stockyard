import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import numpy as np

from simulater.InNOutSpace import Space
from model.helper import *

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image.
def process_frame(frame):
    '''
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    '''
    s = frame.flatten()
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope):
        width, height = s_size[0], s_size[1]
        s_size = height * width
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, height, width, 1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=16,
                                     kernel_size=[2, 2], stride=[1, 1], padding='SAME')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[2, 2], stride=[1, 1], padding='SAME')
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)
            '''
            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                
                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
            '''

class Transporter():
    def __init__(self, sess, width, height, a_size=5, mode=0):
        self.width = width
        self.height = height
        s_size = (self.width, self.height)
        self.mode = mode
        self.name = 'tp'
        model_path = '../SavedModels/A3C/%d-%d-%d' % (self.width, self.height, mode)
        self.env = {}
        self.long_images = []
        self.long_counts = []
        self.num_move = 0

        with tf.device("/cpu:0"):
            #global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
            #trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
            master_network = AC_Network(s_size, a_size, 'global')  # Generate global network
            variables = slim.get_variables_to_restore()
            variables_to_restore = [v for v in variables if v.name.split('/')[0] == 'global']
            saver = tf.train.Saver(variables_to_restore)

        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        self.local_AC = master_network
        #self.update_local_ops = update_target_graph('global', self.name)
        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        self.sess = sess

    def get_block_moves(self, blocks, target, name):
        #self.env = Space(self.width, self.height, goal=self.mode, block_indices=blocks, target=target, allocation_mode=True)
        self.env[name] = Space(self.width, self.height, goal=self.mode, block_indices=blocks, target=target, allocation_mode=True)
        env = self.env[name]
        self.work(env)
        moves = self.env[name].block_moves
        blocks = self.env[name].blocks
        return moves, blocks

    def work(self, env):
        sess = self.sess
        with sess.as_default(), sess.graph.as_default():
            #sess.run(self.update_local_ops)
            #episode_buffer = []
            #episode_values = []
            episode_frames = []
            episode_reward = 0
            episode_step_count = 0
            d = False

            #self.env.new_episode()
            s = env.get_state()
            #s = self.env.get_state().screen_buffer

            s = process_frame(s)
            s2 = s.reshape([7, 5])
            episode_frames.append(s2)

            rnn_state = self.local_AC.state_init
            self.batch_rnn_state = rnn_state
            #while self.env.is_episode_finished() == False:


            while d == False:
                # Take an action using probabilities from policy network output.
                a_dist, v, rnn_state = sess.run(
                    [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                    feed_dict={self.local_AC.inputs: [s],
                               self.local_AC.state_in[0]: rnn_state[0],
                               self.local_AC.state_in[1]: rnn_state[1]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)
                #print(s.reshape([3, 4]))
                #print(a)
                s1, r, d = env.step(a)
                if d == False:
                    #s1 = self.env.get_state().screen_buffer
                    episode_frames.append(s1)
                    s1 = process_frame(s1)
                else:
                    s1 = s

                #episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                #episode_values.append(v[0, 0])

                episode_reward += r
                s = s1
                episode_step_count += 1
                if episode_step_count > 1000:
                    env.block_moves = 100
                    #print(str(s) + str(len(env.blocks)))
                    break

                if d == True and False:
                    images = np.array(episode_frames)
                    self.num_move += env.block_moves
                    if images.shape[1] != 3:
                        images = color_frame(images)
                    big_images = []
                    for image in images:
                        big_images.append(scipy.misc.imresize(image, [self.width*30, self.height*30], interp='nearest'))
                        self.long_images.append(scipy.misc.imresize(image, [self.width*30, self.height*30], interp='nearest'))
                        self.long_counts.append(self.num_move)

                # If the episode hasn't ended, but the experience buffer is full, then we
                # make an update step using that experience rollout.
                #if len(episode_buffer) == 30 and d != True :
                    #episode_buffer = []
                if d == True:
                    break

    def make_long_gif(self):
        time_per_step = 0.1
        #make_gif(self.long_images, '../frames/Alloc/%d-%d-%s/image' % (self.width, self.height, '30') + '_long.gif',
        #                         duration=len(self.long_images) * time_per_step, true_image=True, salience=False)
        make_gif_with_count(self.long_images, self.long_counts, '../frames/Alloc/%d-%d-%s/image' % (self.width, self.height, '30') + '_long.gif',
                                 duration=len(self.long_images) * time_per_step, true_image=True, salience=False)
        self.long_images = []
        self.num_move = 0
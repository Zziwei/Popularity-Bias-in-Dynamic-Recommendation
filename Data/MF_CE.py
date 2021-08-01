import tensorflow as tf
import time
import numpy as np
import utility
from tqdm import tqdm

np.random.seed(0)
tf.set_random_seed(0)


class MF_CE:

    def __init__(self, sess, train_df, num_user, num_item, epoch=100, lr=0.001, hidden=100, neg=5, bs=1024):
        self.sess = sess

        self.num_item = num_item
        self.num_user = num_user

        self.hidden = hidden
        self.neg = neg
        self.batch_size = bs
        self.train_df = train_df
        self.epoch = epoch
        self.lr = lr

        print('******************** MF_CE ********************')
        self._prepare_model()
        print('********************* MF_CE Initialization Done *********************')

    def run(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(1, self.epoch + 1):
            self.train_model(epoch_itr)

        R = self.sess.run(self.R)
        return R

    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_input")
            self.label_input = tf.placeholder(tf.float32, shape=[None, 1], name="label_input")

        with tf.variable_scope("MF_CE", reuse=tf.AUTO_REUSE):
            self.P = tf.get_variable(name="P",
                                     initializer=tf.truncated_normal(shape=[self.num_user,
                                                                            self.hidden],
                                                                     mean=0, stddev=0.03,
                                                                     dtype=tf.float32), dtype=tf.float32)
            self.Q = tf.get_variable(name="Q",
                                     initializer=tf.truncated_normal(shape=[self.num_item,
                                                                            self.hidden],
                                                                     mean=0, stddev=0.03,
                                                                     dtype=tf.float32), dtype=tf.float32)

        p = tf.reduce_sum(tf.nn.embedding_lookup(self.P, self.user_input), 1)  # batch_size x hidden_size

        q = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input), 1)  # batch_size x hidden_size

        self.predict = tf.sigmoid(tf.reduce_sum(p * q, 1, keepdims=True))
        self.predict = tf.clip_by_value(self.predict, clip_value_min=0.001, clip_value_max=0.999)

        self.cost = -tf.reduce_mean(tf.log(self.predict) * self.label_input + tf.log(1 - self.predict) * (1 - self.label_input))

        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimizer = optimizer.minimize(self.cost)

        self.R = tf.sigmoid(tf.matmul(self.P, self.Q, transpose_b=True))
        self.R = tf.clip_by_value(self.R, clip_value_min=0.001, clip_value_max=0.999)

    def train_model(self, itr):
        NS_start_time = time.time() * 1000.0
        epoch_cost = 0.
        user_list, item_list, label_list = utility.neg_sampling(self.num_user, self.num_item, self.train_df, self.neg)

        NS_end_time = time.time() * 1000.0

        print("negative Sampling time : %d ms" % (NS_end_time - NS_start_time))

        num_batch = int(len(user_list) / float(self.batch_size)) + 1
        random_idx = np.random.permutation(len(user_list))
        for i in tqdm(range(num_batch)):
            batch_idx = None
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
            _, tmp_cost = self.sess.run(
                [self.optimizer, self.cost],
                feed_dict={self.user_input: user_list[batch_idx, :],
                           self.item_input: item_list[batch_idx, :],
                           self.label_input: label_list[batch_idx, :]})
            epoch_cost += tmp_cost

        print("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
              "negative Sampling time : %d ms" % (NS_end_time - NS_start_time),
              "negative samples : %d" % len(user_list))


    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))


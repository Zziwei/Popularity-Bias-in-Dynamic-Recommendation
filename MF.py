import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.simplefilter('ignore')  # Ignores Warnings for nicer Plots. Disable for Debugging
import tensorflow as tf
import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from tqdm import tqdm
from scipy.sparse import coo_matrix


class MF:
    def __init__(self, lr, reg, hidden, neg, num_user, num_item):
        self.sess = tf.Session()

        self.num_item = num_item
        self.num_user = num_user

        self.lr = lr
        self.reg = reg
        self.hidden = hidden
        self.neg = neg

        self._prepare_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('*' * 30 + 'MF initialization done ' + '*' * 30)

    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_input")
            self.label_input = tf.placeholder(tf.float32, shape=[None, 1], name="label_input")
            self.prop_input = tf.placeholder(tf.float32, shape=[None, 1], name="prop_input")

        with tf.variable_scope("MF", reuse=tf.AUTO_REUSE):
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

        self.cost1 = -tf.reduce_mean(tf.log(self.predict) * self.label_input / self.prop_input
                                     + tf.log(1 - self.predict) * (1 - self.label_input / self.prop_input))

        self.cost2 = self.reg * 0.5 * (self.l2_norm(self.P) + self.l2_norm(self.Q))  # regularization term

        self.cost = self.cost1 + self.cost2  # the loss function

        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimizer = optimizer.minimize(self.cost)

    def train(self, train_array, neg_array, epoch, bs, verbose=False):
        epoch_cost = 0.
        epoch_cost1 = 0.
        epoch_cost2 = 0.
        user_list, item_list, label_list, prop_list = utility.negative_sampling_better(self.num_user, self.num_item,
                                                                                   train_array, neg_array, self.neg)

        prop_list = np.ones_like(prop_list) * np.mean(utility.position_bias[:20])

        num_batch = int(len(user_list) / float(bs) + 0.5)
        random_idx = np.random.permutation(len(user_list))
        if verbose:
            loop = tqdm(range(num_batch))
        else:
            loop = range(num_batch)
        for i in loop:
            batch_idx = None
            if i == num_batch - 1:
                batch_idx = random_idx[i * bs:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * bs):((i + 1) * bs)]
            _, tmp_cost, tmp_cost1, tmp_cost2 = self.sess.run(  # do the optimization by the minibatch
                [self.optimizer, self.cost, self.cost1, self.cost2],
                feed_dict={self.user_input: user_list[batch_idx, :],
                           self.item_input: item_list[batch_idx, :],
                           self.label_input: label_list[batch_idx, :],
                           self.prop_input: prop_list[batch_idx, :]})
            epoch_cost += tmp_cost
            epoch_cost1 += tmp_cost1
            epoch_cost2 += tmp_cost2

            if np.isnan(epoch_cost1):
                print('wtf')

        if verbose:
            print("Training //", "Epoch %d //" % epoch, " Total cost = {:.4f}".format(epoch_cost),
                  " Total cost1 = {:.4f}".format(epoch_cost1), " Total cost2 = {:.4f}".format(epoch_cost2))

    def record(self):
        self.P_array, self.Q_array = self.sess.run([self.P, self.Q])

    def predict_scores(self, uid):
        R = utility.sigmoid(np.matmul(self.P_array[uid, :], self.Q_array.T))
        R[np.where(R < 0.001)] = 0.001
        R[np.where(R > 0.999)] = 0.999
        return R

    def predict_all_scores(self):
        R = utility.sigmoid(np.matmul(self.P_array, self.Q_array.T))
        R[np.where(R < 0.001)] = 0.001
        R[np.where(R > 0.999)] = 0.999
        return R

    def reset(self):
        self.sess.run(self.P.initializer)
        self.sess.run(self.Q.initializer)

    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))

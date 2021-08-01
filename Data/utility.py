from math import log
import numpy as np
import pandas as pd
import copy
from operator import itemgetter
import time
import tqdm

from scipy import stats
from scipy.sparse import coo_matrix
from multiprocessing import Process, Queue, Pool, Manager
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import pickle

top1 = 5
top2 = 10
top3 = 15
top4 = 20
k_set = [top1, top2, top3, top4]

position_bias = 1. / np.log2(np.arange(5000) + 2)


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def neg_sampling(num_user, num_item, train_df, neg_rate):
    pos_user_array = train_df['uid'].values
    pos_item_array = train_df['iid'].values
    train_mat = coo_matrix((np.ones(len(train_df)),
                            (train_df['uid'].values, train_df['iid'].values)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array.reshape((-1, 1))
    user_neg = np.tile(pos_user_array, neg_rate).reshape((-1, 1))
    pos = pos_item_array.reshape((-1, 1))
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_user_array.shape[0]), replace=True).reshape((-1, 1))
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx, :]
    neg = neg[idx, :]
    pos_lable = np.ones(pos.shape)
    neg_lable = np.zeros(neg.shape)
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), np.concatenate([pos_lable, neg_lable], axis=0)







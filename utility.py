import numpy as np
import copy
from operator import itemgetter
from scipy.sparse import coo_matrix
from multiprocessing import Process, Queue, Pool, Manager

top1 = 5
top2 = 10
top3 = 15
top4 = 20
k_set = [top1, top2, top3, top4]

position_bias = 1. / np.log2(np.arange(5000) + 2)


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def negative_sampling_vanilla(num_user, num_item, train_array, neg_rate):
    pos_user_array = train_array[:, 0]
    pos_item_array = train_array[:, 1]
    pos_prop_array = position_bias[train_array[:, 2]]
    # pos_prop_array[np.where(pos_prop_array < 0.001)] = 0.001

    train_mat = coo_matrix((np.ones(train_array.shape[0]),
                            (pos_user_array, pos_item_array)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array.reshape((-1, 1))
    # user_neg = np.tile(pos_user_array, neg_rate).reshape((-1, 1))
    user_neg = np.random.choice(np.arange(num_user), size=(neg_rate * pos_user_array.shape[0]), replace=True).reshape((-1, 1))
    pos = pos_item_array.reshape((-1, 1))
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_user_array.shape[0]), replace=True).reshape((-1, 1))
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx, :]
    neg = neg[idx, :]
    pos_label = np.ones(pos.shape)
    neg_label = np.zeros(neg.shape)
    pos_prop = pos_prop_array.reshape((-1, 1))
    neg_prop = np.ones(neg.shape)
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0), np.concatenate([pos_prop, neg_prop], axis=0)


def negative_sampling_better(num_user, num_item, train_array, negative_array, neg_rate):
    pos_user_array = train_array[:, 0]
    pos_item_array = train_array[:, 1]
    # pos_prop_array = position_bias[train_array[:, 2]] * feedback_expose_prob
    pos_prop_array = position_bias[train_array[:, 2]]
    num_pos = len(pos_user_array)

    neg_user_array = negative_array[:, 0]
    neg_item_array = negative_array[:, 1]
    neg_prop_array = position_bias[negative_array[:, 2]]
    num_neg = len(neg_user_array)

    user_pos = pos_user_array.reshape((-1, 1))
    pos = pos_item_array.reshape((-1, 1))
    pos_label = np.ones(pos.shape)
    pos_prop = pos_prop_array.reshape((-1, 1))

    num_neg_sample = int(num_pos * neg_rate)
    neg_idx = np.random.choice(np.arange(num_neg), num_neg_sample, replace=True)
    user_neg = neg_user_array[neg_idx].reshape((-1, 1))
    neg = neg_item_array[neg_idx].reshape((-1, 1))
    neg_label = np.zeros(neg.shape)
    neg_prop = neg_prop_array[neg_idx].reshape((-1, 1))

    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), \
           np.concatenate([pos_label, neg_label], axis=0), np.concatenate([pos_prop, neg_prop], axis=0)

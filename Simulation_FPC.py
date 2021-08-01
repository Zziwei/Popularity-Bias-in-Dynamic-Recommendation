import tensorflow as tf
import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from operator import itemgetter
from tqdm import tqdm
from MF_withoutPB import MF_withoutPB
from scipy.sparse import coo_matrix
# import warnings; warnings.simplefilter('ignore') ##Ignores Warnings for nicer Plots. Disable for Debugging


class Simulation:
    def __init__(self, args, truth, truth_like, record=False):
        print(args)

        self.record = record
        self.data = args.data

        self.truth = truth
        self.truth_like = truth_like

        self.num_user = args.num_user
        self.num_item = args.num_item

        self.exp = args.exp
        self.cycle_itr = args.cycle_itr
        self.iteration = args.iteration
        self.K = args.K
        self.epoch = args.epoch
        self.lr = args.lr
        self.reg = args.reg
        self.hidden = args.hidden
        self.neg = args.neg

        self.feedback = []
        self.neg_feedback = []
        self.user_feedback = [[] for _ in range(self.num_user)]

        self.audience_size = np.sum(truth, axis=0)
        self.item_sorted = np.argsort(self.audience_size)

        self.truth_unclick = truth.copy()

        self.feedback_expose_prob = []

        self.FP_mat = np.ones((self.num_user, self.num_item))

    def initial_iterations(self):
        print('*' * 30 + ' Start initial random iterations ' + '*' * 30)

        init_click_mat = np.zeros((self.num_user, self.num_item))

        expose_count = np.ones(self.num_item)
        feedback_item_list = []
        for e in range(self.exp):
            print('-' * 10 + ' Iteration ' + str(e + 1) + ' ' + '-' * 10)
            for u in tqdm(range(self.num_user)):
                u_feedback = self.user_feedback[u]
                u_truth_like = self.truth_like[u]

                scores = np.random.random(self.num_item)
                scores[u_feedback] = -9999.
                topK_iid = np.argpartition(scores, -self.K)[-self.K:]
                topK_iid = topK_iid[np.argsort(scores[topK_iid])[-1::-1]]

                randomK = np.random.random(self.K)
                for k in range(self.K):
                    iid = topK_iid[k]
                    expose_count[iid] += 1.
                    if iid in u_truth_like and randomK[k] <= utility.position_bias[k]:
                        self.user_feedback[u].append(iid)
                        self.feedback.append([u, iid, k])
                        init_click_mat[u, iid] = 1.
                        self.truth_unclick[u, iid] = 0
                        feedback_item_list.append(iid)
                    else:
                        self.neg_feedback.append([u, iid, k])
                        self.FP_mat[u, iid] *= (1 - utility.position_bias[k])

        expose_prob = expose_count / (self.exp * self.num_user + 1)
        self.feedback_expose_prob += list(expose_prob[feedback_item_list])

        print('!' * 100)
        print('Generate ' + str(len(self.feedback)) + ' records.')
        print('!' * 100)

        if self.record:
            np.save('./Data/' + self.data + '/Experiment_FPC/init_click_mat.npy', init_click_mat)

        self.init_popularity = np.sum(init_click_mat, axis=0)
        return self.init_popularity

    def run_simulation(self):
        print('*' * 30 + ' Train MF until converge ' + '*' * 30)
        bs = int(len(self.feedback) * (self.neg + 1) / 50.)
        print('Update bs to ' + str(bs))
        mf = MF_withoutPB(self.lr, self.reg, self.hidden, self.neg, self.num_user, self.num_item)
        feedback_array = np.array(self.feedback)
        neg_feedback_array = np.array(self.neg_feedback)
        for j in range(self.epoch):
            mf.train(feedback_array, neg_feedback_array, j, bs, verbose=True)
            mf.record()

        print('*' * 30 + ' Start simulation ' + '*' * 30)

        itr_click_item = []
        itr_user = []
        itr_rec_item = []

        user_list = np.random.randint(self.num_user, size=self.iteration)
        user_count = np.zeros(self.num_user)
        item_click = np.zeros(self.num_item)

        last_time = time.time()
        itr = 0
        for c in range(int(self.iteration / self.cycle_itr)):
            for _ in tqdm(range(self.cycle_itr)):
                uid = user_list[itr]
                u_feedback = self.user_feedback[uid]
                u_truth_like = self.truth_like[uid]

                user_count[uid] += 1

                ''' do recommendation for the current user and calculate NDCG '''
                scores = mf.predict_scores(uid)

                ''' do False Positive Correction '''
                scores = 1. - (1. - scores) / self.FP_mat[uid, :]

                scores[u_feedback] = -9999.
                topK_iid = np.argpartition(scores, -self.K)[-self.K:]
                topK_iid = topK_iid[np.argsort(scores[topK_iid])[-1::-1]]
                itr_user.append(uid)
                itr_rec_item.append(topK_iid)

                ''' generate new feedback '''
                randomK = np.random.random(self.K)
                click_item = []

                for k in range(self.K):
                    iid = topK_iid[k]
                    if iid in u_truth_like and randomK[k] <= utility.position_bias[k]:  # have a click
                        self.user_feedback[uid].append(iid)
                        self.feedback.append([uid, iid, k])
                        click_item.append(iid)
                        item_click[iid] += 1
                        self.truth_unclick[uid, iid] = 0
                    else:
                        self.neg_feedback.append([uid, iid, k])
                        self.FP_mat[uid, iid] *= (1 - utility.position_bias[k])

                itr_click_item.append(click_item)
                itr += 1

            TPR = item_click / (self.audience_size - self.init_popularity)
            a = TPR[self.item_sorted]
            gc = np.sum(((np.arange(len(a)) + 1.) * 2 - len(a) - 1) * a) / (len(a) * np.sum(a))
            print('#' * 10
                  + ' The iteration %d, up to now total %d clicks, GC=%.4f, this cycle used %.2f s) '
                  % (itr, len(self.feedback), gc, time.time() - last_time)
                  + '#' * 10)

            last_time = time.time()

            ''' update the MF model '''
            bs = int(len(self.feedback) * (self.neg + 1) / 50.)
            print('Update bs to ' + str(bs))
            mf.reset()
            feedback_array = np.array(self.feedback)
            neg_feedback_array = np.array(self.neg_feedback)
            for j in tqdm(range(self.epoch)):
                mf.train(feedback_array, neg_feedback_array, j, bs, verbose=False)
            mf.record()
            print('')

        if self.record:
            np.save('./Data/' + self.data + '/Experiment_FPC/itr_user.npy', np.array(itr_user))
            np.save('./Data/' + self.data + '/Experiment_FPC/itr_rec_item.npy', np.array(itr_rec_item))
            np.save('./Data/' + self.data + '/Experiment_FPC/itr_click_item.npy', np.array(itr_click_item, dtype=object))

        return itr_click_item


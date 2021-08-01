import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.simplefilter('ignore')  # Ignores Warnings for nicer Plots. Disable for Debugging
import time
import numpy as np
import argparse
import utility
from Simulation_withoutPB import Simulation


parser = argparse.ArgumentParser(description='Experiment_withoutPB')
parser.add_argument('--run', type=int, default=10, help='number of experiments to run')
parser.add_argument('--iteration', type=int, default=40000, help='number of iterations to simulate')
parser.add_argument('--exp', type=int, default=1, help='number of initial random exposure iterations')
parser.add_argument('--cycle_itr', type=int, default=50, help='number of iterations in one cycle')
parser.add_argument('--epoch', type=int, default=15, help='number of epochs to train')
parser.add_argument('--K', type=int, default=20, help='number of items to recommend')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--reg', type=float, default=1e-5, help='regularization')
parser.add_argument('--hidden', type=int, default=100, help='latent dimension')
parser.add_argument('--neg', type=int, default=5, help='negative sampling rate')
parser.add_argument('--data', type=str, default='ml1m', help='path to eval in the Data folder')

args = parser.parse_args()

truth = np.load('./Data/' + args.data + '/truth.npy')
args.num_user = truth.shape[0]
args.num_item = truth.shape[1]
audience_size = np.sum(truth, axis=0)
item_sorted = np.argsort(audience_size)
truth_like = list(np.load('./Data/' + args.data + '/user_truth_like.npy', allow_pickle=True))

print('')
print('!' * 30 + ' Total truth ' + str(np.sum(truth)) + ' ' + '!' * 30)
print('')

itr_cumulated_click_count_list = []
itr_GC_TPR_list = []

for r in range(args.run):
    print('')
    print('#' * 100)
    print('#' * 100)
    print(' ' * 50 + ' Experiment run ' + str(r + 1) + ' ' * 50)
    print('#' * 100)
    print('#' * 100)
    simulation = Simulation(args, truth, truth_like)
    init_popularity = simulation.initial_iterations()
    itr_click_item = simulation.run_simulation()

    itr_cumulated_click_count = []
    itr_item_click = np.zeros((args.iteration, args.num_item))
    for itr in range(args.iteration):
        click_item = itr_click_item[itr]
        itr_item_click[itr, click_item] = 1.
        itr_cumulated_click_count.append(
            len(click_item) if itr == 0 else len(click_item) + itr_cumulated_click_count[-1])
    for itr in range(1, args.iteration):
        itr_item_click[itr, :] += itr_item_click[itr - 1, :]
    itr_item_click /= (audience_size - init_popularity).reshape((1, -1))

    itr_GC_TPR = []
    for itr in range(args.iteration):
        a = itr_item_click[itr, item_sorted]
        gc = np.sum(((np.arange(len(a)) + 1.) * 2 - len(a) - 1) * a) / (len(a) * np.sum(a))
        itr_GC_TPR.append(gc)

    itr_cumulated_click_count_list.append(itr_cumulated_click_count)
    itr_GC_TPR_list.append(itr_GC_TPR)

    itr_cumulated_click_count_mean = np.mean(itr_cumulated_click_count_list, axis=0)
    itr_cumulated_click_count_std = np.std(itr_cumulated_click_count_list, axis=0)
    itr_GC_TPR_mean = np.mean(itr_GC_TPR_list, axis=0)
    itr_GC_TPR_std = np.std(itr_GC_TPR_list, axis=0)

    # np.save('./Data/' + args.data + '/Experiment_withoutPB/itr_cumulated_click_count_mean.npy', np.array(itr_cumulated_click_count_mean))
    # np.save('./Data/' + args.data + '/Experiment_withoutPB/itr_cumulated_click_count_std.npy', np.array(itr_cumulated_click_count_std))
    # np.save('./Data/' + args.data + '/Experiment_withoutPB/itr_GC_TPR_mean.npy', np.array(itr_GC_TPR_mean))
    # np.save('./Data/' + args.data + '/Experiment_withoutPB/itr_GC_TPR_std.npy', np.array(itr_GC_TPR_std))



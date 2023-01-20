from synergy.single import Hill, Hill_CI, Hill_2P
from synergy.combination import MuSyC, BRAID, Zimmer, Loewe, Bliss, ZIP, HSA, Schindler, CombinationIndex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import argparse
from utils import *
import pystan
import os
import warnings
from scipy import stats
from sklearn.neighbors import KernelDensity


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='training')

# general configs
parser.add_argument('--save_output', type=str2bool, nargs='?', dest='save_output', const=True, default=True,
                    help="whether to save the output or not")
parser.add_argument('--path', type=str, default="", metavar='N',
                    help="folder to save the numpy arrays")
# deterministic models configs
parser.add_argument('--n_bootstrap', type=int, default=200, metavar='N', help="number of bootstrap operations in \
                    bootstrap-based methods such as MuSyC")
# our model configs
parser.add_argument('--n_iter', type=int, default=1000, metavar='N',
                    help='number of iterations for our model')
parser.add_argument('--n_warmup_iter', type=int, default=500, metavar='N',
                    help='number of warmup iterations for our model')
parser.add_argument('--adapt_delta', type=float, default=0.9, metavar='N', help="the target average proposal \
                    acceptance probability during the adaptation period in Stan training")

args, unknown = parser.parse_known_args()
print("args: ", args)
print("args not parsed in train: ", unknown)

# os.environ["CC"] = "gcc-4.9"
# # just to be sure
# os.environ["CXX"] = "g++-4.9"

# Define our model via Stan
synba = """
        data {
          int<lower=0> N;
          int<lower=0> N_test;
          vector[N] x_1;
          vector[N] x_2;
          vector[N] y;
          vector[N_test] x_1_test;
          vector[N_test] x_2_test;
        }
        parameters {
          real<lower=0> e_0;
          real<lower=0, upper=1> e_1;
          real<lower=0, upper=1> e_2;
          real<lower=0, upper=1> e_3;
          real<lower=log(1e-10), upper=log(1e6)> logC_1;
          real<lower=log(1e-10), upper=log(1e6)> logC_2;
          real<lower=0> h_1;
          real<lower=0> h_2;
          real<lower=0> alpha;
          real<lower=0> sigma;
        }
        model {
            e_0 ~ normal(100, 3);
            e_1 ~ beta(0.46, 0.58);
            e_2 ~ beta(0.46, 0.58);
            e_3 ~ beta(0.46, 0.58);
            h_1 ~ lognormal(0, 1);
            h_2 ~ lognormal(0, 1);
            alpha ~ lognormal(0, 1);
            sigma ~ lognormal(0, 1);
            for(i in 1:N){
                y[i] ~ normal( ( (exp(logC_1))^h_1 * (exp(logC_2))^h_2 * e_0 + x_1[i]^h_1 * (exp(logC_2))^h_2 * e_1 * e_0 + x_2[i]^h_2 * (exp(logC_1))^h_1 * e_2 * e_0 + alpha * x_1[i]^h_1 * x_2[i]^h_2 * e_3 * e_0) / ((exp(logC_1))^h_1 * (exp(logC_2))^h_2 + x_1[i]^h_1 * (exp(logC_2))^h_2 + x_2[i]^h_2 * (exp(logC_1))^h_1 + alpha * x_1[i]^h_1 * x_2[i]^h_2) , sigma);
            }
        }
        generated quantities {
            vector[N_test] y_rep;
            for(i in 1:N_test){
                y_rep[i] = normal_rng( ( (exp(logC_1))^h_1 * (exp(logC_2))^h_2 * e_0 + x_1_test[i]^h_1 * (exp(logC_2))^h_2 * e_1 * e_0 + x_2_test[i]^h_2 * (exp(logC_1))^h_1 * e_2 * e_0 + alpha * x_1_test[i]^h_1 * x_2_test[i]^h_2 * e_3 * e_0) / ((exp(logC_1))^h_1 * (exp(logC_2))^h_2 + x_1_test[i]^h_1 * (exp(logC_2))^h_2 + x_2_test[i]^h_2 * (exp(logC_1))^h_1 + alpha * x_1_test[i]^h_1 * x_2_test[i]^h_2) , sigma);
            }
        }
        """

np.random.seed(100)

u_vec_s, pval_vec_s, u_vec_m, pval_vec_m = np.array([]), np.array([]), np.array([]), np.array([])
ch1_combo_mono_df = pd.read_csv('data/dream/ch1_train_combination_and_monoTherapy.csv', na_values=['.', 'ND'])
ch1_combo_mono = ch1_combo_mono_df.to_numpy()

n_samples = args.n_bootstrap  # n_samples = 200 # 500

sm = pystan.StanModel(model_code=synba)

for r in range(len(ch1_combo_mono)):
    row = int(r)
    combination_id = ch1_combo_mono[row, 13]
    cell_line = ch1_combo_mono[row, 0]

    # Filter out the following:
    # (1) rows with Rep2 data
    # (2) rows that have not passed QA
    # (3) rows with negative combination response(s)

    if not os.path.exists(os.path.join(os.getcwd(), 'data/dream/Raw_Data/ch1_training_combinations_csv/',
                                       combination_id + '.' + cell_line + '.Rep' + '2' + '.csv')):
        df, x1, x2, y, dose_mat = name_to_data_dream(combination_id, cell_line, rep_no='1', add_epsilon=False)
        x1 = x1.flatten()
        x2 = x2.flatten()
        y = y.flatten()
        # print(combination_id, cell_line, dose_mat)
        if ch1_combo_mono[row, 12] == 1 and np.min(y) >= 0:
            # TODO perform the train-test split
            leave_out_mono_1 = np.random.choice(np.array([6, 12, 18, 24, 30]), 1, replace=False)
            leave_out_mono_2 = np.random.choice(np.array([1, 2, 3, 4, 5]), 1, replace=False)
            leave_out_mono = np.append(leave_out_mono_1, leave_out_mono_2)
            leave_out_combo = np.random.choice(np.array([7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35]), 5, replace=False)
            leave_out = np.append(leave_out_mono, leave_out_combo)
            x1_train = np.delete(x1, leave_out)
            x2_train = np.delete(x2, leave_out)
            y_train = np.delete(y, leave_out)
            x1_test = x1[leave_out]
            x2_test = x2[leave_out]
            y_test = y[leave_out]

            # Train SynBa
            data = {'N': len(y_train), 'x_1': x1_train, 'x_2': x2_train, 'y': y_train,
                        'N_test': len(y_test), 'x_1_test': x1_test, 'x_2_test': x2_test}
            # Train the model and generate samples
            fit1 = sm.sampling(data=data, iter=args.n_iter, chains=4, warmup=args.n_warmup_iter, thin=1,
                                seed=101, control=dict(adapt_delta=args.adapt_delta))
            y_rep = fit1['y_rep']
            u = np.zeros(len(y_test))
            for ii in range(len(y_test)):
                u[ii] = len(np.where(y_rep[:, ii] < y_test[ii])[0])/len(y_rep)
            u_vec_s = np.append(u_vec_s, u)
            pval = stats.kstest(u, stats.uniform(loc=0.0, scale=1.0).cdf)[1]
            pval_vec_s = np.append(pval_vec_s, pval)
            print('p-value for the K-S test: ', pval)

            if args.save_output:
                np.save(args.path + 'pval_dream_synba.npy', pval_vec_s)
                np.save(args.path + 'uvec_dream_synba.npy', u_vec_s)

            # Train MuSyC
            model = MuSyC(E0_bounds=(50, 150), E1_bounds=(0, 150), E2_bounds=(0, 150), E3_bounds=(0, 150),
                    variant="full")
            model.fit(x1_train, x2_train, y_train, bootstrap_iterations=n_samples, use_jacobian=False,
                    **{'method': 'trf', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'max_nfev': 2000})
            if model.bootstrap_parameters is not None:
                y_rep = musyc_sample_from_bootstrap(model, x1_test, x2_test, y_test, n_samples_per_boot=10)
                u = np.zeros(len(y_test))
                for ii in range(len(y_test)):
                    u[ii] = len(np.where(y_rep[:,ii] < y_test[ii])[0])/np.shape(y_rep)[0]
                u_vec_m = np.append(u_vec_m, u)
                pval = stats.kstest(u, stats.uniform(loc=0.0, scale=1.0).cdf)[1]
                pval_vec_m = np.append(pval_vec_m, pval)
                print('p-value for the K-S test: ', pval)
                if args.save_output:
                    np.save(args.path + 'pval_dream_musyc.npy', pval_vec_m)
                    np.save(args.path + 'uvec_dream_musyc.npy', u_vec_m)
            else:
                print("model.bootstrap_parameters is None for MuSyC")

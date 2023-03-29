from utils import *
import os
import warnings
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.utilities import print_summary, positive

np.set_printoptions(suppress=True)

from HandGP.utilities import (predict_in_observations_lower, predict_in_observations_upper, compute_prior_hyperparameters, trapezoidal_area, predict_in_observations, fit_Hand, K_multiplicative)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='training')

# general configs
parser.add_argument('--dataset', type=str, default='nci-almanac-subset', metavar='N',
                    help="dataset used during training, e.g. 'dream', 'nci-almanac-subset', ...")
parser.add_argument('--save_output', type=str2bool, nargs='?', dest='save_output', const=True, default=True,
                    help="whether to save the output or not")
parser.add_argument('--path', type=str, default="", metavar='N',
                    help="folder to save the numpy arrays")
parser.add_argument('--batch_no', type=int, default=0, metavar='N',
                    help='the batch number selected for the training on dream')
parser.add_argument('--cell_no', type=int, default=0, metavar='N',
                    help='the cell selected for the training on nci-almanac-subset')

args, unknown = parser.parse_known_args()
print("args not parsed in train: ", unknown)

f64 = gpflow.utilities.to_default_float
np.random.seed(100)
tf.random.set_seed(100)

test_rmse_handgp = []
path = args.path

if args.dataset == 'dream':
    ch1_combo_mono_df = pd.read_csv('data/dream/ch1_train_combination_and_monoTherapy.csv', na_values=['.', 'ND'])
    ch1_combo_mono = ch1_combo_mono_df.to_numpy()
    print('Length of the DREAM dataset:', len(ch1_combo_mono))

    C1_upper_bound, C1_lower_bound, C2_upper_bound, C2_lower_bound = 1e6, 1e-10, 1e6, 1e-10

    for row in range(len(ch1_combo_mono)):
        combination_id = ch1_combo_mono[row, 13]
        cell_line = ch1_combo_mono[row, 0]

        # Filter out the following:
        # (1) rows with Rep2 data
        # (2) rows that have not passed QA
        # (3) rows with negative combination response(s)

        if not os.path.exists(os.path.join(os.getcwd(), 'data/dream/Raw_Data/ch1_training_combinations_csv/',
                                           combination_id + '.' + cell_line + '.Rep' + '2' + '.csv')):
            df, x1, x2, y, dose_mat = name_to_data_dream(combination_id, cell_line, rep_no='1', add_epsilon=False)
            if ch1_combo_mono[row, 12] == 1 and np.min(y) >= 0:
                print(combination_id, cell_line, dose_mat)
                leave_out_mono_1 = np.random.choice(np.array([6, 12, 18, 24, 30]), 1, replace=False)
                leave_out_mono_2 = np.random.choice(np.array([1, 2, 3, 4, 5]), 1, replace=False)
                leave_out_mono = np.append(leave_out_mono_1, leave_out_mono_2)
                leave_out_combo = np.random.choice(np.array([7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35]), 5, replace=False)
                leave_out = np.append(leave_out_mono, leave_out_combo)

                # y /= 100.0
                x1_train, x2_train, y_train = np.delete(x1, leave_out), np.delete(x2, leave_out), np.delete(y, leave_out)
                x1_test, x2_test, y_test = x1[leave_out], x2[leave_out], y[leave_out]

                # HandGP
                # Preprocess data and set hyperparameters
                x_1, x_2, y = x1_train.reshape(-1, 1), x2_train.reshape(-1, 1), y_train.reshape(-1, 1)
                df = pd.DataFrame()  # <---- shape: (0, 0)
                df[['x1', 'x2', 'y']] = np.concatenate((x_1, x_2, y), axis=1)

                Effect = df['y'].values.reshape(-1, 1).copy()
                Dose_A = df['x1'].values.astype(float).copy()
                Dose_B = df['x2'].values.astype(float).copy()
                Dose_AB = np.concatenate((Dose_A.reshape(-1, 1), Dose_B.reshape(-1, 1)), axis=1)

                Effect_B = df[df['x1'] == 0]['y'].to_numpy().reshape(-1, 1).astype(float)
                Effect_A = df[df['x2'] == 0]['y'].to_numpy().reshape(-1, 1).astype(float)
                Dose_A = df[df['x2'] == 0]['x1'].to_numpy().reshape(-1, 1).astype(float)
                Dose_B = df[df['x1'] == 0]['x2'].to_numpy().reshape(-1, 1).astype(float)

                eff_max_a, eff_max_b = np.max(Effect_A), np.max(Effect_B)
                eff_min_a, eff_min_b = np.min(Effect_A), np.min(Effect_B)
                eff_max = np.max(Effect)
                c_a, c_b = eff_max_a / eff_min_a, eff_max_b / eff_min_b

                # set hyperparameters
                A_max, B_max = np.max(Dose_A), np.max(Dose_B)
                alphaA, betaA = compute_prior_hyperparameters(A_max, 0.1 * A_max / c_a)
                alphaB, betaB = compute_prior_hyperparameters(B_max, 0.1 * B_max / c_b)

                eff_max_a = np.max(Effect_A)
                eff_max_b = np.max(Effect_B)
                eff_max = np.max([eff_max_a, eff_max_b])

                alpha_var, beta_var = compute_prior_hyperparameters(eff_max, 0.00001 * eff_max)

                zeros_A = np.zeros((Dose_A.shape))
                zeros_B = np.zeros((Dose_B.shape))

                Dose_A_mono = np.concatenate((Dose_A.reshape(-1, 1), zeros_A.reshape(-1, 1)), axis=0)
                Dose_B_mono = np.concatenate((zeros_B.reshape(-1, 1), Dose_B.reshape(-1, 1)), axis=0)

                Dose_AB_mono = np.concatenate((Dose_A_mono.reshape(-1, 1), Dose_B_mono.reshape(-1, 1)), axis=1)
                Effect_mono = np.concatenate((Effect_A.reshape(-1, 1), Effect_B.reshape(-1, 1)), axis=0)

                Dose_AB = np.concatenate((Dose_AB, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono), axis=0)
                Effect = np.concatenate((Effect.reshape(-1, 1), Effect_mono.reshape(-1, 1), Effect_mono.reshape(-1, 1),
                                         Effect_mono.reshape(-1, 1), Effect_mono.reshape(-1, 1)), axis=0)

                # Train the model
                [l1_init, l2_init] = np.meshgrid(np.linspace(1e-10, np.max(Dose_A), 5),
                                                 np.linspace(1e-10, np.max(Dose_B), 5))
                l1_init, l2_init = l1_init.reshape(-1, 1), l2_init.reshape(-1, 1)
                Lik_null, Lik_full, var_init = np.zeros((25, 1)), np.zeros((25, 1)), np.zeros((25, 1))
                for i in range(0, 25):
                    try:
                        init_lengthscale_da = l1_init[i, 0]
                        init_lengthscale_db = l2_init[i, 0]
                        init_variance = eff_max
                        init_likelihood_variance = 0.01

                        k = K_multiplicative()

                        m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

                        m.likelihood.variance.assign(0.001)
                        m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
                        m.kernel.lengthscale_da.assign(init_lengthscale_da)
                        m.kernel.lengthscale_db.assign(init_lengthscale_db)
                        m.kernel.variance_da.assign(eff_max)

                        m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var),
                                                                             np.float64(beta_var))

                        m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
                        m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))

                        opt = gpflow.optimizers.Scipy()

                        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
                        # print_summary(m)

                        Lik_full[i, 0] = np.asarray(m.training_loss())
                        var_init[i, 0] = np.asarray(m.kernel.variance_da)
                    except:
                        Lik_full[i, 0] = 'NaN'
                        print('Cholesky was not successful')

                try:
                    index = np.where(Lik_full == np.nanmin(Lik_full))[0][0]
                    init_lengthscale_da = l1_init[index, 0]
                    init_lengthscale_db = l2_init[index, 0]
                    init_var = var_init[index, 0]

                    m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

                    m.likelihood.variance.assign(0.001)
                    m.kernel.lengthscale_da.assign(init_lengthscale_da)
                    m.kernel.lengthscale_db.assign(init_lengthscale_db)
                    m.kernel.variance_da.assign(init_var)
                    # priors
                    m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
                    m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
                    m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
                    m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))

                    opt = gpflow.optimizers.Scipy()

                    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

                    print_summary(m)

                    xx = np.concatenate((x1_test.reshape(-1, 1), x2_test.reshape(-1, 1)), axis=1)   # test points must be of shape (N, D)
                    xx = xx.astype(np.double)

                    mean, var = m.predict_y(xx)
                    print(y_test, mean, var)

                    if not np.any(np.isnan(mean)) and not np.any(np.isnan(var)):
                        mean = mean.numpy()
                        rmse = np.sqrt(np.sum((np.squeeze(mean) - y_test) ** 2) / len(y_test))
                        test_rmse_handgp.append(rmse)
                        print('Mean test RMSE for HandGP after', len(test_rmse_handgp), 'examples: ', np.mean(test_rmse_handgp))
                        print('Median test RMSE for HandGP after', len(test_rmse_handgp), 'examples: ', np.median(test_rmse_handgp))

                        if args.save_output:
                            np.save(path + 'te_rmse_handgp_dream_{}.npy'.format(args.batch_no), test_rmse_handgp)
                except:
                    print('None of the Cholesky operations were successful')

if args.dataset == 'nci-almanac-subset':
    data = pd.read_csv('data/nci_almanac/NCI-ALMANAC_subset_555300.csv', na_values=['.', 'ND'])
    data = data.to_numpy()
    compound_list = np.unique(data[:, 2])
    cell_list = np.unique(data[:, 4])
    D = len(compound_list)
    C = len(cell_list)
    assert D == 50
    assert C == 60

    C1_upper_bound, C1_lower_bound, C2_upper_bound, C2_lower_bound = 1e1, 1e-15, 1e1, 1e-15

    cell_line = cell_list[args.cell_no]
    progress = np.array([])
    t = time.time()
    for j in range(D):
        for k in range(D):
            combination_id = compound_list[j] + '.' + compound_list[k]
            combo = combination_id + '.' + cell_line
            combo = combo.replace(" ", "-")
            combo = combo.replace("/", "-")
            if os.path.exists(os.path.join(os.getcwd(), 'data/nci_almanac/combinations/' + combo + '.npy')):
                x1, x2, y = name_to_data_nci_almanac_subset(combination_id, cell_line, add_epsilon=False)
                assert int(len(y)/5) == 3 or int(len(y)/5) == 6
                if np.min(y) >= 0:
                    try:
                        leave_out = np.random.choice(np.arange(len(y)), int(len(y)/5), replace=False)
                        x1_train = np.delete(x1, leave_out)
                        x2_train = np.delete(x2, leave_out)
                        y_train = np.delete(y, leave_out)
                        x1_test = x1[leave_out]
                        x2_test = x2[leave_out]
                        y_test = y[leave_out]

                        # HandGP
                        # Preprocess data and set hyperparameters
                        x_1, x_2, y = x1_train.reshape(-1, 1), x2_train.reshape(-1, 1), y_train.reshape(-1, 1)
                        df = pd.DataFrame()  # <---- shape: (0, 0)
                        df[['x1', 'x2', 'y']] = np.concatenate((x_1, x_2, y), axis=1)

                        Effect = df['y'].values.reshape(-1, 1).copy()
                        Dose_A = df['x1'].values.astype(float).copy()
                        Dose_B = df['x2'].values.astype(float).copy()
                        Dose_AB = np.concatenate((Dose_A.reshape(-1, 1), Dose_B.reshape(-1, 1)), axis=1)

                        Effect_B = df[df['x1'] == 0]['y'].to_numpy().reshape(-1, 1).astype(float)
                        Effect_A = df[df['x2'] == 0]['y'].to_numpy().reshape(-1, 1).astype(float)
                        Dose_A = df[df['x2'] == 0]['x1'].to_numpy().reshape(-1, 1).astype(float)
                        Dose_B = df[df['x1'] == 0]['x2'].to_numpy().reshape(-1, 1).astype(float)

                        eff_max_a, eff_max_b = np.max(Effect_A), np.max(Effect_B)
                        eff_min_a, eff_min_b = np.min(Effect_A), np.min(Effect_B)
                        eff_max = np.max(Effect)
                        c_a, c_b = eff_max_a / eff_min_a, eff_max_b / eff_min_b

                        # set hyperparameters
                        A_max, B_max = np.max(Dose_A), np.max(Dose_B)
                        alphaA, betaA = compute_prior_hyperparameters(A_max, 0.1 * A_max / c_a)
                        alphaB, betaB = compute_prior_hyperparameters(B_max, 0.1 * B_max / c_b)

                        eff_max_a = np.max(Effect_A)
                        eff_max_b = np.max(Effect_B)
                        eff_max = np.max([eff_max_a, eff_max_b])

                        alpha_var, beta_var = compute_prior_hyperparameters(eff_max, 0.00001 * eff_max)

                        zeros_A = np.zeros((Dose_A.shape))
                        zeros_B = np.zeros((Dose_B.shape))

                        Dose_A_mono = np.concatenate((Dose_A.reshape(-1, 1), zeros_A.reshape(-1, 1)), axis=0)
                        Dose_B_mono = np.concatenate((zeros_B.reshape(-1, 1), Dose_B.reshape(-1, 1)), axis=0)

                        Dose_AB_mono = np.concatenate((Dose_A_mono.reshape(-1, 1), Dose_B_mono.reshape(-1, 1)), axis=1)
                        Effect_mono = np.concatenate((Effect_A.reshape(-1, 1), Effect_B.reshape(-1, 1)), axis=0)

                        Dose_AB = np.concatenate((Dose_AB, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono), axis=0)
                        Effect = np.concatenate((Effect.reshape(-1, 1), Effect_mono.reshape(-1, 1), Effect_mono.reshape(-1, 1),
                                                 Effect_mono.reshape(-1, 1), Effect_mono.reshape(-1, 1)), axis=0)

                        # Train the model
                        [l1_init, l2_init] = np.meshgrid(np.linspace(1e-10, np.max(Dose_A), 5),
                                                         np.linspace(1e-10, np.max(Dose_B), 5))
                        l1_init, l2_init = l1_init.reshape(-1, 1), l2_init.reshape(-1, 1)
                        Lik_null, Lik_full, var_init = np.zeros((25, 1)), np.zeros((25, 1)), np.zeros((25, 1))
                        for i in range(0, 25):
                            try:
                                init_lengthscale_da = l1_init[i, 0]
                                init_lengthscale_db = l2_init[i, 0]
                                init_variance = eff_max
                                init_likelihood_variance = 0.01

                                k = K_multiplicative()

                                m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

                                m.likelihood.variance.assign(0.001)
                                m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
                                m.kernel.lengthscale_da.assign(init_lengthscale_da)
                                m.kernel.lengthscale_db.assign(init_lengthscale_db)
                                m.kernel.variance_da.assign(eff_max)
                                m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var),
                                                                                     np.float64(beta_var))
                                m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
                                m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))

                                opt = gpflow.optimizers.Scipy()

                                opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

                                Lik_full[i, 0] = np.asarray(m.training_loss())
                                var_init[i, 0] = np.asarray(m.kernel.variance_da)
                            except:
                                Lik_full[i, 0] = 'NaN'
                                print('Cholesky was not successful')

                        try:
                            index = np.where(Lik_full == np.nanmin(Lik_full))[0][0]
                            init_lengthscale_da = l1_init[index, 0]
                            init_lengthscale_db = l2_init[index, 0]
                            init_var = var_init[index, 0]

                            m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

                            m.likelihood.variance.assign(0.001)
                            m.kernel.lengthscale_da.assign(init_lengthscale_da)
                            m.kernel.lengthscale_db.assign(init_lengthscale_db)
                            m.kernel.variance_da.assign(init_var)
                            # priors
                            # m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(0.14), np.float64(1.14))
                            m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
                            m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
                            m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
                            m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))

                            opt = gpflow.optimizers.Scipy()

                            opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

                            print_summary(m)

                            xx = np.concatenate((x1_test.reshape(-1, 1), x2_test.reshape(-1, 1)), axis=1)   # test points must be of shape (N, D)
                            xx = xx.astype(np.double)

                            mean, var = m.predict_y(xx)
                            print(y_test, mean, var)

                            if not np.any(np.isnan(mean)) and not np.any(np.isnan(var)):
                                mean = mean.numpy()
                                rmse = np.sqrt(np.sum((np.squeeze(mean) - y_test) ** 2) / len(y_test))

                                test_rmse_handgp.append(rmse)
                                print('Mean test RMSE for HandGP after', len(test_rmse_handgp), 'examples: ', np.mean(test_rmse_handgp))
                                print('Median test RMSE for HandGP after', len(test_rmse_handgp), 'examples: ', np.median(test_rmse_handgp))

                                if args.save_output:
                                    np.save(path + 'te_rmse_handgp_nci_{}.npy'.format(args.cell_no), test_rmse_handgp)
                        except:
                            print('None of the Cholesky operations were successful')
                    except:
                        pass

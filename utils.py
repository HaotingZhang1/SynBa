import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from synergy.combination import MuSyC, BRAID, Zimmer
import argparse


def str2bool(v):
    """
    Source code copied from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Compute MuSyC output response given dosage and parameters
def musyc_2d(x_1, x_2, e_0, e_1, e_2, e_3, logC_1, logC_2, h_1, h_2, alpha, sigma, add_noise=False):
    A = (np.exp(logC_1)) ** h_1 * (np.exp(logC_2)) ** h_2 * e_0
    B = x_1 ** h_1 * (np.exp(logC_2)) ** h_2 * e_1 * e_0
    C = x_2 ** h_2 * (np.exp(logC_1)) ** h_1 * e_2 * e_0
    D = alpha * x_1 ** h_1 * x_2 ** h_2 * e_3 * e_0
    AA = (np.exp(logC_1)) ** h_1 * (np.exp(logC_2)) ** h_2
    BB = x_1 ** h_1 * (np.exp(logC_2)) ** h_2
    CC = x_2 ** h_2 * (np.exp(logC_1)) ** h_1
    DD = alpha * x_1 ** h_1 * x_2 ** h_2
    if add_noise:
        z = np.random.normal(0, sigma, len(x_1))
    else:
        z = 0
    y = (A + B + C + D) / (AA + BB + CC + DD) + z
    return y


def draw_vertical_gaussian(support, sd=1.0, height=1.0, xpos=0.0, ypos=0.0, ax=None, **kwargs):
    gaussian = np.exp((-support ** 2.0) / (2 * sd ** 2.0))
    gaussian /= gaussian.max()
    gaussian *= height
    plt.plot(gaussian + xpos, support + ypos, **kwargs)


def name_to_data_dream(combination_id, cell_line, rep_no='1', xmin=None, add_epsilon=False):
    dose_mat = pd.read_csv('data/dream/Raw_Data/ch1_training_combinations_csv/' +
                           combination_id + '.' + cell_line + '.Rep' + rep_no + '.csv', na_values=['.', 'ND'])
    agent2_mono_dose = dose_mat.columns[1:7].to_numpy().astype(np.float32)
    dose_mat = dose_mat.to_numpy()
    agent1_mono_dose = dose_mat[0:6, 0].astype(np.float32)
    if add_epsilon is True:
        if xmin is not None:
            agent1_mono_dose[0] = xmin
            agent2_mono_dose[0] = xmin
        else:
            agent1_mono_dose[0] = agent1_mono_dose[1] * 0.0001
            agent2_mono_dose[0] = agent2_mono_dose[1] * 0.0001
    dose_mat = dose_mat[0:6, 1:7].astype(np.float32)
    x_1 = np.repeat(agent1_mono_dose, 6)
    x_2 = np.tile(agent2_mono_dose, 6)
    y = np.reshape(dose_mat, (36))
    x_1, x_2, y = np.reshape(x_1, (36, 1)), np.reshape(x_2, (36, 1)), np.reshape(y, (36, 1))
    df = pd.DataFrame(np.concatenate((x_1, x_2, y), axis=1), columns=['drug1.conc', 'drug2.conc', 'effect'])
    x_1 = x_1.flatten()
    x_2 = x_2.flatten()
    y = y.flatten()
    return df, x_1, x_2, y, dose_mat


def name_to_data_nci_almanac_subset(combination_id, cell_line, xmin=None, add_epsilon=False):
    combination_id = combination_id.replace(" ", "-")
    cell_line = cell_line.replace(" ", "-")
    cell_line = cell_line.replace("/", "-")
    data = np.load('nci_almanac_data/combinations/' + combination_id + '.' + cell_line + '.npy', allow_pickle=True)
    ind0 = np.where(data[:, 0] == 0)[0]
    ind0c = np.where(data[:, 0] > 0)[0]
    non_0_min = np.min(data[ind0c, 0])
    ind1 = np.where(data[:, 1] == 0)[0]
    ind1c = np.where(data[:, 1] > 0)[0]
    non_1_min = np.min(data[ind1c, 0])
    if add_epsilon is True:
        if xmin is not None:
            data[ind0, 0] = xmin
            data[ind1, 1] = xmin
        else:
            data[ind0, 0] = non_0_min * 0.0001
            data[ind1, 1] = non_1_min * 0.0001
    x_1, x_2, y = data[:, 0].astype(np.float32), data[:, 1].astype(np.float32), data[:, 5].astype(np.float32)
    x_1 = x_1.flatten()
    x_2 = x_2.flatten()
    y = y.flatten()
    return x_1, x_2, y


def musyc_extract_bootstrap_parameters(model):
    E0_boot = model.bootstrap_parameters[:, 0]
    E1_boot = model.bootstrap_parameters[:, 1]
    E2_boot = model.bootstrap_parameters[:, 2]
    E3_boot = model.bootstrap_parameters[:, 3]
    h1_boot = model.bootstrap_parameters[:, 4]
    h2_boot = model.bootstrap_parameters[:, 5]
    C1_boot = model.bootstrap_parameters[:, 6]
    C2_boot = model.bootstrap_parameters[:, 7]
    alpha12_boot = model.bootstrap_parameters[:, 8]
    alpha21_boot = model.bootstrap_parameters[:, 9]
    if np.shape(model.bootstrap_parameters)[1] > 10:
        gamma12_boot = model.bootstrap_parameters[:, 10]
        gamma21_boot = model.bootstrap_parameters[:, 11]
        boot_param = [E0_boot, E1_boot, E2_boot, E3_boot, h1_boot, h2_boot, C1_boot, C2_boot,
                      alpha12_boot, alpha21_boot, gamma12_boot, gamma21_boot]
    else:
        boot_param = [E0_boot, E1_boot, E2_boot, E3_boot, h1_boot, h2_boot, C1_boot, C2_boot,
                      alpha12_boot, alpha21_boot]
    return boot_param


def musyc_sample_from_bootstrap(model, x1, x2, y, n_samples_per_boot=10):
    """
    :param model: The trained MuSyC model
    :param x1: Test dosage 1
    :param x2: Test dosage 2
    :param y: True response of the test dosage (x1, x2)
    :param n_samples_per_boot: The number of required samples per bootstrap
    :return: A matrix containing multiple samples of response for each test dosage.
    """
    n_boot = len(model.bootstrap_parameters[:, 0])
    y_sample_mat = np.zeros((n_samples_per_boot * n_boot, len(y)))
    for i in range(n_boot):
        e_0 = model.bootstrap_parameters[i, 0]
        e_1 = model.bootstrap_parameters[i, 1]
        e_2 = model.bootstrap_parameters[i, 2]
        e_3 = model.bootstrap_parameters[i, 3]
        h_1 = model.bootstrap_parameters[i, 4]
        h_2 = model.bootstrap_parameters[i, 5]
        C_1 = model.bootstrap_parameters[i, 6]
        C_2 = model.bootstrap_parameters[i, 7]
        alpha_12 = model.bootstrap_parameters[i, 8]
        alpha_21 = model.bootstrap_parameters[i, 9]
        if np.shape(model.bootstrap_parameters)[1] > 10:
            gamma_12 = model.bootstrap_parameters[i, 10]
            gamma_21 = model.bootstrap_parameters[i, 11]
            model_boot = MuSyC(E0=e_0, E1=e_1, E2=e_2, E3=e_3, h1=h_1, h2=h_2, C1=C_1, C2=C_2,
                               alpha12=alpha_12, alpha21=alpha_21, gamma12=gamma_12, gamma21=gamma_21, variant="full")
        else:
            model_boot = MuSyC(E0=e_0, E1=e_1, E2=e_2, E3=e_3, h1=h_1, h2=h_2, C1=C_1, C2=C_2,
                               alpha12=alpha_12, alpha21=alpha_21, variant="no_gamma")
        y_mean_pred = model_boot.E(d1=x1, d2=x2)
        sigma = np.sqrt(np.sum((y - y_mean_pred) ** 2) / (len(y) - 1))
        start = n_samples_per_boot * i
        end = n_samples_per_boot * (i + 1)
        y_sample_mat[start:end, :] = np.tile(y_mean_pred, (n_samples_per_boot, 1)) + np.random.randn(n_samples_per_boot,
                                                                                                     len(y)) * sigma
    return y_sample_mat


def test_ll_rmse_2d(model, x1_train, x2_train, y_train, x1_test, x2_test, y_test, model_type="synba", n_samples=100):
    """
    :param model: The trained model object
    :param x1_train: Dosage 1 in training data
    :param x2_train: Dosage 2 in training data
    :param y_train: Response of the dosage (x1_train, x2_train) in training data
    :param x1_test: Dosage 1 in test data
    :param x2_test: Dosage 2 in test data
    :param y_test: Response of the dosage (x1_test, x2_test) in test data
    :param model_type: The model type (can be one of SynBa, MuSyC, BRAID or the Effective Dose model), inserted as
                        a string (i.e. "synba", musyc", "braid" or "zimmer").
    :param n_samples: The number of bootstrap samples to be considered.
    :return logpdf: The estimated test log-likelihood.
    :return te_rmse: The test RMSE.
    :return N: The number of samples used to obtain the above two quantities.
    """
    if model_type == "synba":
        # Read samples from the posterior distribution of the parameters
        e_0, e_1, e_2, e_3 = model['e_0'], model['e_1'], model['e_2'], model['e_3']
        logC_1, logC_2, h_1, h_2 = model['logC_1'], model['logC_2'], model['h_1'], model['h_2']
        alpha, sigma = model['alpha'], model['sigma']
        # Compute test log-likelihood
        logpdf = 0
        if len(sigma) < n_samples:
            n_samples = len(sigma)
        logpdf_vec = np.zeros(n_samples)
        N = 0
        for i in np.arange(len(sigma) - n_samples, len(sigma)):
            A = (np.exp(logC_1[i])) ** h_1[i] * (np.exp(logC_2[i])) ** h_2[i] * e_0[i]
            B = x1_test ** h_1[i] * (np.exp(logC_2[i])) ** h_2[i] * e_1[i] * e_0[i]
            C = x2_test ** h_2[i] * (np.exp(logC_1[i])) ** h_1[i] * e_2[i] * e_0[i]
            D = alpha[i] * x1_test ** h_1[i] * x2_test ** h_2[i] * e_3[i] * e_0[i]
            AA = (np.exp(logC_1[i])) ** h_1[i] * (np.exp(logC_2[i])) ** h_2[i]
            BB = x1_test ** h_1[i] * (np.exp(logC_2[i])) ** h_2[i]
            CC = x2_test ** h_2[i] * (np.exp(logC_1[i])) ** h_1[i]
            DD = alpha[i] * x1_test ** h_1[i] * x2_test ** h_2[i]
            y_test_pred = (A + B + C + D) / (AA + BB + CC + DD)
            # test log likelihood
            logpdf_i = np.sum(stats.norm.logpdf(y_test_pred - y_test, loc=0, scale=sigma[i]))
            if N == 0 and not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                logpdf = logpdf_i
                N += 1
            elif not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                logpdf = np.logaddexp(logpdf, logpdf_i)
                N += 1
            logpdf_vec[i - (len(sigma) - n_samples)] = logpdf_i
        logpdf -= np.log(N)
        logpdf /= len(y_test)
        # Compute test RMSE
        y_rep = model['y_rep']
        y_rep_mean = np.mean(y_rep, axis=0)
        te_rmse_synba = np.sqrt(np.sum((y_rep_mean - y_test) ** 2) / len(y_test))
        return logpdf, te_rmse_synba, N
    elif model_type == "musyc_bootstrap":
        E0_boot = model.bootstrap_parameters[:, 0]
        E1_boot = model.bootstrap_parameters[:, 1]
        E2_boot = model.bootstrap_parameters[:, 2]
        E3_boot = model.bootstrap_parameters[:, 3]
        h1_boot = model.bootstrap_parameters[:, 4]
        h2_boot = model.bootstrap_parameters[:, 5]
        C1_boot = model.bootstrap_parameters[:, 6]
        C2_boot = model.bootstrap_parameters[:, 7]
        alpha12_boot = model.bootstrap_parameters[:, 8]
        alpha21_boot = model.bootstrap_parameters[:, 9]
        if np.shape(model.bootstrap_parameters)[1] > 10:
            gamma12_boot = model.bootstrap_parameters[:, 10]
            gamma21_boot = model.bootstrap_parameters[:, 11]
            if len(E0_boot) < n_samples:
                n_samples = len(E0_boot)
            logpdf_vec = np.zeros(n_samples)
            N = 0
            for i in np.arange(len(E0_boot) - n_samples, len(E0_boot)):
                model_boot = MuSyC(E0=E0_boot[i], E1=E1_boot[i], E2=E2_boot[i], E3=E3_boot[i],
                                   h1=h1_boot[i], h2=h2_boot[i], C1=C1_boot[i], C2=C2_boot[i],
                                   alpha12=alpha12_boot[i], alpha21=alpha21_boot[i],
                                   gamma12=gamma12_boot[i], gamma21=gamma21_boot[i], variant="full")
                y_train_pred = model_boot.E(d1=x1_train, d2=x2_train)
                # compute the unbiased estimate for the standard deviation
                sigma = np.sqrt(np.sum((y_train - y_train_pred) ** 2) / (len(y_train) - 1))
                y_test_pred = model_boot.E(d1=x1_test, d2=x2_test)
                # compute test log likelihood
                logpdf_i = np.sum(stats.norm.logpdf(y_test_pred - y_test, loc=0, scale=sigma))
                if N == 0 and not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                    y_test_pred_mean = y_test_pred
                    logpdf = logpdf_i
                    N += 1
                elif not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                    y_test_pred_mean += y_test_pred
                    logpdf = np.logaddexp(logpdf, logpdf_i)
                    N += 1
                logpdf_vec[i - (len(E0_boot) - n_samples)] = logpdf_i
        else:
            if len(E0_boot) < n_samples:
                n_samples = len(E0_boot)
            logpdf_vec = np.zeros(n_samples)
            N = 0
            for i in np.arange(len(E0_boot) - n_samples, len(E0_boot)):
                model_boot = MuSyC(E0=E0_boot[i], E1=E1_boot[i], E2=E2_boot[i], E3=E3_boot[i],
                                   h1=h1_boot[i], h2=h2_boot[i], C1=C1_boot[i], C2=C2_boot[i],
                                   alpha12=alpha12_boot[i], alpha21=alpha21_boot[i],
                                   variant="no_gamma")
                y_train_pred = model_boot.E(d1=x1_train, d2=x2_train)
                # # compute the unbiased estimate for the standard deviation
                sigma = np.sqrt(np.sum((y_train - y_train_pred) ** 2) / (len(y_train) - 1))
                y_test_pred = model_boot.E(d1=x1_test, d2=x2_test)
                # compute test log likelihood
                logpdf_i = np.sum(stats.norm.logpdf(y_test_pred - y_test, loc=0, scale=sigma))
                if N == 0 and not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                    y_test_pred_mean = y_test_pred
                    logpdf = logpdf_i
                    N += 1
                elif not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                    y_test_pred_mean += y_test_pred
                    logpdf = np.logaddexp(logpdf, logpdf_i)
                    N += 1
                logpdf_vec[i - (len(E0_boot) - n_samples)] = logpdf_i
        if N > 0:
            logpdf -= np.log(N)
            logpdf /= len(y_test)
            y_test_pred_mean /= N
            te_rmse_musyc = np.sqrt(np.sum((y_test_pred_mean - y_test) ** 2) / len(y_test))
        else:
            logpdf = np.nan
            te_rmse_musyc = np.nan
        return logpdf, te_rmse_musyc, N
    elif model_type == "braid":
        E0_boot = model.bootstrap_parameters[:, 0]
        E1_boot = model.bootstrap_parameters[:, 1]
        E2_boot = model.bootstrap_parameters[:, 2]
        E3_boot = model.bootstrap_parameters[:, 3]
        h1_boot = model.bootstrap_parameters[:, 4]
        h2_boot = model.bootstrap_parameters[:, 5]
        C1_boot = model.bootstrap_parameters[:, 6]
        C2_boot = model.bootstrap_parameters[:, 7]
        kappa_boot = model.bootstrap_parameters[:, 8]
        if len(E0_boot) < n_samples:
            n_samples = len(E0_boot)
        logpdf_vec = np.zeros(n_samples)
        N = 0
        for i in np.arange(len(E0_boot) - n_samples, len(E0_boot)):
            model_boot = BRAID(E0=E0_boot[i], E1=E1_boot[i], E2=E2_boot[i], E3=E3_boot[i],
                               h1=h1_boot[i], h2=h2_boot[i], C1=C1_boot[i], C2=C2_boot[i],
                               kappa=kappa_boot[i], variant="kappa")
            y_train_pred = model_boot.E(d1=x1_train, d2=x2_train)
            # compute the unbiased estimate for the standard deviation
            sigma = np.sqrt(np.sum((y_train - y_train_pred) ** 2) / (len(y_train) - 1))
            y_test_pred = model_boot.E(d1=x1_test, d2=x2_test)
            # compute test log likelihood
            logpdf_i = np.sum(stats.norm.logpdf(y_test_pred - y_test, loc=0, scale=sigma))
            if N == 0 and not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                y_test_pred_mean = y_test_pred
                logpdf = logpdf_i
                N += 1
            elif not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                y_test_pred_mean += y_test_pred
                logpdf = np.logaddexp(logpdf, logpdf_i)
                N += 1
            logpdf_vec[i - (len(E0_boot) - n_samples)] = logpdf_i
        if N > 0:
            logpdf -= np.log(N)
            logpdf /= len(y_test)
            y_test_pred_mean /= N
            te_rmse_braid = np.sqrt(np.sum((y_test_pred_mean - y_test) ** 2) / len(y_test))
        else:
            logpdf = np.nan
            te_rmse_braid = np.nan
        return logpdf, te_rmse_braid, N
    elif model_type == "zimmer":
        h1_boot = model.bootstrap_parameters[:, 0]
        h2_boot = model.bootstrap_parameters[:, 1]
        C1_boot = model.bootstrap_parameters[:, 2]
        C2_boot = model.bootstrap_parameters[:, 3]
        alpha12_boot = model.bootstrap_parameters[:, 4]
        alpha21_boot = model.bootstrap_parameters[:, 5]
        if len(C1_boot) < n_samples:
            n_samples = len(C1_boot)
        logpdf_vec = np.zeros(n_samples)
        N = 0
        for i in np.arange(len(C1_boot) - n_samples, len(C1_boot)):
            model_boot = Zimmer(h1=h1_boot[i], h2=h2_boot[i], C1=C1_boot[i], C2=C2_boot[i],
                                a12=alpha12_boot[i], a21=alpha21_boot[i])
            y_train_pred = model_boot.E(d1=x1_train, d2=x2_train)
            # compute the unbiased estimate for the standard deviation
            sigma = np.sqrt(
                np.sum((y_train - y_train_pred * np.maximum(100, np.max(y_train))) ** 2) / (len(y_train) - 1))
            y_test_pred = model_boot.E(d1=x1_test, d2=x2_test)
            # compute test log likelihood
            logpdf_i = np.sum(stats.norm.logpdf(y_test_pred * np.maximum(100, np.max(y_train)) - y_test,
                                                loc=0, scale=sigma))
            if N == 0 and not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                y_test_pred_mean = y_test_pred
                logpdf = logpdf_i
                N += 1
            elif not np.isnan(logpdf_i) and not np.isinf(logpdf_i):
                y_test_pred_mean += y_test_pred
                logpdf = np.logaddexp(logpdf, logpdf_i)
                N += 1
            logpdf_vec[i - (len(C1_boot) - n_samples)] = logpdf_i
        if N > 0:
            logpdf -= np.log(N)
            logpdf /= len(y_test)
            y_test_pred_mean /= N
            te_rmse_zimmer = np.sqrt(np.sum((y_test_pred_mean * np.maximum(100,
                                                            np.max(y_train)) - y_test) ** 2) / len(y_test))
        else:
            logpdf = np.nan
            te_rmse_zimmer = np.nan
        # print(np.shape(logpdf_vec))
        return logpdf, te_rmse_zimmer, N

    else:
        print("Model type not included.")

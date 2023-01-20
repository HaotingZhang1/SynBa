from utils import *
import pystan
import os
import warnings
import time


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='training')

# general configs
parser.add_argument('--dataset', type=str, default='dream', metavar='N',
                    help="dataset used during training, e.g. 'dream', 'nci-almanac-subset', ...")
parser.add_argument('--save_output', type=str2bool, nargs='?', dest='save_output', const=True, default=False,
                    help="whether to save the output or not")
parser.add_argument('--path', type=str, default="", metavar='N',
                    help="folder to save the numpy arrays")
# deterministic models configs
parser.add_argument('--n_bootstrap', type=int, default=100, metavar='N', help="number of bootstrap operations in \
                    bootstrap-based methods such as MuSyC")
# SynBa configs
parser.add_argument('--n_iter', type=int, default=1000, metavar='N',
                    help='number of iterations for SynBa')
parser.add_argument('--n_warmup_iter', type=int, default=500, metavar='N',
                    help='number of warmup iterations for SynBa')
parser.add_argument('--adapt_delta', type=float, default=0.9, metavar='N', help="the target average proposal \
                    acceptance probability during the adaptation period in Stan training")

args, unknown = parser.parse_known_args()
print("args not parsed in train: ", unknown)

# os.environ["CC"] = "gcc-4.9"
# # just to be sure
# os.environ["CXX"] = "g++-4.9"

# Define SynBa via Stan
assert args.dataset == 'dream' or args.dataset == 'nci-almanac-subset'
if args.dataset == 'dream':
    synba_unif = """
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
            e_1 ~ uniform(0, 1);
            e_2 ~ uniform(0, 1);
            e_3 ~ uniform(0, 1);
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
    synba_emp_beta = """
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
elif args.dataset == 'nci-almanac-subset':
    synba_unif = """
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
          real<lower=log(1e-15), upper=log(1e1)> logC_1;
          real<lower=log(1e-15), upper=log(1e1)> logC_2;
          real<lower=0> h_1;
          real<lower=0> h_2;
          real<lower=0> alpha;
          real<lower=0> sigma;
        }
        model {
            e_0 ~ normal(100, 3);
            e_1 ~ uniform(0, 1);
            e_2 ~ uniform(0, 1);
            e_3 ~ uniform(0, 1);
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
    synba_emp_beta = """
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
          real<lower=log(1e-15), upper=log(1e1)> logC_1;
          real<lower=log(1e-15), upper=log(1e1)> logC_2;
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

sm1 = pystan.StanModel(model_code=synba_unif)
sm2 = pystan.StanModel(model_code=synba_emp_beta)

test_ll_synba_unif, test_ll_synba_emp_beta, test_ll_musyc, test_ll_zimmer, test_ll_braid = [], [], [], [], []
test_rmse_synba_unif, test_rmse_synba_emp_beta, test_rmse_musyc, test_rmse_zimmer, test_rmse_braid = [], [], [], [], []
n_samples = args.n_bootstrap # 200 # 500
path = args.path
np.random.seed(100)

if args.dataset == 'dream':
    ch1_combo_mono_df = pd.read_csv('data/dream/ch1_train_combination_and_monoTherapy.csv', na_values=['.', 'ND'])
    ch1_combo_mono = ch1_combo_mono_df.to_numpy()
    print('Length of the DREAM dataset:', len(ch1_combo_mono))

    C1_upper_bound, C1_lower_bound, C2_upper_bound, C2_lower_bound = 1e6, 1e-10, 1e6, 1e-10

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
            if ch1_combo_mono[row, 12] == 1 and np.min(y) >= 0:
                print(combination_id, cell_line, dose_mat)
                leave_out_mono_1 = np.random.choice(np.array([6, 12, 18, 24, 30]), 1, replace=False)
                leave_out_mono_2 = np.random.choice(np.array([1, 2, 3, 4, 5]), 1, replace=False)
                leave_out_mono = np.append(leave_out_mono_1, leave_out_mono_2)
                leave_out_combo = np.random.choice(np.array([7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35]), 5, replace=False)
                leave_out = np.append(leave_out_mono, leave_out_combo)
                # print(leave_out)

                x1_train = np.delete(x1, leave_out)
                x2_train = np.delete(x2, leave_out)
                y_train = np.delete(y, leave_out)
                x1_test = x1[leave_out]
                x2_test = x2[leave_out]
                y_test = y[leave_out]

                # x1_train = x1_train[1:]
                # x2_train = x2_train[1:]
                # y_train = y_train[1:]

                # MuSyC
                successfully_trained = True

                print("MuSyC:")
                t = time.time()
                model_musyc = MuSyC(E0_bounds=(50, 150), E1_bounds=(0, 150), E2_bounds=(0, 150), E3_bounds=(0, 150),
                                    variant="full")
                model_musyc.fit(x1_train, x2_train, y_train, bootstrap_iterations=n_samples, use_jacobian=False,
                                **{'method': 'trf', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'max_nfev': 2000})

                if model_musyc.bootstrap_parameters is not None:
                    te_ll_musyc, te_rmse_musyc, te_N_musyc = test_ll_rmse_2d(model_musyc, x1_train, x2_train, y_train,
                                                                              x1_test, x2_test, y_test,
                                                                              model_type="musyc_bootstrap",
                                                                              n_samples=n_samples)
                    if np.isnan(te_ll_musyc) or np.isnan(te_rmse_musyc):
                        successfully_trained = False
                else:
                    successfully_trained = False

                # BRAID
                # 9 parameters: E0, E1, E2, E3, C1, C2, h1, h2, kappa
                if successfully_trained:
                    print("BRAID:")
                    model_braid = BRAID(E0_bounds=(50, 150), E1_bounds=(0, 100), E2_bounds=(0, 100), E3_bounds=(0, 100),
                                        h1_bounds=(1e-5, 100), C1_bounds=(C1_lower_bound, C1_upper_bound),
                                        h2_bounds=(1e-5, 100), C2_bounds=(C2_lower_bound, C2_upper_bound),
                                        variant="kappa")
                    model_braid.fit(x1_train, x2_train, y_train, bootstrap_iterations=n_samples,
                                    **{'method': 'trf', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'max_nfev': 2000})

                    if model_braid.bootstrap_parameters is not None:
                        te_ll_braid, te_rmse_braid, te_N_braid = test_ll_rmse_2d(model_braid, x1_train, x2_train,
                                                                                 y_train, x1_test, x2_test, y_test,
                                                                                 model_type="braid",
                                                                                 n_samples=n_samples)
                        if np.isnan(te_ll_braid) or np.isnan(te_rmse_braid):
                            successfully_trained = False
                    else:
                        successfully_trained = False

                # Effective dose model
                # 6 parameters: C1, C2, h1, h2, a12, a21
                if successfully_trained:
                    print("Zimmer:")
                    model_zimmer = Zimmer(h1_bounds=(1e-10, 500), C1_bounds=(C1_lower_bound, C1_upper_bound),
                                          h2_bounds=(1e-10, 500), C2_bounds=(C2_lower_bound, C2_upper_bound))
                    x1_train_eps = x1_train + 1e-10
                    x2_train_eps = x2_train + 1e-10
                    y_train_normalised = y_train / np.maximum(100, np.max(y_train))
                    model_zimmer.fit(x1_train_eps, x2_train_eps, y_train_normalised, bootstrap_iterations=n_samples,
                                     **{'method': 'trf', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'max_nfev': 2000})

                    if model_zimmer.bootstrap_parameters is not None:
                        te_ll_zimmer, te_rmse_zimmer, te_N_zimmer = test_ll_rmse_2d(model_zimmer, x1_train, x2_train,
                                                                                    y_train, x1_test, x2_test, y_test,
                                                                                    model_type="zimmer",
                                                                                    n_samples=n_samples)
                        if np.isnan(te_ll_zimmer) or np.isnan(te_rmse_zimmer):
                            successfully_trained = False
                    else:
                        successfully_trained = False

                # SynBa
                if successfully_trained:
                    print("SynBa:")
                    # Put our data in a dictionary
                    t = time.time()
                    # Put our data in a dictionary
                    data = {'N': len(y_train), 'x_1': x1_train, 'x_2': x2_train, 'y': y_train,
                            'N_test': len(y_test), 'x_1_test': x1_test, 'x_2_test': x2_test}

                    # MODEL OPTION 1
                    # Train the model and generate samples
                    fit1 = sm1.sampling(data=data, iter=args.n_iter, chains=4, warmup=args.n_warmup_iter, thin=1,
                                        seed=101, control=dict(adapt_delta=0.9))
                    print("SynBa Elapsed time:", time.time() - t, 'seconds')

                    te_ll_synba_unif, te_rmse_synba_unif, te_N_synba_unif = test_ll_rmse_2d(fit1, x1_train, x2_train,
                                            y_train, x1_test, x2_test, y_test, model_type="synba", n_samples=n_samples)

                    # MODEL OPTION 2
                    # Train the model and generate samples
                    fit2 = sm2.sampling(data=data, iter=args.n_iter, chains=4, warmup=args.n_warmup_iter, thin=1,
                                        seed=101, control=dict(adapt_delta=0.9))
                    print("SynBa Elapsed time:", time.time() - t, 'seconds')

                    te_ll_synba_emp_beta, te_rmse_synba_emp_beta, te_N_synba_emp_beta = test_ll_rmse_2d(fit2, x1_train,
                                x2_train, y_train, x1_test, x2_test, y_test, model_type="synba", n_samples=n_samples)

                if successfully_trained:
                    test_ll_musyc.append(te_ll_musyc)
                    print('Mean test log-likelihood for MuSyC after', len(test_ll_musyc), 'examples: ', np.mean(test_ll_musyc))
                    test_ll_braid.append(te_ll_braid)
                    print('Mean test log-likelihood for BRAID after', len(test_ll_braid), 'examples: ', np.mean(test_ll_braid))
                    test_ll_zimmer.append(te_ll_zimmer)
                    print('Mean test log-likelihood for Effective Dose Model after', len(test_ll_zimmer), 'examples: ', np.mean(test_ll_zimmer))
                    test_ll_synba_unif.append(te_ll_synba_unif)
                    print('Mean test log-likelihood for SynBa(unif) after', len(test_ll_synba_unif), 'examples: ',
                          np.mean(test_ll_synba_unif))
                    test_ll_synba_emp_beta.append(te_ll_synba_emp_beta)
                    print('Mean test log-likelihood for SynBa(beta) after', len(test_ll_synba_emp_beta), 'examples: ',
                          np.mean(test_ll_synba_emp_beta))
                    test_rmse_musyc.append(te_rmse_musyc)
                    print('Mean test RMSE for MuSyC after', len(test_rmse_musyc), 'examples: ', np.mean(test_rmse_musyc))
                    test_rmse_braid.append(te_rmse_braid)
                    print('Mean test RMSE for BRAID after', len(test_rmse_braid), 'examples: ', np.mean(test_rmse_braid))
                    test_rmse_zimmer.append(te_rmse_zimmer)
                    print('Mean test RMSE for Effective Dose Model after', len(test_rmse_zimmer), 'examples: ', np.mean(test_rmse_zimmer))
                    test_rmse_synba_unif.append(te_rmse_synba_unif)
                    print('Mean test RMSE for SynBa(unif) after', len(test_rmse_synba_unif), 'examples: ',
                          np.mean(test_rmse_synba_unif))
                    test_rmse_synba_emp_beta.append(te_rmse_synba_emp_beta)
                    print('Mean test RMSE for SynBa(beta) after', len(test_rmse_synba_emp_beta), 'examples: ',
                          np.mean(test_rmse_synba_emp_beta))
                    if args.save_output:
                        np.save(path + 'te_ll_musyc_dream.npy', test_ll_musyc)
                        np.save(path + 'te_ll_braid_dream.npy', test_ll_braid)
                        np.save(path + 'te_ll_zimmer_dream.npy', test_ll_zimmer)
                        np.save(path + 'te_ll_synba_unif_dream.npy', test_ll_synba_unif)
                        np.save(path + 'te_ll_synba_emp_beta_dream.npy', test_ll_synba_emp_beta)
                        np.save(path + 'te_rmse_musyc_dream.npy', test_rmse_musyc)
                        np.save(path + 'te_rmse_braid_dream.npy', test_rmse_braid)
                        np.save(path + 'te_rmse_zimmer_dream.npy', test_rmse_zimmer)
                        np.save(path + 'te_rmse_synba_unif_dream.npy', test_rmse_synba_unif)
                        np.save(path + 'te_rmse_synba_emp_beta_dream.npy', test_rmse_synba_emp_beta)

if args.dataset == 'nci-almanac-subset':
    data = pd.read_csv('data/nci_almanac/NCI-ALMANAC_subset_555300.csv', na_values=['.', 'ND'])
    data = data.to_numpy()
    compound_list = np.unique(data[:, 2])
    cell_list = np.unique(data[:, 4])
    D = len(compound_list)
    C = len(cell_list)
    assert D == 50
    assert C == 60
    # print('cells in NCI-ALMANAC_subset_555300: ', cell_list)
    # print('number of cells in NCI-ALMANAC_subset_555300: ', C)
    # print('compounds in NCI-ALMANAC_subset_555300: ', compound_list)
    # print('number of compounds in NCI-ALMANAC_subset_555300: ', D)

    C1_upper_bound, C1_lower_bound, C2_upper_bound, C2_lower_bound = 1e1, 1e-15, 1e1, 1e-15

    # examples = 0
    cell_line = cell_list[args.cell_no]
    progress = np.array([])
    # print('cell progress: %d / %d' % (i+1, C))
    t = time.time()
    for j in range(D):
        for k in range(D):
            combination_id = compound_list[j] + '.' + compound_list[k]
            combo = combination_id + '.' + cell_line
            combo = combo.replace(" ", "-")
            combo = combo.replace("/", "-")
            if os.path.exists(os.path.join(os.getcwd(), 'data/nci_almanac/combinations/' + combo + '.npy')):
                # examples += 1
                # print('example {} / 36120:'.format(examples))
                x1, x2, y = name_to_data_nci_almanac_subset(combination_id, cell_line, add_epsilon=False)
                assert int(len(y)/5) == 3 or int(len(y)/5) == 6
                if np.min(y) >= 0:
                    leave_out = np.random.choice(np.arange(len(y)), int(len(y)/5), replace=False)
                    x1_train = np.delete(x1, leave_out)
                    x2_train = np.delete(x2, leave_out)
                    y_train = np.delete(y, leave_out)
                    x1_test = x1[leave_out]
                    x2_test = x2[leave_out]
                    y_test = y[leave_out]

                    # MuSyC
                    print("MuSyC:")
                    successfully_trained = True
                    t = time.time()
                    model_musyc = MuSyC(E0_bounds=(50, 150), E1_bounds=(0, 150), E2_bounds=(0, 150),
                                            E3_bounds=(0, 150), variant="no_gamma")
                    model_musyc.fit(x1_train, x2_train, y_train, bootstrap_iterations=n_samples, use_jacobian=False,
                                    **{'method': 'trf', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4, 'max_nfev': 2000})

                    if model_musyc.bootstrap_parameters is not None:
                        te_ll_musyc, te_rmse_musyc, te_N_musyc = test_ll_rmse_2d(model_musyc, x1_train, x2_train,
                                 y_train, x1_test, x2_test, y_test, model_type="musyc_bootstrap", n_samples=n_samples)
                        if np.isnan(te_ll_musyc) or np.isnan(te_rmse_musyc):
                            successfully_trained = False
                    else:
                        successfully_trained = False

                    # BRAID
                    # 9 parameters: E0, E1, E2, E3, C1, C2, h1, h2, kappa
                    if successfully_trained:
                        print("BRAID:")
                        model_braid = BRAID(E0_bounds=(50, 150), E1_bounds=(0, 100), E2_bounds=(0, 100),
                                            E3_bounds=(0, 100),
                                            h1_bounds=(1e-5, 100), C1_bounds=(C1_lower_bound, C1_upper_bound),
                                            h2_bounds=(1e-5, 100), C2_bounds=(C2_lower_bound, C2_upper_bound),
                                            variant="kappa")
                        model_braid.fit(x1_train, x2_train, y_train, bootstrap_iterations=n_samples,
                                        **{'method': 'trf', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4,
                                           'max_nfev': 2000})

                        if model_braid.bootstrap_parameters is not None:
                            te_ll_braid, te_rmse_braid, te_N_braid = test_ll_rmse_2d(model_braid, x1_train, x2_train,
                                     y_train, x1_test, x2_test, y_test, model_type="braid", n_samples=n_samples)
                            if np.isnan(te_ll_braid) or np.isnan(te_rmse_braid):
                                successfully_trained = False
                        else:
                            successfully_trained = False

                    # Effective dose model
                    # 6 parameters: C1, C2, h1, h2, a12, a21
                    if successfully_trained:
                        print("Zimmer:")
                        model_zimmer = Zimmer(h1_bounds=(1e-10, 500), C1_bounds=(C1_lower_bound, C1_upper_bound),
                                              h2_bounds=(1e-10, 500), C2_bounds=(C2_lower_bound, C2_upper_bound))
                        x1_train_eps = x1_train + 1e-15
                        x2_train_eps = x2_train + 1e-15
                        y_train_normalised = y_train / np.maximum(100, np.max(y_train))
                        model_zimmer.fit(x1_train_eps, x2_train_eps, y_train_normalised, bootstrap_iterations=n_samples,
                                         **{'method': 'trf', 'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4,
                                            'max_nfev': 2000})

                        if model_zimmer.bootstrap_parameters is not None:
                            te_ll_zimmer, te_rmse_zimmer, te_N_zimmer = test_ll_rmse_2d(model_zimmer, x1_train,
                                                            x2_train, y_train, x1_test, x2_test, y_test,
                                                            model_type="zimmer", n_samples=n_samples)
                            if np.isnan(te_ll_zimmer) or np.isnan(te_rmse_zimmer):
                                successfully_trained = False
                        else:
                            successfully_trained = False

                    # SynBa
                    if successfully_trained:
                        print("SynBa:")
                        # Put our data in a dictionary
                        t = time.time()
                        # Put our data in a dictionary
                        data = {'N': len(y_train), 'x_1': x1_train, 'x_2': x2_train, 'y': y_train,
                                'N_test': len(y_test), 'x_1_test': x1_test, 'x_2_test': x2_test}
                        # MODEL OPTION 1
                        # Train the model and generate samples
                        fit1 = sm1.sampling(data=data, iter=args.n_iter, chains=4, warmup=args.n_warmup_iter, thin=1,
                                            seed=101, control=dict(adapt_delta=0.9))
                        print("SynBa Elapsed time:", time.time() - t, 'seconds')

                        te_ll_synba_unif, te_rmse_synba_unif, te_N_synba_unif = test_ll_rmse_2d(fit1, x1_train,
                                                            x2_train, y_train, x1_test, x2_test, y_test,
                                                            model_type="synba", n_samples=n_samples)

                        # MODEL OPTION 2
                        # Train the model and generate samples
                        fit2 = sm2.sampling(data=data, iter=args.n_iter, chains=4, warmup=args.n_warmup_iter, thin=1,
                                            seed=101, control=dict(adapt_delta=0.9))
                        print("SynBa Elapsed time:", time.time() - t, 'seconds')

                        te_ll_synba_emp_beta, te_rmse_synba_emp_beta, te_N_synba_emp_beta = test_ll_rmse_2d(fit2,
                                                            x1_train, x2_train, y_train, x1_test, x2_test, y_test,
                                                                model_type="synba", n_samples=n_samples)

                    if successfully_trained:
                        test_ll_musyc.append(te_ll_musyc)
                        print('Mean test log-likelihood for MuSyC after', len(test_ll_musyc), 'examples: ', np.mean(test_ll_musyc))
                        test_ll_braid.append(te_ll_braid)
                        print('Mean test log-likelihood for BRAID after', len(test_ll_braid), 'examples: ', np.mean(test_ll_braid))
                        test_ll_zimmer.append(te_ll_zimmer)
                        print('Mean test log-likelihood for Effective Dose Model after', len(test_ll_zimmer), 'examples: ', np.mean(test_ll_zimmer))
                        test_ll_synba_unif.append(te_ll_synba_unif)
                        print('Mean test log-likelihood for SynBa(unif) after', len(test_ll_synba_unif), 'examples: ', np.mean(test_ll_synba_unif))
                        test_ll_synba_emp_beta.append(te_ll_synba_emp_beta)
                        print('Mean test log-likelihood for SynBa(beta) after', len(test_ll_synba_emp_beta), 'examples: ', np.mean(test_ll_synba_emp_beta))

                        if args.save_output:
                            np.save(path + 'te_ll_musyc_nci.npy'.format(args.cell_no), test_ll_musyc)
                            np.save(path + 'te_ll_braid_nci.npy'.format(args.cell_no), test_ll_braid)
                            np.save(path + 'te_ll_zimmer_nci.npy'.format(args.cell_no), test_ll_zimmer)
                            np.save(path + 'te_ll_synba_unif_nci.npy'.format(args.cell_no), test_ll_synba_unif)
                            np.save(path + 'te_ll_synba_emp_beta_nci.npy'.format(args.cell_no), test_ll_synba_emp_beta)

                        test_rmse_musyc.append(te_rmse_musyc)
                        print('Mean test RMSE for MuSyC after', len(test_rmse_musyc), 'examples: ', np.mean(test_rmse_musyc))
                        test_rmse_braid.append(te_rmse_braid)
                        print('Mean test RMSE for BRAID after', len(test_rmse_braid), 'examples: ', np.mean(test_rmse_braid))
                        test_rmse_zimmer.append(te_rmse_zimmer)
                        print('Mean test RMSE for Effective Dose Model after', len(test_rmse_zimmer), 'examples: ', np.mean(test_rmse_zimmer))
                        test_rmse_synba_unif.append(te_rmse_synba_unif)
                        print('Mean test RMSE for SynBa(unif) after', len(test_rmse_synba_unif), 'examples: ', np.mean(test_rmse_synba_unif))
                        test_rmse_synba_emp_beta.append(te_rmse_synba_emp_beta)
                        print('Mean test RMSE for SynBa(beta) after', len(test_rmse_synba_emp_beta), 'examples: ', np.mean(test_rmse_synba_emp_beta))

                        if args.save_output:
                            np.save(path + 'te_rmse_musyc_nci.npy'.format(args.cell_no), test_rmse_musyc)
                            np.save(path + 'te_rmse_braid_nci.npy'.format(args.cell_no), test_rmse_braid)
                            np.save(path + 'te_rmse_zimmer_nci.npy'.format(args.cell_no), test_rmse_zimmer)
                            np.save(path + 'te_rmse_synba_unif_nci.npy'.format(args.cell_no), test_rmse_synba_unif)
                            np.save(path + 'te_rmse_synba_emp_beta_nci.npy'.format(args.cell_no), test_rmse_synba_emp_beta)

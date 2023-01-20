from utils import *
import pystan
import warnings
import matplotlib as mpl


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rc('font', size=16)
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams['font.family'] = "arial"

warnings.filterwarnings("ignore")

model_1 = """
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}
parameters {
  real<lower=0> e0;
  real<lower=0, upper=1> einf;
  real<lower=log(1e-10), upper=log(1e6)> logC;
  real<lower=0> h;
  real<lower=0> sigma;
}
model {
    h ~ lognormal(0, 1);
    einf ~ beta(1, 1);
    sigma ~ lognormal(0, 1);
    e0 ~ normal(100, 3);
    for(i in 1:N)
    Y[i] ~ normal( e0 + (einf * e0 - e0) / (1.0 + (exp(logC) / x[i]) ^ h), sigma)  T[0, ];
}
"""

model_2 = """
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] Y;
}
parameters {
  real<lower=0> e0;
  real<lower=0, upper=1> einf;
  real<lower=log(1e-10), upper=log(1e6)> logC;
  real<lower=0> h;
  real<lower=0> sigma;
}
model {
    h ~ lognormal(0, 1);
    einf ~ beta(0.46, 0.58);
    sigma ~ lognormal(0, 1);
    e0 ~ normal(100, 3);
    for(i in 1:N)
    Y[i] ~ normal( e0 + (einf * e0 - e0) / (1.0 + (exp(logC) / x[i]) ^ h), sigma)  T[0, ];
}
"""

# Compile model
sm1 = pystan.StanModel(model_code=model_1)
sm2 = pystan.StanModel(model_code=model_2)

# combination_id = 'FASN.MTOR_1'
# cell_line = 'MDA-MB-231'
# Agent 1:
agent1_mono_dose = np.array([0., 0.01, 0.03, 0.1, 0.3, 1.])
y_mono_1 = np.array([100., 96.1, 91.6, 78.2, 62.3, 40.8])

seq = np.arange(start=1, stop=6)   # shuffle all 5 non-zero dosages
np.random.seed(100)
np.random.shuffle(seq)
seq = np.append(0, seq)

agent1_mono_dose[0] = 1e-8
log_x_min, log_x_max = np.log(1e-8)-0.5, int(np.ceil(np.max(np.log(agent1_mono_dose)))) + 8
log_x_plot = np.linspace(log_x_min, log_x_max, 100)
ylim_min = 0
ylim_max = np.max(y_mono_1) + 20

#### 1. Uniform prior for the normalised Einf
# Add points 1-by-1
for l in range(6):
    x0 = agent1_mono_dose[seq[0:(l+1)]]
    y0 = y_mono_1[seq[0:(l+1)]]
    print('dosage: ', x0, 'response: ', y0)
    data = {'N': len(x0), 'x': x0, 'Y': y0}

    # Train the model and generate samples
    fit = sm1.sampling(data=data, iter=2000, chains=4, warmup=500, thin=1, seed=101, control=dict(adapt_delta=0.9))

    # Extracting traces
    e0, einf, logC, h, sigma = fit['e0'], fit['einf'], fit['logC'], fit['h'], fit['sigma']

    # Plotting regression line
    yy_max = np.repeat(-100, len(x0))
    yy_min = np.repeat(100, len(x0))
    # Plot a subset of sampled regression lines
    fig, ax1 = plt.subplots()
    for i in range(len(e0)):
        yy = e0[i] + (einf[i] * e0[i] - e0[i]) / (1.0 + (np.exp(logC[i]) / np.exp(log_x_plot)) ** h[i])
        ax1.plot(log_x_plot, yy, color='lightsteelblue', alpha=0.005)
        yy_x = e0[i] + (einf[i] * e0[i] - e0[i]) / (1.0 + (np.exp(logC[i]) / x0) ** h[i])
        yy_max = np.maximum(yy_max, yy_x + sigma[i])
        yy_min = np.minimum(yy_min, yy_x - sigma[i])

    for i in range(len(x0)):
        support = np.linspace(yy_min[i], yy_max[i], 100)
        p_x = np.zeros(np.shape(support))
        for j in range(len(e0)):
            delta_support = support - (e0[j] + (einf[j] * e0[j] - e0[j]) / (1.0 + (np.exp(logC[j]) / x0[i]) ** h[j]))
            gaussian = np.exp((-delta_support ** 2.0) / (2 * sigma[j] ** 2.0))
            gaussian *= -1
            p_x += gaussian
        p_x = p_x / np.max(np.abs(p_x)) * 0.5
        ind = np.where(np.abs(p_x)>0.01)[0]
        ax1.plot(p_x[ind] + np.log(x0[i]), support[ind], c='blue', linewidth=1)
        ax1.plot(np.zeros_like(p_x[ind]) + np.log(x0[i]), support[ind], c='blue', linewidth=0.7)

    # Plot data
    ax1.scatter(np.log(x0), y0, c='blue')
    ax1.set_xlim(log_x_min, log_x_max)
    ax1.set_ylim(ylim_min, ylim_max)

    # Plot IC50 (with direct histogram)
    A, B = 1.5, 0.2
    ax2 = ax1.twinx()
    ax2.axis('off')
    ax2.hist(logC, 30, density=True, color='r', alpha=0.5)
    ax2.set_ylim(0, A)
    # Plot Emax (with direct histogram)
    einf_rescaled = einf * e0
    ax3x = ax1.twinx()
    ax3 = ax3x.twiny()
    ax3.hist(einf_rescaled, 30, density=True, color='green', alpha=0.5, orientation="horizontal")
    ax3.set_xlim(0, B)
    ax3.set_ylim(ylim_min, ylim_max)
    ax3x.axis('off')
    ax3.axis('off')

    fig.savefig("figures/fig3/uniform_%d.png" % (l+1), dpi=350)
    plt.show()


#### 2. Beta(0.46, 0.58) prior for the normalised Einf
# Plot the prior
e0 = np.random.normal(loc=100, scale=3, size=10000)
einf = np.random.beta(a=0.46, b=0.58, size=10000)
logC = np.random.uniform(low=np.log(1e-10), high=np.log(1e6), size=10000)
h = np.random.lognormal(mean=0, sigma=1, size=10000)
sigma = np.random.lognormal(mean=0, sigma=1, size=10000)

# Add points 1-by-1
for l in range(6):
    x0 = agent1_mono_dose[seq[0:(l+1)]]
    y0 = y_mono_1[seq[0:(l+1)]]
    print('dosage: ', x0, 'response: ', y0)
    data = {'N': len(x0), 'x': x0, 'Y': y0}

    # Train the model and generate samples
    fit = sm2.sampling(data=data, iter=2000, chains=4, warmup=500, thin=1, seed=101, control=dict(adapt_delta=0.9))

    # Extracting traces
    e0, einf, logC, h, sigma = fit['e0'], fit['einf'], fit['logC'], fit['h'], fit['sigma']

    # Plotting regression line
    yy_max = np.repeat(-100, len(x0))
    yy_min = np.repeat(100, len(x0))
    # Plot a subset of sampled regression lines
    fig, ax1 = plt.subplots()
    for i in range(len(e0)):
        yy = e0[i] + (einf[i] * e0[i] - e0[i]) / (1.0 + (np.exp(logC[i]) / np.exp(log_x_plot)) ** h[i])
        ax1.plot(log_x_plot, yy, color='lightsteelblue', alpha=0.005)
        yy_x = e0[i] + (einf[i] * e0[i] - e0[i]) / (1.0 + (np.exp(logC[i]) / x0) ** h[i])
        yy_max = np.maximum(yy_max, yy_x + sigma[i])
        yy_min = np.minimum(yy_min, yy_x - sigma[i])

    for i in range(len(x0)):
        support = np.linspace(yy_min[i], yy_max[i], 100)
        p_x = np.zeros(np.shape(support))
        for j in range(len(e0)):
            delta_support = support - (e0[j] + (einf[j] * e0[j] - e0[j]) / (1.0 + (np.exp(logC[j]) / x0[i]) ** h[j]))
            gaussian = np.exp((-delta_support ** 2.0) / (2 * sigma[j] ** 2.0))
            gaussian *= -1
            p_x += gaussian
        p_x = p_x / np.max(np.abs(p_x)) * 0.5
        ind = np.where(np.abs(p_x)>0.01)[0]
        ax1.plot(p_x[ind] + np.log(x0[i]), support[ind], c='blue', linewidth=1)
        ax1.plot(np.zeros_like(p_x[ind]) + np.log(x0[i]), support[ind], c='blue', linewidth=0.7)

    # Plot data
    ax1.scatter(np.log(x0), y0, c='blue')
    ax1.set_xlim(log_x_min, log_x_max)
    ax1.set_ylim(ylim_min, ylim_max)

    # Plot IC50 (with direct histogram)
    ax2 = ax1.twinx()
    ax2.axis('off')
    ax2.hist(logC, 30, density=True, color='r', alpha=0.5)
    ax2.set_ylim(0, A)
    # Plot Emax (with direct histogram)
    einf_rescaled = einf * e0
    ax3x = ax1.twinx()
    ax3 = ax3x.twiny()
    ax3.hist(einf_rescaled, 30, density=True, color='green', alpha=0.5, orientation="horizontal")
    ax3.set_xlim(0, B)
    ax3.set_ylim(ylim_min, ylim_max)
    ax3x.axis('off')
    ax3.axis('off')

    fig.savefig("figures/fig3/beta_%d.png" % (l+1), dpi=350)
    plt.show()

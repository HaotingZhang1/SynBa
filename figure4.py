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
import random
import warnings
from scipy import stats
from sklearn.neighbors import KernelDensity
import matplotlib as mpl


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rc('font', size=15) # 12
mpl.rc('xtick', labelsize=15) # 12
mpl.rc('ytick', labelsize=15) # 12
mpl.rcParams['font.family'] = "arial"

warnings.filterwarnings("ignore")

model_mono = """
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

model_combo = """
data {
  int<lower=0> N;
  vector[N] x_1;
  vector[N] x_2;
  vector[N] y;
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
"""


# combination_id = 'ADAM17.AKT'
# cell_line = 'BT-20'
dose_mat = np.array([[100., 97.8, 99.7, 94.7, 95., 92.9],
                     [96.5, 89.8, 97.7, 92.4, 88.4, 87.4],
                     [97.6, 88.8, 94.4, 90.1, 86.9, 88.7],
                     [95.6, 84.2, 86.3, 87.9, 84.9, 80.2],
                     [88.8, 80.2, 84.1, 81.9, 77.8, 76.5],
                     [79.4, 64.4, 69.4, 69.7, 67., 69.4]])
agent1_mono_dose = np.array([0., 1e-2, 3e-2, 0.1, 0.3, 1.])
agent2_mono_dose = np.array([0., 7.5e-1, 2.5, 7.5, 25., 75.])
agent1_mono_dose[0] = 1e-5
agent2_mono_dose[0] = 1e-5
print(dose_mat)
plt.figure()
sns.heatmap(dose_mat, annot=True, xticklabels=False, yticklabels=False, fmt='g')
plt.savefig("figures/fig4/data_heatmap.png", dpi=350)
plt.show()
x_1 = np.repeat(agent1_mono_dose, 6)
x_2 = np.tile(agent2_mono_dose, 6)
y = np.reshape(dose_mat, (36))

data = {'N': len(y), 'x_1': x_1, 'x_2': x_2, 'y': y}

#### Learn the monotherapy model
## Agent 1
y_mono_1 = dose_mat[:, 0]
y_mono_2 = dose_mat[0, :]
sm = pystan.StanModel(model_code=model_mono)

x0 = agent1_mono_dose
y0 = y_mono_1
# x0 = agent2_mono_dose[seq[0:(l+1)]]
# y0 = y_mono_2[seq[0:(l+1)]]
print('dosage: ', x0, 'response: ', y0)
data = {'N': len(x0), 'x': x0, 'Y': y0}

# Train the model and generate samples
fit = sm.sampling(data=data, iter=2000, chains=4, warmup=500, thin=1, seed=101, control=dict(adapt_delta=0.9))

# Extracting traces
e0, einf, logC, h, sigma = fit['e0'], fit['einf'], fit['logC'], fit['h'], fit['sigma']

# Plotting regression line
log_x_min, log_x_max = np.log(1e-5)-0.5, int(np.ceil(np.max(np.log(agent1_mono_dose)))) + 8
log_x_plot = np.linspace(log_x_min, log_x_max, 100)
ylim_min = 0  # np.maximum(0, np.min(y_mono_1) - 50)
ylim_max = np.max(y_mono_1) + 20

yy_max = np.repeat(-100, len(x0))
yy_min = np.repeat(100, len(x0))
# Plot a subset of sampled regression lines
fig, ax1 = plt.subplots()  # plt.figure()
# for i in np.random.randint(0, len(einf), 1000):
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
    ind = np.where(np.abs(p_x) > 0.01)[0]
    ax1.plot(p_x[ind] + np.log(x0[i]), support[ind], c='blue', linewidth=1)
    ax1.plot(np.zeros_like(p_x[ind]) + np.log(x0[i]), support[ind], c='blue', linewidth=0.7)

# Plot data
# plt.scatter(log_x_min + 0.1, 100, c='blue')
ax1.scatter(np.log(x0), y0, c='blue')
# ax1.set_xlabel('$log(x)$')
# ax1.set_ylabel('$y$')
# plt.title('Fitted Regression Line')
ax1.set_xlim(log_x_min, log_x_max)
ax1.set_ylim(ylim_min, ylim_max)

# Plot IC50 (with direct histogram)
ax2 = ax1.twinx()
ax2.axis('off')
ax2.hist(logC, 30, density=True, color='r', alpha=0.5)
ax2.set_ylim(0, 1.5)
# sns.kdeplot(logC, shade=True, ax=ax2)
# Plot Emax (with direct histogram)
einf_rescaled = einf * e0
ax3x = ax1.twinx()
ax3 = ax3x.twiny()
ax3.hist(einf_rescaled, 30, density=True, color='green', alpha=0.5, orientation="horizontal")
ax3.set_xlim(0, 0.2)
ax3.set_ylim(ylim_min, ylim_max)
ax3x.axis('off')
ax3.axis('off')

fig.savefig("figures/fig4/mono_1.png", dpi=350)
plt.show()  # plt.show(block=True)

## Agent 2
x0 = agent2_mono_dose
y0 = y_mono_2
# x0 = agent2_mono_dose[seq[0:(l+1)]]
# y0 = y_mono_2[seq[0:(l+1)]]
print('dosage: ', x0, 'response: ', y0)
data = {'N': len(x0), 'x': x0, 'Y': y0}

# Train the model and generate samples
fit = sm.sampling(data=data, iter=2000, chains=4, warmup=500, thin=1, seed=101, control=dict(adapt_delta=0.9))

# Extracting traces
e0, einf, logC, h, sigma = fit['e0'], fit['einf'], fit['logC'], fit['h'], fit['sigma']

# Plotting regression line
log_x_min, log_x_max = np.log(1e-5)-0.5, int(np.ceil(np.max(np.log(agent1_mono_dose)))) + 8
log_x_plot = np.linspace(log_x_min, log_x_max, 100)
ylim_min = 0  # np.maximum(0, np.min(y_mono_1) - 50)
ylim_max = np.max(y_mono_1) + 20

yy_max = np.repeat(-100, len(x0))
yy_min = np.repeat(100, len(x0))
# Plot a subset of sampled regression lines
fig, ax1 = plt.subplots()  # plt.figure()
# for i in np.random.randint(0, len(einf), 1000):
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
    ind = np.where(np.abs(p_x) > 0.01)[0]
    ax1.plot(p_x[ind] + np.log(x0[i]), support[ind], c='blue', linewidth=1)
    ax1.plot(np.zeros_like(p_x[ind]) + np.log(x0[i]), support[ind], c='blue', linewidth=0.7)

# Plot data
# plt.scatter(log_x_min + 0.1, 100, c='blue')
ax1.scatter(np.log(x0), y0, c='blue')
# ax1.set_xlabel('$log(x)$')
# ax1.set_ylabel('$y$')
# plt.title('Fitted Regression Line')
ax1.set_xlim(log_x_min, log_x_max)
ax1.set_ylim(ylim_min, ylim_max)

# Plot IC50 (with direct histogram)
ax2 = ax1.twinx()
ax2.axis('off')
ax2.hist(logC, 30, density=True, color='r', alpha=0.5)
ax2.set_ylim(0, 1.5)
# sns.kdeplot(logC, shade=True, ax=ax2)
# Plot Emax (with direct histogram)
einf_rescaled = einf * e0
ax3x = ax1.twinx()
ax3 = ax3x.twiny()
ax3.hist(einf_rescaled, 30, density=True, color='green', alpha=0.5, orientation="horizontal")
ax3.set_xlim(0, 0.2)
ax3.set_ylim(ylim_min, ylim_max)
ax3x.axis('off')
ax3.axis('off')

fig.savefig("figures/fig4/mono_2.png", dpi=350)
plt.show()  # plt.show(block=True)


#### Learn the combination model
# Plot the prior
np.random.seed(123)
e0 = np.random.normal(loc=100, scale=3, size=100000)
e1 = np.random.beta(a=0.46, b=0.58, size=100000)
e2 = np.random.beta(a=0.46, b=0.58, size=100000)
e3 = np.random.beta(a=0.46, b=0.58, size=100000)
delta_hsa = e0 * (np.minimum(e1, e2) - e3)
alpha = np.random.lognormal(mean=0, sigma=1, size=100000)

# plt.hist2d(delta_hsa, np.log(alpha), bins=50)
sns.kdeplot(x=np.log(alpha), y=delta_hsa, cmap="Blues", fill=True, # shade=True, shade_lowest=False,
            thresh=0, clip=((-5, 5), (-100, 100)))
# sns.lineplot(x=[-5, 5], y=[0, 0], color='r', linewidth=1.5, linestyle='--')
sns.lineplot(x=[-5, 5], y=[0, 0], color='r', linewidth=1.5, linestyle='--')  # 5 is a popular threshold for Delta(HSA)
sns.lineplot(x=[0, 0], y=[-100, 100], color='r', linewidth=1.5, linestyle='--', estimator=None)
# plt.xlabel(r'$\log(\alpha)$')
# plt.ylabel(r'$\Delta(HSA)$')
plt.xlim([-3, 3])
plt.ylim([-50, 50])
plt.savefig("figures/fig4/synergy_prior.png", dpi=350)
plt.show()

# Compile model
sm = pystan.StanModel(model_code=model_combo)

# Include all points
data = {'N': len(y), 'x_1': x_1, 'x_2': x_2, 'y': y}
assert len(y) == 36

# Train the model and generate samples
fit = sm.sampling(data=data, iter=2000, chains=4, warmup=500, thin=1, seed=101, control=dict(adapt_delta=0.9))

# Plot the posterior
e0, e1, e2, e3, alpha = fit['e_0'], fit['e_1'], fit['e_2'], fit['e_3'], fit['alpha']
delta_hsa = e0 * (np.minimum(e1, e2) - e3)

# plt.hist2d(delta_hsa, np.log(alpha), bins=50)
sns.kdeplot(x=np.log(alpha), y=delta_hsa, cmap="Blues", fill=True, # shade=True, shade_lowest=False,
            thresh=0, clip=((-5, 5), (-100, 100)))
# sns.lineplot(x=[-5, 5], y=[0, 0], color='r', linewidth=1.5, linestyle='--')
sns.lineplot(x=[-5, 5], y=[0, 0], color='r', linewidth=1.5, linestyle='--')  # 5 is a popular threshold for Delta(HSA)
sns.lineplot(x=[0, 0], y=[-100, 100], color='r', linewidth=1.5, linestyle='--', estimator=None)
# plt.xlabel(r'$\log(\alpha)$')
# plt.ylabel(r'$\Delta(HSA)$')
plt.xlim([-3, 3])
plt.ylim([-40, 40]) # plt.ylim([-50, 50])
plt.savefig("figures/fig4/synergy_posterior.png", dpi=350)
plt.show()

plt.plot()
delta_hsa = 100 * (np.minimum(e1[1000:-1], e2[1000:-1]) - e3[1000:-1])
val, _, _ = plt.hist(delta_hsa, bins=30, range=(-100, 100), density=True, histtype='step')
plt.plot([0, 0], [0, np.max(val)], color='r')
plt.savefig("figures/fig4/hsa_posterior_1.png", dpi=350)
plt.show()

plt.plot()
delta_hsa = 100 * (np.minimum(e1[1000:-1], e2[1000:-1]) - e3[1000:-1])
val, _, _ = plt.hist(delta_hsa, bins=30, range=(-100, 100), histtype='step')
mask = (delta_hsa > 0)
val, _, _ = plt.hist(delta_hsa[mask], bins=15, range=(0, 100), histtype='bar', color='aliceblue', lw=0)
plt.plot([0, 0], [0, np.max(val)], color='r')
plt.savefig("figures/fig4/hsa_posterior_2.png", dpi=350)
plt.show()

print('P(Delta HSA>0) =', len(np.where(delta_hsa > 0)[0])/len(delta_hsa))

plt.plot()
alpha_vec = alpha[1000:-1]
val, _, _ = plt.hist(np.log(alpha_vec), bins=30, range=(-2, 3), density=True, histtype='step')
plt.plot([0, 0], [0, np.max(val)], color='r')
plt.xlim(-2.1, 3)
plt.savefig("figures/fig4/alpha_posterior_1.png", dpi=350)
plt.show()

plt.plot()
alpha_vec = alpha[1000:-1]
val, _, _ = plt.hist(np.log(alpha_vec), bins=30, range=(-2, 3), histtype='step')
mask = (np.log(alpha_vec) > 0)
plt.hist(np.log(alpha_vec[mask]), bins=18, range=(0, 3), histtype='bar', color='aliceblue', lw=0)
# plt.plot([1, 1], [0, np.max(val)], color='r')
plt.plot([0, 0], [0, np.max(val)], color='r')
plt.xlim(-2.1, 3)
plt.savefig("figures/fig4/alpha_posterior_2.png", dpi=350)
plt.show()

print('P(alpha>1) =', len(np.where(alpha_vec > 1)[0])/len(alpha_vec))

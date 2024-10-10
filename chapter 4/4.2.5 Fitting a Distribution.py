"""
Program to do a maximum likelihood
estimate (MLE) with generalised
extreme value (GEV) distribution
"""

from scipy.stats import genextreme
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt


# Fix random seed
np.random.seed(21)


# Likelihood function (the GEV distribution)
def likelihood_fun(theta, data):
    lik = -np.sum(
        np.log(
            genextreme.pdf(x=data, c=theta)
        )
    )
    return lik


# Loss function (to be maximised)
def fit_distribution(data):
    x0 = np.random.rand(1, )
    estimated_params = fmin(func=likelihood_fun,
                            x0=x0,
                            disp=True,
                            args=(data,))
    return estimated_params


# Generate synthetic data
c = -0.2  # True parameter for GEV distribution
rvs = genextreme.rvs(c, size=1000)  # Random variates

# Fit the parameters
c_est = fit_distribution(data=rvs)  # Estimated parameter

# Print results
print('Obs. c = {:.4f}'.format(c))
print('Est. c = {:.4f}'.format(c_est[0]))

# Analysis
x_obs = np.linspace(genextreme.ppf(0.01, c),
                    genextreme.ppf(0.99, c), 100)
x_est = np.linspace(genextreme.ppf(0.01, c_est),
                    genextreme.ppf(0.99, c_est), 100)
rvs_ = genextreme.rvs(c_est, size=1000)

# Display the estimate
fig, ax = plt.subplots()
ax.plot(x_obs, genextreme.pdf(x_obs, c),
        'k-', lw=2, alpha=0.9, label='Obs: GEV pdf')
ax.plot(x_est, genextreme.pdf(x_est, c_est),
        'r-', lw=5, alpha=0.4, label='Fit: GEV pdf')
ax.hist(rvs, color='k', density=True, bins='auto',
        histtype='step', alpha=0.7)
ax.hist(rvs_, color='r', density=True, bins='auto',
        histtype='step', alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('$f(x)$')
ax.grid(ls='--')
ax.legend(loc='best', frameon=False)
plt.show()
plt.tight_layout()
plt.savefig('MLE_fit.pdf', dpi=300)

"""
Monte Carlo error propagation
on a simple forward model:
Manning's equation
V = (1/n) * R^(2/3) * S^(1/2),
where,
V is the flow velocity,
n is the Manning's roughness coefficient,
R is the hydraulic radius, and
S is the channel slope
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
np.random.seed(11)


def compute_velocity(n=1., b=1., h=1., S=1.):
    R = (b * h) / (2 * b + h)
    V = (1 / n) * R ** (2 / 3) * S ** (1 / 2)
    return V


"""
Monte-Carlo simulation #1
"""
Nsim = 100000
n_var = stats.halfnorm.rvs(loc=0.030, scale=1e-2, size=Nsim)  # Manning's coef, loc=mean; scale=std
b_var = stats.norm.rvs(loc=2.0, scale=1e-1, size=Nsim)  #
h_var = stats.norm.rvs(loc=1.3, scale=1e-1, size=Nsim)
S_var = stats.norm.rvs(loc=np.tan(15 * np.pi / 180),
                       scale=1e-2, size=Nsim)
data = np.vstack([n_var, b_var, h_var, S_var]).T  # make computations easy

V_var = np.zeros_like(n_var)
for i in range(Nsim):
    V_var[i] = compute_velocity(
        n=data[i, 0],
        b=data[i, 1],
        h=data[i, 2],
        S=data[i, 3]
    )


def normal_distribution(x, mu, sig):
    rv = 1.0 / (sig * np.sqrt(2.0 * np.pi))
    normal_values = rv * np.exp(
        -(x - mu) ** 2.0 / (2.0 * sig ** 2)
    )
    return normal_values


# Plots
xmin, xmax = V_var.min(), V_var.max()
x = np.linspace(xmin, xmax, Nsim)
y = normal_distribution(x, np.mean(V_var), np.std(V_var))

plt.hist(V_var, density=True, bins=150,
         label="$V_{" + "MC" + "}$")
plt.plot(x, y, color='red', lw=3, label="Normal")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Relative Probability")
plt.xlim(xmin, xmax)
plt.legend()
plt.grid(ls="--")
plt.tight_layout()
plt.savefig("mc1.pdf", dpi=300)

"""
Monte-Carlo simulation #2
"""
Nsim = 100000
n_var = stats.halfnorm.rvs(loc=0.030, scale=1e-1, size=Nsim)
b_var = stats.norm.rvs(loc=2.0, scale=1e-1, size=Nsim)
h_var = stats.norm.rvs(loc=1.3, scale=1e-1, size=Nsim)
S_var = stats.norm.rvs(loc=np.tan(15 * np.pi / 180),
                       scale=1e-2, size=Nsim)
data = np.vstack([n_var, b_var, h_var, S_var]).T

V_var = np.zeros_like(n_var)
for i in range(Nsim):
    V_var[i] = compute_velocity(
        n=data[i, 0],
        b=data[i, 1],
        h=data[i, 2],
        S=data[i, 3])

# Plots
xmin, xmax = V_var.min(), V_var.max()
x = np.linspace(xmin, xmax, Nsim)
y = normal_distribution(x, np.mean(V_var), np.std(V_var))

plt.figure()
plt.hist(V_var, density=True, bins=150,
         label="$V_{" + "MC" + "}$")
plt.plot(x, y, color='red', lw=3, label="Normal")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Relative Probability")
plt.xlim(xmin, xmax)
plt.legend()
plt.grid(ls="--")
plt.tight_layout()
plt.savefig("mc2.pdf", dpi=300)

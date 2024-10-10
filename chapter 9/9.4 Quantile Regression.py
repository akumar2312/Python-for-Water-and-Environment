"""
Program to do Quantile regression
on synthetic dataset.
"""

from sklearn.linear_model import QuantileRegressor
import numpy as np
import matplotlib.pyplot as plt


"""
Generate synthetic normal data
"""
np.random.seed(21)
x = np.arange(start=0.5 * np.pi, stop=2 * np.pi, step=0.05)
X = x.reshape(x.shape[0], -1)
mean_ = 10.0 + 0.1 * np.sin(x)
scale_ = 0.02 + 0.02 * np.abs(x)
y_true = np.random.normal(loc=mean_,
                          scale=scale_,
                          size=x.shape[0])


"""
Quantile regression
"""
idx_up = 0
idx_dn = 0
x_good = list()
y_good = list()

preds = dict()
x_outlier = list()
y_outlier = list()

qnt = [0.1, 0.25, 0.5, 0.75, 0.9]
for q in qnt:
    qreg = QuantileRegressor(quantile=q,
                             alpha=0,
                             solver='highs')
    qreg.fit(X=X, y=y_true)
    preds[q] = qreg.predict(X)

    # Segregate the detected outliers using index
    if q == qnt[0]:
        idx_dn = preds[q] > y_true

    elif q == qnt[-1]:
        idx_up = preds[q] < y_true

idx_outlier = np.logical_or(idx_up, idx_dn)
x_outlier.append(x[idx_outlier])
y_outlier.append(y_true[idx_outlier])
x_outlier = np.array(x_outlier).ravel()
y_outlier = np.array(y_outlier).ravel()

idx_good = ~np.logical_or(idx_up, idx_dn)
x_good.append(x[idx_good])
y_good.append(y_true[idx_good])
x_good = np.array(x_good).ravel()
y_good = np.array(y_good).ravel()

"""
Plot the scatter and then show
fit of the data on the scatter
"""
plt.figure(figsize=[8, 6])
# Quantile plots
plt.plot(X, y_true,
         color="black",
         linestyle="dashed",
         label="True y")
for q, y_pred in preds.items():
    plt.plot(X, y_pred, label="Q={}".format(q))

# Outliers data plot
plt.scatter(x=x_outlier, y=y_outlier, marker='o',
            alpha=0.5, color='k',
            label='Outside: Q={} and Q={}'.format(qnt[0],
                                                 qnt[-1]))

# Inside data plot
plt.scatter(x=x_good, y=y_good, marker='x',
            alpha=0.5, color='b',
            label='Within: Q={} and Q={}'.format(qnt[0],
                                                qnt[-1]))

plt.xlabel('x')
plt.ylabel('y')
plt.ylim([9.5, 10.8])
plt.grid(ls='--')
plt.legend(loc='upper center', ncol=3)
plt.show()
plt.tight_layout()
plt.savefig('quantile_reg.pdf', dpi=300)

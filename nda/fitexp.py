import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize


def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C


def tst():
    noisy_y = np.load('/homecentral/srao/Documents/code/mypybox/nda/data/long_tau_vs_ac_matalphaVsCorlength_EI.npy')
    noisy_y = noisy_y[0, 8, 8:200]
    t = np.arange(noisy_y.size)
    A, K, C = fit_exp_nonlinear(t, noisy_y)
    fit_y = model_func(t, A, K, C)
    return A, K, C

# plt.ion()
# plt.show()
# plt.waitforbuttonpress()

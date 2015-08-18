import numpy as np
import pylab as plt
import sys, code


def zk(tc):
    atTheta = np.arange(0, 180.0, 22.5)
    n, _ = tc.shape
    for i in np.arange(n):
        firingRate = tc[i, :]
        zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return zk

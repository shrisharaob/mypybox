import numpy as np
def GenSpkTrain(nSpks, rate):
    return np.random.poisson(rate, ( nSpks,))

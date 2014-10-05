import numpy as np
import pylab as plt

def CircVar(firingRate, atTheta):
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(firingRate)


if __name__ == "__main__":
    N = 19600
    tc = np.load('tuningCurves_anatomic.npy')
    theta = np.arange(0, 360, 45)
    circVariance = np.zeros((N,))
    for i in np.arange(N):
        circVariance[i] = CircVar(tc[i, :], theta)
        
    plt.hist(circVariance, 25, fc = 'k', edgecolor = 'w')
    plt.xlabel('Circular vaiance')
    plt.ylabel('Neuron count')
    plt.title('NE = NI = 1.96E4, K = 2E3, C = 100, KSI = 1.2')
    plt.savefig('ori_cvDistr_anatomic.png')
    

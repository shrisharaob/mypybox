import numpy as np
import pylab as plt
import sys, code

def CircVar(firingRate, atTheta):
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(firingRate)

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return 


if __name__ == "__main__":
    N = 19600
    tc = np.load('tuningCurves_bidirEE.npy')
    theta = np.arange(0, 360, 45)
    circVariance = np.zeros((N,))
    for i in np.arange(N):
        circVariance[i] = CircVar(tc[i, :], theta)

    print "here"
    keyboard()
    circVariance = circVariance[np.logical_not(np.isnan(circVariance))]
    plt.hist(circVariance, 25, fc = 'k', edgecolor = 'w')
    plt.xlabel('Circular vaiance')
    plt.ylabel('Neuron count')
    plt.title('NE = NI = 1.96E4, K = 2E3, C = 100, KSI = 1.2')
    plt.savefig('ori_cvDistr_bidirEE.png')
    

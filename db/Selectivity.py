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
    NE = 10000
    NI = 10000
#    tc = np.load('tuningCurves_bidirEE.npy')
#    tc = np.load('tuningCurves_bidirII_a0t3T3xi12tr15.npy')
    dbName  = 'a0t3T3xi12tr15'
    dbName = sys.argv[1]
    tc = np.load('tuningCurves_' + dbName + '.npy')
    print tc.shape
    theta = np.arange(0, 180, 22.5)
    theta = np.arange(0, 360, 45.0)
    circVariance = np.zeros((NE + NI,))
    neuronIdx = np.arange(NE + NI)
    for i, kNeuron in enumerate(neuronIdx):
        print kNeuron
        circVariance[i] = CircVar(tc[kNeuron, :], theta)

    np.save('Selectivity_' + dbName, circVariance)
    cvE = circVariance[:NE]
    cvI = circVariance[NE:]
    cvE = cvE[np.logical_not(np.isnan(cvE))]
    cvI = cvE[np.logical_not(np.isnan(cvI))]
#    circVariance = circVariance[np.logical_not(np.isnan(circVariance))]

    # PLOT
    plt.hist(cvE, 25, fc = 'k', edgecolor = 'w')
    plt.xlabel('Circular vaiance, E neurons', fontsize = 20)
    plt.ylabel('Neuron count', fontsize = 20)
    plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0, \; \xi = 1.2$', fontsize = 20)
    filename = 'ori_cvDistr_E_' + dbName 
    plt.savefig(filename)
    plt.clf()
    plt.hist(cvI, 25, fc = 'k', edgecolor = 'w')
    plt.xlabel('Circular vaiance, E neurons', fontsize = 20)
    plt.ylabel('Neuron count', fontsize = 20)
    #plt.title('NE = NI = 1.96E4, K = 2E3, C = 100, KSI = 1.2')
    plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0, \; \xi = 1.2$', fontsize = 20)
    filename = 'ori_cvDistr_I_' + dbName 
    plt.savefig(filename)

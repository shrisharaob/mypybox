import numpy as np
import pylab as plt
import sys, code
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

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
#    dbName  = 'a0t3T3xi12tr15'
    dbName = sys.argv[1]
    tc = np.load('./data/tuningCurves_' + dbName + '.npy')
    print tc.shape
    theta = np.arange(0, 180, 22.5)
#    theta = np.arange(0, 360, 45.0)
    circVariance = np.zeros((NE + NI,))
    neuronIdx = np.arange(NE + NI)
    for i, kNeuron in enumerate(neuronIdx):
#        print kNeuron
        circVariance[i] = CircVar(tc[kNeuron, :], theta)

    np.save('./data/Selectivity_' + dbName, circVariance)
    cvE = circVariance[:NE]
    cvI = circVariance[NE:]
    cvE = cvE[np.logical_not(np.isnan(cvE))]
    cvI = cvI[np.logical_not(np.isnan(cvI))]
#    circVariance = circVariance[np.logical_not(np.isnan(circVariance))]

#    keyboard()
    # PLOT
    f00 = plt.figure();
    plt.ioff()
    plt.hist(cvE, 25, fc = 'k', edgecolor = 'w')
    plt.xlabel('Circular vaiance, E neurons', fontsize = 20)
    plt.ylabel('Neuron count', fontsize = 20)
#    plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0, \; \xi = 1.2$', fontsize = 20)
 #   plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0$', fontsize = 20)
    filename = 'ori_cvDistr_E_' + dbName 
#    figFolder = '/homecentral/srao/Documents/cnrs/figures/feb23/'
    figFolder = '/homecentral/srao/Documents/code/mypybox/db/figs/'
#    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [5.0, 4.26])
    kb.keyboard()
    labels = [item.get_text() for item in f00.gca().get_xticklabels()]
    labels[0] = "0.0\nhighly tuned"
    labels[-1] = "1.0\nnot selective"
    f00.gca().set_xticklabels(labels)
    plt.draw()

    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [6.0, 5.26])

    
#    plt.savefig(filename)
#    plt.waitforbuttonpress()
    plt.clf()
    plt.ioff()
    plt.hist(cvI, 25, fc = 'k', edgecolor = 'w')
    plt.xlabel('Circular vaiance, I neurons') 
    plt.ylabel('Neuron count') 
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    tmpLabels = labels
    print labels
    labels[0] = "0.0\nhighly tuned"
    labels[-1] = "1.0\nnot selective"

    print tmpLabels
    plt.gca().set_xticklabels(labels)
    plt.draw()
    #plt.title('NE = NI = 1.96E4, K = 2E3, C = 100, KSI = 1.2')
#    plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0, \; \xi = 1.2$', fontsize = 20)
#    plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0$', fontsize = 20)
#    plt.title(r'NE = NI = 1E4, K = 1E3', fontsize = 20)
#    plt.title(r'NI = 1E4, K = 2E3', fontsize = 20)
    filename = 'ori_cvDistr_I_' + dbName 
    #Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=8, labelFontsize = 8, titleSize = 8,  paperSize = [4.26, 3.46])

    kb.keyboard()
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [4.26, 3.26])
    #plt.savefig(filename)
    print figFolder + filename
    # plt.ion()
    # plt.show()
    # plt.waitforbuttonpress()

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
    NE = 40000
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
#    plt.hist(cvE, 25, fc = 'k', edgecolor = 'w')
    cveCnt, cvEbins = np.histogram(cvE, 26)
    plt.plot(cvEbins[:-1], cveCnt, 'k.-', label = 'E')

    
#    plt.xlabel('Circular vaiance, E neurons', fontsize = 20)
    plt.ylabel('Neuron count', fontsize = 20)
#    plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0, \; \xi = 1.2$', fontsize = 20)
 #   plt.title(r'NE = NI = 1E4, K = 1E3, C = 100, $\alpha = 0.0$', fontsize = 20)
    filename = 'ori_cvDistr_E_' + dbName 
#    figFolder = '/homecentral/srao/Documents/cnrs/figures/feb23/'
    figFolder = '/homecentral/srao/Documents/code/mypybox/db/figs/'
#    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [5.0, 4.26])
 #   kb.keyboard()
    # labels = [item.get_text() for item in f00.gca().get_xticklabels()]
    # labels[0] = "0.0\nhighly tuned"
    # labels[-1] = "1.0\nnot selective"
    # f00.gca().set_xticklabels(labels)
    # plt.draw()

#    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [4.26/1.3, 3.26/1.3])

    
#    plt.savefig(filename)
#    plt.waitforbuttonpress()
    #plt.clf()

#    plt.hist(cvI, 25, fc = 'k', edgecolor = 'w')
    cviCnt, cvIbins = np.histogram(cvI, 26)
    plt.plot(cvIbins[:-1], cviCnt, 'r.-', label = 'I')
    plt.xlabel('Circular vaiance') 
    plt.ylabel('Neuron count')
    plt.title('Distribution of circular variance')
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    tmpLabels = labels
    print labels
    labels[0] = "0.0\nhighly tuned"
    labels[-1] = "1.0\nnot selective"

    print tmpLabels
    plt.gca().set_xticklabels(labels)
    plt.draw()
    filename = 'ori_cvDistr_EI_' + dbName
    plt.legend()
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [8.0/1.8, 6.26/1.8])

    plt.clf()
    plt.plot(cvEbins[:-1], cveCnt / float(cveCnt.sum()), 'k.-', label = 'E')
    plt.plot(cvIbins[:-1], cviCnt / float(cviCnt.sum()), 'r.-', label = 'I')
    plt.xlabel('Circular vaiance') 
    plt.ylabel('Normalized count')
    plt.title('Distribution of circular variance')
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    tmpLabels = labels
    print labels
    labels[0] = "0.0\nhighly tuned"
    labels[-1] = "1.0\nnot selective"
    plt.legend()
    filename = 'ori_cvDistr_EI_normalized_' + dbName
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [8.0/1.8, 6.26/1.8])
    #plt.savefig(filename)
    print figFolder + filename
    # plt.ion()
    # plt.show()
    # plt.waitforbuttonpress()

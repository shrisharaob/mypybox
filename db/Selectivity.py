import numpy as np
import pylab as plt
import sys, code
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
#import Keyboard as kb

def CircVar(firingRate, atTheta):
    out = np.nan
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    if(firingRate.mean() > 0.0):
        out = 1 - np.absolute(zk) / np.sum(firingRate)
    return out

def zk(tc):
    atTheta = np.arange(0, 180.0, 22.5)
    n, _ = tc.shape
    for i in np.arange(n):
        firingRate = tc[i, :]
        zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return zk

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
#    NE = 40000
    NE = 20000
    NI = 20000
#    tc = np.load('tuningCurves_bidirEE.npy')
#    tc = np.load('tuningCurves_bidirII_a0t3T3xi12tr15.npy')
#    dbName  = 'a0t3T3xi12tr15'
    dbName = sys.argv[1]
    print './data/tuningCurves_' + dbName + '.npy'
    tc = np.load('./data/tuningCurves_' + dbName + '.npy')
    print tc.shape
    theta = np.arange(0, 180, 22.5)
#    theta = np.array([0., 22.5, 45., 56.25, 67.5, 90. , 112.5, 123.75, 135., 157.5])
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
#    figFolder = '/homecentral/srao/Documents/code/mypybox/db/figs/'
    figFolder = '/homecentral/srao/cuda/data/poster/figs/'    
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
    plt.legend(loc=0)
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [8.0/1.8, 6.26/1.8])
    plt.clf()
    plt.plot(cvEbins[:-1], cveCnt / float(cveCnt.sum()), 'k.-', label = 'E (%.4s)'%(cvE.mean()))
    plt.plot(cvIbins[:-1], cviCnt / float(cviCnt.sum()), 'r.-', label = 'I (%.4s)'%(cvI.mean()))
    plt.xlabel('Circular vaiance') 
    plt.ylabel('Normalized count')
    plt.title('Distribution of circular variance')
    plt.xlim((0, 1.0))
    plt.draw()
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    xticks = plt.gca().get_xticks()
    tmp = int(len(labels) / 2.0)
    labels[tmp] = xticks[tmp]
    labels[0] =  "%s\nhighly\ntuned"%(xticks[0])
    for ii, i in enumerate(np.arange(.2, .9, .2)):
        labels[ii+1] = '%s'%(i)
    labels[-1] =  "%s\nnot\nselective"%(xticks[-1])

    
    # tmpLabels = labels
    # print labels
    # labels[0] = "0.0\nhighly tuned"
    # labels[-1] = "1.0\nnot selective"

    print labels
    plt.gca().set_xticklabels(labels)

    plt.gca().xaxis.set_label_coords(0.5, -0.1);
    
    plt.legend(loc=0)
    filename = 'ori_cvDistr_EI_normalized_' + dbName
#    kb.keyboard()
#    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', tickFontsize=10, labelFontsize = 10, titleSize = 10,  paperSize = [8.0/1.8, 6.26/1.8])
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.142, 0.15, .77, .74])

#    pageSizePreFactore = 1.
 #   Print2Pdf(plt.gcf(),  figFolder + filename, [pageSizePreFactore*5.25,  pageSizePreFactore*4.], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.125, 0.15, .78, .75])

    
    #plt.savefig(filename)
    print figFolder + filename
    # plt.ion()
    # plt.show()
    # plt.waitforbuttonpress()

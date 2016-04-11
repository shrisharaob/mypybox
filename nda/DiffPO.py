import numpy as np
import code
import sys
import pylab as plt
from multiprocessing import Pool
from functools import partial
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
from collections import defaultdict
import GetPO
#import statsmodels.api as sm
from scipy.stats import uniform
import scipy.stats as stats

def GetPreSynaptic(nPostNeurons, idxVec, sparseConvec):
    preNeurons = defaultdict(list)
    for kNeuron in range(nPostNeurons.size):
        kPostNeurons = sparseConvec[idxVec[kNeuron] : idxVec[kNeuron] + nPostNeurons[kNeuron]]
        for n in kPostNeurons:
            preNeurons[n].append(kNeuron)
    return preNeurons

def POdiffHist(PO, POPresynaptic, titleText):
    print POPresynaptic.size,  PO.size
#    plt.hist(POPresynaptic, 25, normed = 1)
    stats.probplot(POPresynaptic / np.pi, dist='uniform', fit=False,plot=plt)
    plt.gca().lines.remove((plt.gca().get_lines())[1])
    xxlim = np.max([plt.gca().get_xlim()[1], plt.gca().get_ylim()[1]])
#    minXY = xxlim = np.min([plt.gca().get_xlim()[0], plt.gca().get_ylim()[0]])
    xx = np.linspace(0, xxlim)
    # plt.xlim(minXY, xxlim)
    # plt.ylim(minXY, xxlim)
    plt.plot(xx, xx, 'r')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('p = 0.8, PO = %.3srad'%(PO) + ' cv = ' + titleText)
#    plt.title('control, PO = %.5srad'%(PO) + ' cv = ' + titleText)    
    
#    plt.hist(POPresynaptic - PO, 25, normed = 1)
    
def POdiffCosHist(PO, POPresynaptic, titleText):
    print POPresynaptic.size,  PO.size
    plt.hist(np.cos(2 * (POPresynaptic - PO)), 25, normed = 1)
    plt.title('Control, PO = %.5srad'%(PO) + ' cv = ' + titleText)
    plt.show()

#    plt.gca().lines.remove((plt.gca().get_lines())[1])
 #   xxlim = np.max([plt.gca().get_xlim()[1], plt.gca().get_ylim()[1]])
#    minXY = xxlim = np.min([plt.gca().get_xlim()[0], plt.gca().get_ylim()[0]])

    # plt.xlim(minXY, xxlim)
    # plt.ylim(minXY, xxlim)
    




if __name__ == '__main__':
    [foldername, tcfilename, NE, NI] = DefaultArgs(sys.argv[1:], ['', '', 20000, 20000])
    NE = int(NE)
    NI = int(NI)
    foldername = '/homecentral/srao/cuda/data/poster/' + foldername + '/'
#    foldername = '/homecentral/srao/cuda/' + foldername + '/'    
    datDtype = np.int32
    idxVec = np.zeros((NE + NI, ), dtype = datDtype)
    nPostNeurons = np.zeros((NE + NI, ), dtype = datDtype)
    # requires = ['CONTIGUOUS', 'ALIGNED']
    # idxVec = np.require(idxVec, datDtype, requires)
    # nPostNeurons = np.require(nPostNeurons, datDtype, requires);
    fpIdxVec = open(foldername + 'idxVec.dat', 'rb')
    fpNpostNeurons = open(foldername + 'nPostNeurons.dat', 'rb')
    fpsparsevec = open(foldername + 'sparseConVec.dat', 'rb')
    idxVec = np.fromfile(fpIdxVec, dtype = datDtype)
    nPostNeurons = np.fromfile(fpNpostNeurons, dtype = datDtype)
    sparseVec = np.zeros(shape = (nPostNeurons.sum(), ), dtype = datDtype)
#    sparseVec = np.require(sparseVec, datDtype, requires)
    sparseVec = np.fromfile(fpsparsevec, dtype = datDtype)
    fpIdxVec.close()
    fpNpostNeurons.close()
    fpsparsevec.close()
    print nPostNeurons.size, idxVec.size, sparseVec.size
    print nPostNeurons[:10]    
    
    preNeurons = GetPreSynaptic(nPostNeurons, idxVec, sparseVec)

    tc = np.load('/homecentral/srao/db/data/tuningCurves_' + tcfilename + '.npy')
    cv = np.load('/homecentral/srao/db/data/Selectivity_' + tcfilename + '.npy')
    print tc.shape
    po = GetPO.POofPopulation(tc)
    feedForwardPO = np.loadtxt('/homecentral/srao/cuda/randnDelta.csv')
    vIdx = np.arange(NE, NE+NI)[cv[NE:] < 0.6]
#    randIdx = np.random.randint(NE, NE + NI, 25)
    randIdx = np.random.choice(vIdx, 25)
#    plt.ion()
    print po.size, feedForwardPO.size
    for m, mNeuron in enumerate(randIdx):

        # print len(preNeurons[m])
        titleText = '%.4s'%(cv[mNeuron])
        # POdiffHist(po[m], feedForwardPO[preNeurons[m]], titleText)
        # filename = 'diffPO_%s'%(mNeuron)
        # Print2Pdf(plt.gcf(),  foldername + '/figs/' + filename,  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.15, 0.15, .77, .74])
        plt.ion()
        POdiffCosHist(po[m], feedForwardPO[preNeurons[m]], titleText)
        plt.waitforbuttonpress()
        plt.clf()

    
        



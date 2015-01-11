#script to compute fano factor as a function of bi-directional connectivity alpha
import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys, os
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
from enum import Enum
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append("/homecentral/srao/Documents/code/mypybox/nda/spkStats")
import FanoFactorDynamics as ffd
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig





def FanoInIntervalForAllTheta(dbName, spkTimeStart, spkTimeEnd, nTrials, neuronId):
    avgSpkCnt = np.zeros((nTrials, ))
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    fanoFactor = np.empty((1, ))
    fanoFactor[:] = np.nan
    print "neuronId", neuronId
    thetas = np.arange(0., 180., 22.5)
    ffvsTheta = np.empty((thetas.size, ))
    thetas = thetas.astype(int)
    for mm, mTheta in enumerate(thetas):
        for kTrial in np.arange(nTrials):
            kTheta = int((kTrial+1) * 1000) + mTheta
            nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND spkTimes > %s AND spkTimes < %s ", (neuronId, kTheta, float(spkTimeStart), float(spkTimeEnd)))
            avgSpkCnt[kTrial] = float(nSpks)
        ffvsTheta[mm] = np.var(avgSpkCnt) / np.mean(avgSpkCnt) 
    dbCursor.close()
    db.close()
    return  ffvsTheta

if __name__ == "__main__":
    alpha = np.array([0.0])
    NE = 10000
    NI = 10000
    simDuration = 3000
    spkTimeStart = 2000.0
    spkTimeEnd = 3000.0
    simDT = 0.05
    tau = 3.0
    binSize = 50.0  # in ms fano factor observation window
    print alpha
    [dbName, computeType, nTrials, alpha, simDuration, simDT, NE, NI, tau] = DefaultArgs(sys.argv[1:], ['', 'plot', 16, alpha, simDuration, simDT, NE, NI, tau])
    thetas = np.arange(0., 180., 22.5)
    if(computeType == 'compute'):
        neuronsList = np.arange(NE + NI)
        ffMat = np.empty((NE+NI, len(thetas))) # matrix N_NEURONS-by-theta with each element containing the fano factor 
        ffMat[:] = np.nan
        func = partial(FanoInIntervalForAllTheta, dbName, spkTimeStart, spkTimeEnd, nTrials)
        p = Pool(8)
        result = p.map(func, neuronsList) 
        p.close()
        ffMat = np.asarray(result)
        filename = os.path.splitext(sys.argv[0])[0]
        np.save(filename + '_' + dbName, ffMat)

    #     for ll, lTheta in enumerate(thetas):
    #         print "alpha = ", lTheta
    #         out = result[ll]
    # #        out = ffunc(lTheta)
    #         meanSpkCnt = out[0][:, 0]
    #         spkVar = out[0][:, 1]
    #         tmpidx = ~(np.isnan(meanSpkCnt))
    #         tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
    #         meanSpkCnt = meanSpkCnt[tmpidx]
    #         spkVar = spkVar[tmpidx]
    #         ff[tmpidx] = spkVar / meanSpkCnt
    #         ffMat[:, ll] = ff

        kb.keyboard()

    else:
        print "plotting ", "here"
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/tuningCurves_bidirII_%s.npy'%((dbName, )))
        filename = os.path.splitext(sys.argv[0])[0]
        ff = np.load(filename + '_' + dbName + '.npy')
        prefferedOri = np.argmax(tc, 1)
        ffMat = np.empty((NE+NI, len(thetas)))
        for kNeuron in np.arange(NE + NI):
            ffMat[kNeuron, :] = np.roll(ff[kNeuron, :], -1 * prefferedOri[kNeuron])

        plt.ion()

        plt.plot(thetas, np.nanmean(ffMat[:10000], 0), 'ko-', label='E')
        plt.plot(thetas, np.nanmean(ffMat[10000:], 0), 'ro-', label='I')
        plt.xlabel(r'Stimulus orientation ($\deg$)', fontsize = 20)
        plt.ylabel('Mean fano factor', fontsize = 20)
        plt.title(r'$\alpha = 0.5,\; \tau = 3.0,\; \xi = 1.2$', fontsize = 20)
        plt.legend()
        
        kb.keyboard()

        #f, ax = plt.subplots(2, 4)
        #f.set_size_inches(26.5,10.5)
        # print ax.shape
        # for i in np.arange(len(thetas)):
        #     subscripts = np.unravel_index(i, (2, 4))
        #     tmp = ffMat[NE:, i]
        #     tmp = tmp[tmp != 0]
        #     cnts, bins = np.histogram(tmp, 50)
        #     #cnts=cnts.astype(float)
        #     ax[subscripts].bar(bins[:-1], cnts, color = 'r', edgecolor = 'r', width = bins[1]-bins[0])
        #     tmp = ffMat[:NE, i]
        #     tmp = tmp[tmp != 0]
        #     cnts, bins = np.histogram(tmp, 50)
        #     #cnts=cnts.astype(float)
        #     ax[subscripts].bar(bins[:-1], cnts, color = 'k', edgecolor = 'k', width = bins[1]-bins[0])
        #     ax[subscripts].set_title(r'$\theta = %s$'%(thetas[i]))

        kb.keyboard()
#        ReportFig('FFvsOri_%s'%(dbName), 'alpha = %s, tau = 3ms, T = 1sec <br> average over 16 trials'%(alpha[0]), 'Fano factor vs Orientation', 'png', '', 'summary_alpha%s'%(alpha[0]))
 #           tmp[:, i] = cnts / np.sum(cnts)

        

#script to compute fano factor as a function of bi-directional connectivity alpha
basefolder  = "/home/shrisha/Documents/code/mypybox"
basefolder = "/homecentral/srao/Documents/code/mypybox"
#import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from enum import Enum
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
#import FanoFactorDynamics as ffd
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig




def CircVar(fanofactor, atTheta):
    zk = np.dot(fanofactor, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(fanofactor)

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
    simDuration = 4000
    spkTimeStart = 2000.0
    spkTimeEnd = 4000.0
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
        p = Pool(20)
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
    
    if(computeType == 'ff_circvar'):
        filename = os.path.splitext(sys.argv[0])[0]
        filename = './data/' + filename + '_' + dbName + '.npy'
        ff =  np.load(filename) 
        circVariance = np.zeros((NE + NI, ))
        theta = np.arange(0.0, 180.0, 22.5)
        circVarThresh = 0.4
        maxFiringRateThresh = 10
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/tuningCurves_%s.npy'%((dbName, )))
        ccv = np.load('/homecentral/srao/Documents/code/mypybox/db/Selectivity_' + dbName + '.npy')
        maxRates = np.max(tc, 1)
        nid = np.arange(NE + NI) # id vector 
        useNeuronId = nid[np.logical_and(ccv < circVarThresh, maxRates> maxFiringRateThresh)]
        print "valid Neuons NE = ", len(useNeuronId < NE), " NI = ", len(useNeuronId > NE)
        for i, kNeuron in enumerate(nid):
            circVariance[i] = CircVar(ff[kNeuron, :], theta)
        filename = filename + computeType
        print " saving as ", filename
        np.save(filename, circVariance)
        # plotting 
        # fcve = circVariance[:NE]
        # fcvi = circVariance[NE:]
        # fcve = fcve[fcve != 0.]
        # fcve = fcvi[fcvi != 0.]
        # nBins = 100
        # cntE, binsE = np.histogram(fcve[~np.isnan(fcve)], nBins)
        # cntI, binsI = np.histogram(fcvi[~np.isnan(fcvi)], nBins)

        fcve = circVariance[useNeuronId[useNeuronId < NE]]
        fcvi = circVariance[useNeuronId[useNeuronId > NE]]
        nBins = 100
        cntE, binsE = np.histogram(fcve[~np.isnan(fcve)], nBins)
        cntI, binsI = np.histogram(fcvi[~np.isnan(fcvi)], nBins)
        plt.bar(binsI[:-1], cntI, color = 'r', edgecolor = 'r', width = np.mean(np.diff(binsI)))
        plt.bar(binsE[:-1], cntE, color = 'k', edgecolor = 'k', width = np.mean(np.diff(binsE)))
        plt.legend(('I', 'E'), loc=0)
        plt.xlabel('Circular variance of fano factor', fontsize = 20)
        plt.ylabel('Counts')
        plt.title(r'Distribution of circular variance of Fano factor, $\alpha = %s$'%(alpha))
        plt.ion()
        plt.show()
        kb.keyboard()
    
    if(computeType == 'ff_po_scatter'):
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/tuningCurves_bidirII_%s.npy'%((dbName, )))
        filename = os.path.splitext(sys.argv[0])[0]
        filename = './data/' + filename + '_' + dbName + '.npy'
        ff =  np.load(filename) 
#        ff = np.load(filename + '_' + dbName + '.npy')
        prefferedOri = np.argmax(tc, 1)
        ffMat = np.empty((NE+NI, len(thetas)))
        for kNeuron in np.arange(NE + NI):
            ffMat[kNeuron, :] = np.roll(ff[kNeuron, :], -1 * prefferedOri[kNeuron])
        
        plt.plot(ffMat[NE:,0], ffMat[NE:, 4], '.r')
        plt.plot(ffMat[:NE,0], ffMat[:NE, 4], '.k')
        plt.ion()
        kb.keyboard()

    else:
        print "plotting ", "here"
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/tuningCurves_bidirII_%s.npy'%((dbName, ))); 
        #tc = np.load('/home/shrisha/Documents/cnrs/tmp/jan30/tuningCurves_allAnglesa0T4xi12C100Tr100.npy')
        filename = os.path.splitext(sys.argv[0])[0]
        ff = np.load(filename + '_' + dbName + '.npy')
        #ff = np.load('/home/shrisha/Documents/cnrs/tmp/jan30/FFvsOri_allAnglesa0T4xi12C100Tr100.npy')
        prefferedOri = np.argmax(tc, 1)
        ffMat = np.empty((NE+NI, len(thetas)))
        for kNeuron in np.arange(NE + NI):
            ffMat[kNeuron, :] = np.roll(ff[kNeuron, :], -1 * prefferedOri[kNeuron])

        plt.ion()
        circVar = np.load('/home/shrisha/Documents/cnrs/tmp/jan30/Selectivity_allAnglesa0T4xi12C100Tr100.npy')
        nid = np.arange(NE + NI); plotId = np.logical_and(circVar < 0.3, np.max(tc, 1) > 10)
        meanE = np.nanmean(ffMat[plotId[:NE], :], 0)
        meanI = np.nanmean(ffMat[plotId[NE:], :], 0)
        meanE = np.roll(meanE, 4)
        meanI = np.roll(meanI, 4)
        plt.plot(thetas, meanE, 'ko-', label='E')
        plt.plot(thetas, meanI, 'ro-', label='I')
        plt.xlabel(r'Stimulus orientation ($\deg$)', fontsize = 20)
        plt.ylabel('Mean fano factor', fontsize = 20)
        plt.title(r'$\alpha = 0.0,\; \tau = 3.0,\; \xi = 1.2$', fontsize = 20)
        plt.legend()
        plt.ion()
        plt.show()
        plt.waitforbuttonpress()
#        kb.keyboard()

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

        

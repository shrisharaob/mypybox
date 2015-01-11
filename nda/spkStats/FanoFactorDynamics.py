#script to compute fano factor in time
import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys
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
import SpkStats
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig


def MovingAverage(x, window_size):
    window = np.ones(int(window_size))/float(window_size) # window size in ms
    window = np.ones(int(window_size)) # window size in ms
    return np.convolve(x, window, 'valid')

def LineSlopeFit(x, y):
    # returns slope (m)  for least sqares linear fit:  y = mx + c
    xMean = np.mean(x)
    yMean = np.mean(y)
    centeredX = x - xMean
    slope = np.sum(centeredX * (y - yMean)) / np.sum(centeredX **2)
    intercept = yMean - (slope * xMean)
    return slope

def MeanMatchedFano(spkCntMeans, spkCntVars, nResample = 50, nBins = 20):
    # resample neurons to compute fano factor with mean count  distribution remaining the same for the period 
    # spkCntMeans : nNeurons-by-nTimeWindos
    # spkCntVars :  nNeurons-by-nTimeWindos
    nNeurons, nTimeWindows = spkCntMeans.shape
    cntDistr = np.zeros((nTimeWindows, nBins))
    bins = np.arange(nBins+1)
    neuronIdx = np.arange(nNeurons)
    histDataIdx = np.empty((nNeurons, nTimeWindows))
    histDataIdx[:] = np.nan
    for mWin in np.arange(nTimeWindows):
        mSpkMean = spkCntMeans[:, mWin] # nNeurons-by-1
        mSpkVar = spkCntVars[:, mWin]
        cnts, _ = np.histogram(mSpkMean, bins)
        cntDistr[mWin, :] = cnts
        histDataIdx[:, mWin] = np.digitize(mSpkMean, bins)
    # FIND GREATEST COMMON DISTRIBUTION
    greatestCommonCntDistr = np.zeros((nBins, ))
    for lBin in np.arange(nBins):
        greatestCommonCntDistr[lBin] = np.min(cntDistr[:, lBin])
        
    # MATCH THE DISTRIBUSTIONS
    # discard neurons so that the height of a given bin equals that of greatestCommonCntDistr
    iterFano = np.empty((nResample, ))
    iterFano[:] = np.nan
    for nIter in np.arange(nResample):
        finalDiscardMask = np.empty((nNeurons, ), dtype = bool) # logical array nNeurons-by-1 with elements False to be discarded before computing the slope
        finalDiscardMask[:] = True
        for kWin in np.arange(nTimeWindows): # windows are the obs windows for computing the Fano Factor
            kWinCntDistr = cntDistr[kWin, :]
            for iBin in np.arange(nBins): # bins are the spk count bins
                epsilon = 1e-4 # stop when heightDiffInBin <= eps
                heightDiffInBin = kWinCntDistr[iBin] - greatestCommonCntDistr[iBin]                 
                nTries = 0 
                print iBin
                print heightDiffInBin
                resampledSpkMeans = spkCntMeans[:, kWin]
                while((heightDiffInBin > epsilon) & (nTries < nNeurons)):
                    nTries += 1
                    neuronsInThisBin = neuronIdx[histDataIdx[:, mWin] == (iBin + 1)]
                    neuronToDiscard = np.random.choice(neuronsInThisBin, int(np.floor(heightDiffInBin)))
                    discardMask = ~ np.in1d(neuronIdx, neuronToDiscard)
                    resampledSpkMeans = resampledSpkMeans[discardMask]
                    nIterDistr, _ = np.histogram(resampledSpkMeans, bins)
                    heightDiffInBin = nIterDistr[iBin] - greatestCommonCntDistr[iBin]
                finalDiscardMask = finalDiscardMask & discardMask
                iterFano[nIter] = LineSlopeFit(spkCntMeans[finalDiscardMask, kWin], spkCntVars[finalDiscardMask, kWin])

    return np.nanmean(iterFano)



def LineSlopeFitForAllWindows(spkCntMeans, spkCntVars, timeWindowId):
    # returns slope for specified time window
    # spkCntMeans : nNeurons-by-nTimeWindows
    # spkCntVars : nNeurons-by-nTimeWindows
    x = spkCntMeans[:, timeWindowId]
    y = spkCntVars[:, timeWindowId]
    return LineSlopeFit(x, y)

def MovingSpkCnt(spkTimes, bins, winSize):
    meanSpkCount = np.zeros((len(bins), ))
    if(spkTimes.size > 0):
        counts, bins = np.histogram(spkTimes, bins)
        counts = counts.astype('float')
        mvSpkCnt = MovingAverage(counts, winSize)
    return mvSpkCnt



def FanoBeforeAndAfter(dbName, spkTimeStart, spkTimeEnd, nTrials, neuronId):
    avgSpkCnt = np.zeros((nTrials, ))
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    fanoFactor = np.empty((1, ))
    fanoFactor[:] = np.nan
    print "neuronId", neuronId
    for kTrial in np.arange(nTrials):
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND spkTimes > %s AND spkTimes < %s ", (neuronId, kTrial, float(spkTimeStart), float(spkTimeEnd)))
        avgSpkCnt[kTrial] = float(nSpks)
    dbCursor.close()
    db.close()
    kb.keyboard()
    # ff = np.var(avgSpkCnt) / np.mean(avgSpkCnt)
    #    return  np.var(avgSpkCnt) / np.mean(avgSpkCnt), np.var(avgSpkCnt), np.mean(avgSpkCnt)
    return  np.array([np.mean(avgSpkCnt, 0), np.var(avgSpkCnt, 0)])


# def TestFanoInTime(nTrials, winSize, neuronId):
#     print "testing fano in time "
#     histBinSize = 1.0
#     spkTimes = np.load('GenSpikes.npy')
#     bins = np.arange(spkTimeStart, spkTimeEnd + 0.001, histBinSize)
#     nBins = len(bins)
#     avgSpkCnt = np.zeros((nTrials, nBins-winSize))
#     for kTrial in np.arange(nTrials):
#         print kTrial
#         avgSpkCnt[kTrial, :] = MovingSpkCnt(spkTimes[kTrial], bins, winSize)
#     kb.keyboard()
#     return np.array([np.mean(avgSpkCnt, 0), np.var(avgSpkCnt, 0)])

# def TestFanoAux(nNeurons, nTrials):
#     spkTimes = np.load('GenSpikes.npy')
#     for kNeuron in np.arange(nNeurons):
        

def FanoInTime(dbName, spkTimeStart, spkTimeEnd, winSize, nTrials, neuronId):
    histBinSize = 1.0
    bins = np.arange(spkTimeStart, spkTimeEnd + 0.001, histBinSize)
    nBins = len(bins)
    avgSpkCnt = np.zeros((nTrials, nBins-winSize))
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    fanoFactor = np.empty((nBins, ))
    fanoFactor[:] = np.nan
#    print "neuronId", neuronId
    if(neuronId == 0):
       nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE theta = %s AND spkTimes > %s AND spkTimes < %s ", (0, float(spkTimeStart), float(spkTimeEnd))) 
       print "Mean FR: ", float(nSpks) / (10000. *  (spkTimeEnd - spkTimeStart) * 1e-3)
    for kTrial in np.arange(nTrials):
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND spkTimes > %s AND spkTimes < %s ", (neuronId, kTrial, float(spkTimeStart), float(spkTimeEnd)))
        if(nSpks > 0):
            spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))
            avgSpkCnt[kTrial, :] = MovingSpkCnt(spkTimes, bins, winSize)
    dbCursor.close()
    db.close()
    return  np.array([np.mean(avgSpkCnt, 0), np.var(avgSpkCnt, 0)])

if __name__ == "__main__":
    alpha = 0.0
    NE = 5000
    NI = 5000
    simDuration = 1000.0
    spkTimeStart = 0.0
    spkTimeEnd = 4000.0
    simDT = 0.05
    tau = 3.0
    winSize = 50.0  # in ms fano factor observation window
    dbName = 'a0t3T6xi12C0n1'
    computeType = 'compute'

    print alpha
    [dbName, computeType, alpha, tau, spkTimeStart, spkTimeEnd, simDT, NE, NI] = DefaultArgs(sys.argv[1:], [dbName, 'plot', alpha, tau, spkTimeStart, spkTimeEnd, simDT, NE, NI])
#    thetas = np.arange(0., 180., 22.5)
#    thetas = np.arange(20)
    nTrials = 100
    nBins = spkTimeEnd - spkTimeStart + 1
    filetag = '_w%s_'%((winSize, )) 
    if(computeType == 'compute'):
        neuronsList = np.arange(NE + NI)
#        p = Pool(16)
        func0 = partial(FanoBeforeAndAfter, dbName, spkTimeStart, spkTimeEnd, nTrials)
        ffunc = partial(FanoInTime, dbName, spkTimeStart, spkTimeEnd, winSize, nTrials)
 
        func0(0)

        result = p.map(func0, neuronsList)
        result = np.asarray(result) # nNeurons-by-2-by-nTimeWindows

        print result
        kb.keyboard()

        #print "bEFORE "
        #result = p.map(ffunc, neuronsList)

#        np.save('FanoFactorDynamics_spkCnt_var' + filetag + dbName, np.asarray(result))
        # meanMatchedFFE = MeanMatchedFano(result[:NE, 0, :], result[:NE, 1, :])
        # meanMatchedFFI = MeanMatchedFano(result[NE:, 0, :], result[NE:, 1, :])
        # np.save('FanoFactorDynamics_meanMatchedFano_' + dbName, np.asarray([meanMatchedFFE, meanMatchedFFI]))


        # E neurons 
        spkCntMeans = result[:NE, 0, :]
        spkCntVars = result[:NE, 1, :]
        print "msc, scv shapes: ", spkCntVars.shape, spkCntMeans.shape
        nNeurons, nTimeWindows = spkCntVars.shape
        FanoFuncRegFit = partial(LineSlopeFitForAllWindows, spkCntMeans, spkCntVars)
        outE =  p.map(FanoFuncRegFit, np.arange(nTimeWindows))
#        outE = tmp[0]
 #       outEIntercept = tmp[1]
        #print "intercept:", np.nanmean(outEIntercept)

        # I neurons
        spkCntMeans = result[NE:, 0, :]
        spkCntVars = result[NE:, 1, :]
        _, nTimeWindows = spkCntVars.shape
        FanoFuncRegFit = partial(LineSlopeFitForAllWindows, spkCntMeans, spkCntVars)
        outI = p.map(FanoFuncRegFit, np.arange(nTimeWindows))
        #outI = tmp[0]
        #outIIntercept = tmp[1]
        #print "intercept:", np.nanmean(outIIntercept)
        p.close()
#        x = np.asarray(result)
 #       print x.shape
 #       print "ffe: ", np.nanmean(x[:NE], 0), "ffi: ", np.nanmean(x[NE:], 0) 
        np.save('FanoFactorDynamics' + filetag + dbName, np.asarray([outE, outI]))
  #      kb.keyboard()
    else:
        filename = 'FanoFactorDynamics' + filetag + dbName + '.npy'
        print "loading file", filename
        y = np.load(filename)
#        kb.keyboard()
        plt.ioff()
        xLim = int(y.shape[1] * 0.5)
        xAxis = np.arange(-1 * (xLim + 1), xLim, 1.0)
        print y.shape, xAxis.shape, (y.shape[1] * 0.5) + 1, y.shape[1] * 0.5
        #plt.plot(xAxis, np.nanmean(y[:NE, :], 0), 'k', label='E')
        #plt.plot(xAxis, np.nanmean(y[NE:, :], 0), 'r', label='I')
        plt.plot(xAxis, y[0, :], 'k', label = 'E')
        plt.plot(xAxis, y[1, :], 'r', label = 'I')
        plt.legend(loc = 0)
        plt.xlabel('Time (ms)', fontsize = 20)
        plt.ylabel('Mean fano factor', fontsize = 20)
        plt.title(r'$\alpha = %s, \; \tau = %s$'%((alpha, tau)), fontsize = 20)
        plt.grid()
        plt.savefig('FanoFactorDynamics_' + dbName)
        #kb.keyboard()


































        # ffMat = np.empty((NE+NI, len(thetas))) # matrix N_NEURONS-by-theta with each element containing the fano factor 
        # ffMat[:] = np.nan








 
#    #        out = ffunc(lTheta)
#             meanSpkCnt = out[0][:, 0]
#             spkVar = out[0][:, 1]
#             tmpidx = ~(np.isnan(meanSpkCnt))
#             tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
#             meanSpkCnt = meanSpkCnt[tmpidx]
#             spkVar = spkVar[tmpidx]
#             ff[tmpidx] = spkVar / meanSpkCnt
#             ffMat[:, ll] = ff

#         kb.keyboard()

#     else:
#         print "plotting "
#         tc = np.load('/homecentral/srao/Documents/code/mypybox/db/tuningCurves_bidirII_%s.npy'%((dbName, )))
#         ff = np.load('./data/ffMat_ffvsTheta_%s.npy'%((dbName, )))
#         prefferedOri = np.argmax(tc, 1)
#         ffMat = np.empty((NE+NI, len(thetas)))
#         for kNeuron in np.arange(NE + NI):
#             ffMat[kNeuron, :] = np.roll(ff[kNeuron, :], -1 * prefferedOri[kNeuron])

# #        tmp = np.empty((50, 8))
#         plt.ioff()
#         f, ax = plt.subplots(2, 4)
#         f.set_size_inches(26.5,10.5)
#         print ax.shape
#         for i in np.arange(len(thetas)):
#             subscripts = np.unravel_index(i, (2, 4))
#             tmp = ffMat[NE:, i]
#             tmp = tmp[tmp != 0]
#             cnts, bins = np.histogram(tmp, 50)
#             #cnts=cnts.astype(float)
#             ax[subscripts].bar(bins[:-1], cnts, color = 'r', edgecolor = 'r', width = bins[1]-bins[0])
#             tmp = ffMat[:NE, i]
#             tmp = tmp[tmp != 0]
#             cnts, bins = np.histogram(tmp, 50)
#             #cnts=cnts.astype(float)
#             ax[subscripts].bar(bins[:-1], cnts, color = 'k', edgecolor = 'k', width = bins[1]-bins[0])
#             ax[subscripts].set_title(r'$\theta = %s$'%(thetas[i]))
            
#         ReportFig('FFvsOri_%s'%(dbName), 'alpha = %s, tau = 3ms, T = 1sec <br> average over 16 trials'%(alpha[0]), 'Fano factor vs Orientation', 'png', '', 'summary_alpha%s'%(alpha[0]))
#  #           tmp[:, i] = cnts / np.sum(cnts)

        



# def FanoFactor(spkTimes, binSize, discardTime, simDuration):
#     fanoFac = 0.0
#     meanSpkCount = 0.0
#     spkCntVar = 0.0
#     fr = 0.0
#     histBinSize = 1.0
#     if(len(spkTimes) > 0):
#         bins = np.arange(discardTime, simDuration+0.0001, histBinSize)
#         counts, bins = np.histogram(spkTimes, bins)
#         counts = counts.astype('float')
#         mvSpkCnt = MovingAverage(counts[1:-1], binSize)
#         kb.keyboard()
#         meanSpkCount = np.mean(mvSpkCnt)
#         spkCntVar = np.var(mvSpkCnt)
# #        fr = float(np.sum(spkTimes > 2000.0)) / ((simDuration)*1e-3)
#         # if(meanSpkCount != 0):
#         #     fanoFac = spkCntVar/ meanSpkCount # the first and last bins are discarded
#     return meanSpkCount, spkCntVar

# def AvgFano(dbName, neuronsList, simDuration, simDT, binSize, theta = 0):
#     # binsize in ms
#     print "fldr this file"
#     nNeurons = len(neuronsList)
#     avgFanoFactor = 0;
#     nValidNeurons = 0;
#     db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
#     dbCursor = db.cursor()
#     db.autocommit(True)
#     out = np.empty((nNeurons, 2))
#     out[:] = np.nan
#     fanoFac = np.zeros((nNeurons,))
#     dicardT = 2000.0
#     validDuration = simDuration - dicardT
# #    fr = np.zeros(neuronsList.shape)
#     trials = np.arange(15)
#     for i, kNeuron in enumerate(neuronsList):
#         for kk, kTrial in enumerate(trials):
#             nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND trial = %s", (kNeuron, theta, kTrial))
#             nValidNeurons += 1
#             if(nSpks > 0):
#                 spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))




















#                 spkTimes = spkTimes[spkTimes < simDuration]
#                 spkTimes = spkTimes[spkTimes > dicardT]
#                 tmp = FanoFactor(spkTimes, binSize, dicardT, simDuration)
#                 out[i, 0] = tmp[0]
#                 out[i, 1] = tmp[1]
# #            fanoFac[i] = tmp[2]

    #      dbCursor.close()
#     db.close()
#     return out

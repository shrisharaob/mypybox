#script to compute fano factor in time
import numpy as np
import scipy.stats as stat
import code, sys
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append("/homecentral/srao/Documents/code/mypybox/nda/spkStats")
import SpkStats
import Keyboard as kb
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig


def MovingAverage(x, window_size):
    window = np.ones(int(window_size))/float(window_size) # window size in ms
    window = np.ones(int(window_size)) # window size in ms
    return np.convolve(x, window, 'valid')

def ObjLin(x, a, b):
    return a * x + b;
    
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
    bins = np.arange(nBins+1) + 0.5
    bins = np.concatenate(([0.0], bins))
    nBins = len(bins) - 1
    cntDistr = np.zeros((nTimeWindows, nBins))
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
    iterFano = np.empty((nResample, nTimeWindows))
    iterFano[:] = np.nan
    for nIter in np.arange(nResample):
        print "iter: ", nIter
        finalDiscardMask = np.empty((nNeurons, ), dtype = bool) # logical array nNeurons-by-1 with elements False to be discarded before computing the slope
        finalDiscardMask[:] = True
        for kWin in np.arange(nTimeWindows): # windows are the obs windows for computing the Fano Factor
            kWinCntDistr = cntDistr[kWin, :]
            for iBin in np.arange(nBins): # bins are the spk count bins
                epsilon = 1e-4 # stop when heightDiffInBin <= eps
                heightDiffInBin = kWinCntDistr[iBin] - greatestCommonCntDistr[iBin]                 
                nTries = 0 
                resampledSpkMeans = spkCntMeans[:, kWin]
                discardMask = np.empty((nNeurons, ), dtype = bool)
                discardMask[:] = True
                while((heightDiffInBin > epsilon) & (nTries < nNeurons)):
                    nTries += 1
                    oldDiscardMask = discardMask
                    neuronsInThisBin = neuronIdx[histDataIdx[:, mWin] == (iBin + 1)]
                    neuronToDiscard = np.random.choice(neuronsInThisBin, int(np.floor(heightDiffInBin)))
                    discardMask = ~ np.in1d(neuronIdx, neuronToDiscard)
                    oldDiscardMask = oldDiscardMask & discardMask
                    nIterDistr, _ = np.histogram(resampledSpkMeans[oldDiscardMask], bins)
                    heightDiffInBin = nIterDistr[iBin] - greatestCommonCntDistr[iBin]
                print "kBin: ", iBin, "ntries:", nTries
                finalDiscardMask = finalDiscardMask & discardMask
                iterFano[nIter, kWin] = LineSlopeFit(spkCntMeans[finalDiscardMask, kWin], spkCntVars[finalDiscardMask, kWin])
    return np.nanmean(iterFano, 0)



def LineSlopeFitForAllWindows(spkCntMeans, spkCntVars, timeWindowId):
    # returns slope for specified time window
    # spkCntMeans : nNeurons-by-nTimeWindows
    # spkCntVars : nNeurons-by-nTimeWindows
    x = spkCntMeans[:, timeWindowId]
    y = spkCntVars[:, timeWindowId]
    return LineSlopeFit(x, y)

def MovingSpkCnt(spkTimes, bins, winSize):
#    mvSpkCnt = np.zeros((len(bins), ))
    mvSpkCnt = 0.0
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
#    print "neuronId", neuronId
    for kTrial in np.arange(nTrials):
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND spkTimes > %s AND spkTimes < %s ", (neuronId, kTrial, float(spkTimeStart), float(spkTimeEnd)))
        avgSpkCnt[kTrial] = float(nSpks)
    dbCursor.close()
    db.close()
#    kb.keyboard()
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
        

def FanoInTime(spkArrayList, spkTimeStart, spkTimeEnd, winSize, nTrials, neuronId, discardT = 1000.0):
    histBinSize = 1.0
    bins = np.arange(spkTimeStart, spkTimeEnd + 0.001, histBinSize)
    nBins = len(bins)
    avgSpkCnt = np.zeros((nTrials, nBins-winSize))
    fanoFactor = np.empty((nBins, ))
    fanoFactor[:] = np.nan
    for kTrial in np.arange(nTrials):
        starray = spkArrayList[kTrial]
        nSpks = len(starray)
        if(nSpks > 0):
            spkTimes = starray[starray[:, 1] == neuronId, 0]
            spkTimes = spkTimes[spkTimes > discardT]            
            avgSpkCnt[kTrial, :] = MovingSpkCnt(spkTimes, bins, winSize)
    return  np.array([np.nanmean(avgSpkCnt, 0), np.nanvar(avgSpkCnt, 0)])

if __name__ == "__main__":
    [foldername, nTrials, alpha, computeType, NE, NI, simDuration, spkTimeStart, spkTimeEnd, simDT, tau, winSize] = DefaultArgs(sys.argv[1:], ['', 100, 0.0, 'compute', 10000, 10000, 3000.0, 1500.0, 2500.0, 0.05, 3.0, 50])
    alpha = float(alpha)
    NE = int(NE)
    NI = int(NI)
    simDuration = float(simDuration)
    spkTimeStart = float(spkTimeStart)
    spkTimeEnd = float(spkTimeEnd)
    simDT = float(simDT)
    tau = float(tau)
    winSize = int(50)  # in ms fano factor observation window
    computeType = 'compute'
    nTrials = int(nTrials)                  
    nBins = spkTimeEnd - spkTimeStart + 1
    filetag = '_w%s_'%((winSize, ))
    bf = '/homecentral/srao/cuda/data/bidir/e2i/g4x/'
    if(computeType == 'mm'): #mean matched
        y = np.load('FanoFactorDynamics_spkCnt_var' + filetag + dbName + '.npy')
        sc = y[:, 0, :]
        sv = y[:, 1, :]
        meanMatchedFFE = MeanMatchedFano(sc, sv, 2, 10)

    if(computeType == 'ba'): #before after
        neuronsList = np.arange(NE + NI)
        p = Pool(22)
        func0 = partial(FanoBeforeAndAfter, dbName, spkTimeStart, spkTimeEnd, nTrials)
        result = p.map(func0, neuronsList)
        result = np.asarray(result) # nNeurons-by-2-by-nTimeWindows
        np.save('FanoFactorDynamics_spkCnt_var_chunk' + filetag + dbName, np.asarray(result))

    if(computeType == 'compute'):
        spkArrayList = []
        print 'in folder: ', bf + foldername
        for i in range(nTrials):
            filename = '/spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr%s.csv'%(int(alpha), int(tau), int(simDuration), i)                                    
            fullpath2file = bf + foldername + filename
            print 'loading file: ', filename[1:]
            spkArrayList.append(np.loadtxt(fullpath2file, delimiter = ';'))                                                                                                                                
        neuronsList = np.arange(NE + NI)
        p = Pool(24)
 #      func0 = partial(FanoBeforeAndAfter, dbName, spkTimeStart, spkTimeEnd, nTrials)
        ffunc = partial(FanoInTime, spkArrayList, spkTimeStart, spkTimeEnd, winSize, nTrials)
        #ffunc(neuronsList)
        result = p.map(ffunc, neuronsList)
        result = np.asarray(result) # nNeurons-by-2-by-nTimeWindows
        #result = p.map(ffunc, neuronsList)
        np.save('FanoFactorDynamics_spkCnt_var_' + foldername + filetag, np.asarray(result))
        # E neurons 
        spkCntMeans = result[:NE, 0, :]
        spkCntVars = result[:NE, 1, :]
        print "msc, scv shapes: ", spkCntVars.shape, spkCntMeans.shape
        nNeurons, nTimeWindows = spkCntVars.shape
 #      popt, pcov = curve_fit(ObjLin, spkCntMeans, spkCntVars, p0 = (0.1, 0.1))
        FanoFuncRegFit = partial(LineSlopeFitForAllWindows, spkCntMeans, spkCntVars)
        outE =  p.map(FanoFuncRegFit, np.arange(nTimeWindows))
 #      outE = tmp[0]
 #      outEIntercept = tmp[1]
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
        np.save('FanoFactorDynamics' + filetag + foldername, np.asarray([outE, outI]))
  #      kb.keyboard()
    if(computeType == 'plot'):
        filename = 'FanoFactorDynamics' + filetag + dbName + '.npy'
#        filename = 'FanoFactorDynamics_' + dbName + '.npy'

        print "loading file", filename
        y = np.load(filename)
#        kb.keyboard()
        plt.ioff()
        xLim = int(y.shape[1] * 0.5)
        xAxis = np.arange(-1 * (xLim + 1), xLim, 1.0)
        #xAxis = np.arange(0, 2 * xLim + 1, 1.0)
        print y.shape, xAxis.shape, (y.shape[1] * 0.5) + 1, y.shape[1] * 0.5
        #plt.plot(xAxis, np.nanmean(y[:NE, :], 0), 'k', label='E')
        #plt.plot(xAxis, np.nanmean(y[NE:, :], 0), 'r', label='I')
        plt.plot(xAxis, y[0, :], 'k', label = 'E')
        plt.plot(xAxis, y[1, :], 'r', label = 'I')

#        plt.plot(xAxis, 0.5 * (y[1, :] +  y[0, :]), 'k')

#        plt.legend(loc = 0)
        plt.xlabel('Time (ms)', fontsize = 20)
        plt.ylabel('Mean fano factor', fontsize = 20)
        plt.title(r'$\alpha = %s, \; \tau = %s$, window size = %sms'%((alpha, tau, winSize)), fontsize = 20)
#        plt.title('Poisson spikes, window size = %sms'%(winSize), fontsize = 20)
        plt.grid()
        figname = 'FanoFactorDynamics_' + filetag + dbName + '.png'
        print " saving figure as ", figname 

        plt.text(0.79, -500, 'Firing rate (Hz)')
        plt.text(0.77, -500, 'C = 0, E: 0.99 I: 1.81')
        plt.text(0.755, -500, 'C = 1, E: 4.23 I: 7.48')
        kb.keyboard()
        plt.savefig(figname, format = 'png')

































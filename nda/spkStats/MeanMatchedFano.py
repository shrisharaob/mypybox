import numpy as np
import code, sys
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
from FanoFactorDynamics import LineSlopeFit
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig

def MeanMatchedFano(spkCntMeans, spkCntVars, nResample = 2, nBins = 20):
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
    
 #   kb.keyboard()
    # MATCH THE DISTRIBUSTIONS
    # discard neurons so that the height of a given bin equals that of greatestCommonCntDistr
    iterFano = np.empty((nResample, nTimeWindows))
    iterFano[:] = np.nan
    for nIter in np.arange(nResample):
        print "iter :", nIter
        finalDiscardMask = np.empty((nNeurons, ), dtype = bool) # logical array nNeurons-by-1 with elements False to be discarded before computing the slope
        finalDiscardMask[:] = True
        for kWin in np.arange(nTimeWindows): # windows are the obs windows for computing the Fano Factor
            kWinCntDistr = cntDistr[kWin, :]
            for iBin in np.arange(nBins): # bins are the spk count bins
                epsilon = 1e-4 # stop when heightDiffInBin <= eps
                heightDiffInBin = kWinCntDistr[iBin] - greatestCommonCntDistr[iBin]                 
                nTries = 0 
                neuronsInThisBin = neuronIdx[histDataIdx[:, mWin] == (iBin + 1)]
                discardMask = np.empty((nNeurons, ), dtype = bool)
                discardMask[:] = True
                if(neuronsInThisBin.size > 0):
                    while((heightDiffInBin > epsilon) & (nTries < nNeurons) & (neuronsInThisBin.size > 1 )):
                        nTries += 1
                        neuronToDiscard = np.random.choice(neuronsInThisBin, int(np.floor(heightDiffInBin)))
                        neuronsInThisBin = np.setdiff1d(neuronsInThisBin, neuronToDiscard)
                        if(nTries == 1):
                            discardMask = ~ np.in1d(neuronIdx, neuronToDiscard)
                        else:
                            discardMask  = discardMask & (~ np.in1d(neuronIdx, neuronToDiscard))
                        resampledSpkMeans = spkCntMeans[discardMask, kWin] 
                        nIterDistr, _ = np.histogram(resampledSpkMeans, bins)
                        heightDiffInBin = nIterDistr[iBin] - greatestCommonCntDistr[iBin]
                        
                finalDiscardMask = finalDiscardMask & discardMask
            # plt.ioff()
            # plt.plot(spkCntMeans[finalDiscardMask, kWin], spkCntVars[finalDiscardMask, kWin], 'k.')
            # maxXY = np.max([plt.xlim(), plt.ylim()])
            # xeyLine = np.linspace(0, maxXY, 10)
            # plt.plot(xeyLine, xeyLine, 'g')
            # plt.xlabel('Mean spike count')
            # plt.ylabel('Spike count variance')
            # ReportFig('test_FanoDynamics_%s'%('a0t3'), 'alpha = %s, tau = 3ms, tWindow = %s,  T = 6sec, stimulus onset at 5s <br> average over 100 trials'%(0, kWin), 'Fano factor dynamics', 'png', '', 'fano_dynamics_%s'%(kWin))
  #            kb.keyboard()
            iterFano[nIter, kWin] = LineSlopeFit(spkCntMeans[finalDiscardMask, kWin], spkCntVars[finalDiscardMask, kWin])

    return np.nanmean(iterFano, 0)

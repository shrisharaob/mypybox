#!/usr/bin/python
import numpy as np

def ComputeCV(spkTimes):
# spkTimes : nSpks x nNeurons     
    nSpks, nNeurons = spkTimes.shape
    nNeurons = 10000
    cv = np.empty(nNeurons, dtype = float)
    cv.fill(np.nan)
    minSpkies = 100
    print "computing cv ..."
    for kNeuron in np.arange(nNeurons):
        kSpk = spkTimes[spkTimes[:, 1] == kNeuron, 0]
        if(kSpk.size > minSpkies):
            isi = np.diff(kSpk)
            avgISI = np.mean(isi)
            if(avgISI > 0):
                cv[kNeuron] = np.std(isi) / avgISI;
    return cv



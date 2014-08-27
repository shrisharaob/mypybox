#!/usr/bin/python
import numpy as np

def ComputeCV(spkTimes, minSpikes = 100):
# spkTimes : nSpks x nNeurons     
    nSpks, junk = spkTimes.shape
    neuronIds = np.unique(spkTimes[:, 1])
    nNeurons = neuronIds.size
    cv = np.empty(nNeurons, dtype = float)
    cv.fill(np.nan)
    print "computing cv ... "
    for kNeuron in np.arange(nNeurons):
        kSpk = spkTimes[spkTimes[:, 1] == neuronIds[kNeuron], 0]
        if(kSpk.size > minSpikes):
            isi = np.diff(kSpk)
            avgISI = np.mean(isi)
            if(avgISI > 0):
                cv[kNeuron] = np.std(isi) / avgISI;
    print "done"
    return cv

def FiringRate(spkTimes, simDuration, minSpikes = 5):
    nSpks, junk = spkTimes.shape
    neuronIds = np.unique(spkTimes[:, 1])
    nNeurons = neuronIds.size
    r = np.empty(nNeurons, dtype = float)
    r.fill(np.nan)
    for kNeuron in np.arange(nNeurons):
        kSpk = spkTimes[spkTimes[:, 1] == neuronIds[kNeuron], 0]
        if(kSpk.size > minSpikes):
            r[kNeuron] = float(len(kSpk)) / simDuration

    return r        

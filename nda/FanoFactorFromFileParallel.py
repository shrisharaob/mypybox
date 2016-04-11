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
from multiprocessing import Pool
from functools import partial

def AvgFano(spkArrayList, nTrials, alpha, discardT, neuronsList): #, simDuration, simDT):
    nNeurons = neuronsList.size
    print '#', nNeurons
    nValidNeurons = 0;
    fanoFactor = []
    validNeurons = []
    spkCounts = np.zeros((nTrials, nNeurons))
    meanSpkCnt = []
    spkCountVar = []    
    for mTrial in range(nTrials):
        starray = spkArrayList[mTrial]
        for i, kNeuron in enumerate(neuronsList):
 #               validNeurons.append(kNeuron)
 #               nValidNeurons += 1
                spkTimes = starray[starray[:, 1] == kNeuron, 0]
                spkTimes = spkTimes[spkTimes > discardT]
                spkCounts[mTrial, kNeuron] = spkTimes.size
    for kk, kNeuron in enumerate(neuronsList):
        fanoFactor.append(spkCounts[:, kk].var() / spkCounts[:, kk].mean())
        spkCountVar.append(spkCounts[:, kk].var())
        fanoFactor.append(spkCounts[:, kk].var() / spkCounts[:, kk].mean())        

    return np.array(fanoFactor),  [np.array(meanSpkCnt), np.array(spkCountVar)]

if __name__ == '__main__':
    [foldername, alpha, filetag, nTrials, NE, NI, simDuration] = DefaultArgs(sys.argv[1:], ['', '', '', 3, 20000, 20000, 3000])
    nTrials = int(nTrials)
    NE = int(NE)
    NI = int(NI)
    simDuration = int(simDuration)
    bf = '/homecentral/srao/cuda/data/pub/bidir/'    
    spkArrayList = []
    for i in range(nTrials):
        filename = bf + foldername +'/spkTimes_xi0.8_theta0_0.%s0_3.0_cntrst100.0_%s_tr%s.csv'%(int(alpha), simDuration, i)
        print 'loading file: ', filename
        spkArrayList.append(np.loadtxt(filename, delimiter = ';'))
    neuronsList = np.arange(NE+NI)
    print 'computing fano factor ...',
    sys.stdout.flush()
#    fanoFactor = AvgFano(spkArrayList, neuronsList, nTrials, 1000.0)
    discardTime = 1000.0
    pfunc = partial(AvgFano, spkArrayList, nTrials, alpha, discardTime)
#    results = pfunc(neuronsList)
    pyWorkers = Pool(48)
    results = pyWorkers.map(pfunc, range(NE+NI))
    fanoFactor = results[0]
    spkCntMeanVar = results[1]    
    print 'done'
    np.save('./data/fanofactor_p%s_'%(int(alpha), ) + filetag, fanoFactor)
    np.save('./data/spkCnt_mean_var_p%s_'%(alpha) + filetag, np.array(spkCntMeanVar))
    

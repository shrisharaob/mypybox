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

def AvgFano(spkArrayList, neuronsList, nTrials, alpha, discardT, filetag): #, simDuration, simDT):
    nNeurons = len(neuronsList)
    nValidNeurons = 0;
    fanoFactor = []
    validNeurons = []
 #   validDuration = simDuration - dicardT
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
#                spkTimes = spkTimes[spkTimes < 1000.0]                
                spkCounts[mTrial, kNeuron] = spkTimes.size
    for kk, kNeuron in enumerate(neuronsList):
        meanSpkCnt.append(spkCounts[:, kk].mean())
        spkCountVar.append(spkCounts[:, kk].var())
        fanoFactor.append(spkCounts[:, kk].var() / spkCounts[:, kk].mean())
    np.save('./data/spkCnt_mean_var_bidir_%s_p%s'%(filetag, alpha), [np.array(meanSpkCnt), np.array(spkCountVar)])
    return np.array(fanoFactor) #, validNeurons

def GenPoissionSpkies(rate, nSpkiesPerTrial, nTrials):
    spkArrayList = []
    for i in range(nTrials):
        spkTimes = np.zeros((nSpkiesPerTrial, 2))
        spkTimes[:, 1] = 0
        spkTimes[:, 0] = np.cumsum(np.random.exponential(1.0/rate, size = (nSpkiesPerTrial, ))) * 1e3
        spkArrayList.append(spkTimes)
    return spkArrayList

def GenPoissionSpkies_Alt(rates, nSpkiesPerTrial, nNeurons, endTime = 3000):
    spkArrayList = np.empty((nSpkiesPerTrial * nNeurons, 2))
    spkArrayList[:] = np.nan
    rowCounter = 0
    rowCounterOld = 0
    for i in range(nNeurons):
        spkTimes = np.zeros((nSpkiesPerTrial, 2))
        spkTimes[:, 1] = i
        tmpSpks = np.cumsum(np.random.exponential(1.0/rates[i], size = (nSpkiesPerTrial, ))) * 1e3
        tmpSpks = tmpSpks[tmpSpks <= endTime]
        nTmpSpks = tmpSpks.size
        if nTmpSpks > 0:
            rowCounter += nTmpSpks
            spkTimes[:nTmpSpks, 0] = tmpSpks
            spkArrayList[rowCounterOld : rowCounter, : ] = spkTimes[:nTmpSpks, :]
            rowCounterOld = rowCounter
    return spkArrayList[:rowCounter, :]
    
#   spkArrayList[i * nSpkiesPerTrial : i * nSpkiesPerTrial + nSpkiesPerTrial, :] = spkTimes
    
def GenPoissionSpkies_Alt_misc(nTrials, nNeurons):
    starray = []
    rates = np.exp(np.random.normal(0, 1, size=(nNeurons, )))
    nSpikesPerTrial = 5
    for kk in range(nTrials):
        starray.append(GenPoissionSpkies_Alt(rates, nSpikesPerTrial, nNeurons))
    return starray

if __name__ == '__main__':
    #USAGE: python FanoFactorFromFile.py i2i/p1/fano 1 3 I2I
    [bidirType, alpha, tau_syn, nTrials, NE, NI] = DefaultArgs(sys.argv[1:], ['', '', '', 100, 20000, 20000])
    nTrials = int(nTrials)
    NE = int(NE)
    NI = int(NI)
#   bf = '/homecentral/srao/cuda/data/poster/'
    bf = '/homecentral/srao/cuda/data/pub/bidir/'
#   bf = '/homecentral/srao/cuda/data/pub/kffi400/'
#   bf = '/homecentral/srao/cuda/data/pub/k_orig_paper/'
#    bf = '/homecentral/srao/cuda/data/pub/k_orig_paper/ki12ke8/'
    computeType = '' #'test'
    spkArrayList = []
    if int(tau_syn) == 3:
        foldername = bidirType + '/p%s'%(alpha) + '/fano'
    else:
        foldername = bidirType + '/tau%s/p%s'%((tau_syn, alpha)) + '/fano'
 #   foldername = 'p%s'%(alpha)
    if bidirType == 'e2i':
        filetag = 'E2I'
    if bidirType == 'e2e':
        filetag = 'E2E'
    if bidirType == 'i2i':
        filetag = 'I2I'
    filetag = filetag + '_tau%s'%(tau_syn)
#    filetag = filetag + '_tau%s'%(tau_syn) + '_k_orig_paper_keki8_'    
    if computeType == 'test':
        print "Running Poission spike test"
        nTrials = 50000
        NNN = 10000
#        spkArrayList = GenPoissionSpkies(rates, 50, nTrials)
        spkArrayList = GenPoissionSpkies_Alt_misc(nTrials, NNN)
#        print spkArrayList
        fanoFactor = AvgFano(spkArrayList, np.arange(NNN), nTrials, 0, 0.0, 'poission_test')
#        print fanoFactor
        print 'done'
        np.save('fanofactor_poission_test', fanoFactor)
    else:
        for i in range(nTrials):
            filename = bf + foldername +'/spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_3000_tr%s.csv'%(int(alpha), int(tau_syn), i)
            print 'loading file: ', filename
            spkArrayList.append(np.loadtxt(filename, delimiter = ';'))
        neuronsList = np.arange(NE+NI)
        print 'computing fano factor ...',
        sys.stdout.flush()
        fanoFactor = AvgFano(spkArrayList, neuronsList, nTrials, alpha, 1000.0, filetag)
        print 'done'
        np.save('./data/fanofactor_p%s_'%(int(alpha)) + filetag, fanoFactor)
#        np.save('./data/fanofactor_kffi400_p%s_'%(int(alpha)) + filetag, fanoFactor)
#        np.save('./data/fanofactor_k_orig_paper_p%s_'%(int(alpha)) + filetag, fanoFactor)        
    
    


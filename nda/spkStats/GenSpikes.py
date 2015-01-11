import numpy as np
import scipy.stats as stat
import code, sys, os
import pylab as plt
sys.path.append("/home/shrisha/Documents/code/mypybox")
import Keyboard as kb
from enum import Enum
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append("/home/shrisha/Documents/code/mypybox/nda/spkStats")
#import SpkStats
sys.path.append("/home/shrisha/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
import time

def ExpRNG(rate):
    return (-1 * np.log(np.random.rand()) / rate)

def SpikeTimes(rate, t, dt):
    # GENERATE POISSON SPIKES WITH LAMDA = RATE IN THE INTERVAL [0, TSTOP) 
    # time in s
    # rate in Hz ( 1 / sec)
    # RETURNS SPIKE TIMES in ms
#    dt = 0.001 # 1ms
    rate = float(rate)
    spkProbInDt = np.random.rand(t.size)
    IF_SPK = (rate * dt) > spkProbInDt
    return np.squeeze(np.array(np.where(IF_SPK == True)) * dt)

def SpikeTimesISI(rate, tStop, mNeuron):
    # GENERATE POISSON SPIKES WITH LAMDA = RATE IN THE INTERVAL [0, TSTOP) 
    # time in s
    # rate in Hz ( 1 / sec)
    # RETURNS SPIKE TIMES in ms
    rate = float(rate)
    print rate
    st = np.empty((0))
    N = int(rate * tStop) # estimate number of spike expected
    np.random.seed(int(time.clock() * float(mNeuron)))
    isi = (-1 * np.log(np.random.rand(N)) / rate)
    st = np.concatenate((st, np.cumsum(isi)), axis = 0)
    IF_LOOP = False
    if st[-1] <= tStop:
        IF_LOOP = True
    while(IF_LOOP):
        isi = ExpRNG(rate)
        st = np.concatenate((st, np.array([st[-1] + isi])), axis = 0)
        if st[-1] <= tStop:
            IF_LOOP = True
        else:
            IF_LOOP = False
            st = st[:-1]
    return st*1e3 # return spk times in ms

def Trials(nNeurons, rate, trialId):
    out = np.empty((0, 2))
    print trialId
    for mNeuron in np.arange(nNeurons):
        st = SpikeTimesISI(rate[mNeuron], tStop, mNeuron)
        tmp = np.transpose(np.array([st, np.ones((st.size)) * mNeuron]))
        out = np.concatenate((out, tmp), axis = 0)
    filename = './tmpData/spk_tr%s'%(trialId) + '.csv'
    fp = open(filename,'w')
    np.savetxt(fp, out, fmt='%.5f', delimiter = ';')
    fp.close()

def PlotScatter(spkFilename, winsize):
    st = np.loadtxt(spkFilename, delimiter = ';')
    neurons = np.unique(st[:, 1])
    nNeurons = neurons.size
    plt.ion()
    ff = np.empty((nNeurons, ))
    print ff.shape
    ff[:] = np.nan
    for kk, kNeuron in enumerate(neurons):
        kSpks = st[st[:, 1] == kNeuron, 0]
        if(kSpks.size > 2):
            bins = np.arange(0.0, kSpks[-1], winsize)
            cnts, _ = np.histogram(kSpks, bins)
            plt.plot(np.mean(cnts), np.var(cnts), 'ko')
            ff[kk] = np.var(cnts) / np.mean(cnts)
    xlim = plt.xlim()[1]
    plt.plot(np.arange(xlim), np.arange(xlim), 'r')
    plt.figure()
    ff = ff[~(np.isnan(ff))]
    plt.hist(ff, 25)

if __name__ == '__main__':
    nTrials = 1
    nNeurons = 100
    rate = np.random.rand(nNeurons) * 10.0
    dt = 0.001 # 1ms
    tStop = 500.0 # in seconds
    t = np.arange(0.0, float(tStop), dt)
    runType = 'compute' 
    runType = 'plot'

    if(runType == 'compute'):
        print 'generating spks'
        p = Pool(4)
        func = partial(Trials, nNeurons, rate)
        p.map(func, np.arange(nTrials))
        p.close()
    else :
        print 'plotting'
        PlotScatter('./tmpData/spk_tr%s'%(0) + '.csv', 50.0)
        kb.keyboard()

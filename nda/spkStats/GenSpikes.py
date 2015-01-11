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
import SpkStats
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
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

def Trials(nNeurons, trialId):
    out = np.empty((0, 2))
    print trialId
    
    for mNeuron in np.arange(nNeurons):
        st = SpikeTimesISI(rate, tStop, mNeuron)
        tmp = np.transpose(np.array([st, np.ones((st.size)) * mNeuron]))
        out = np.concatenate((out, tmp), axis = 0)
    filename = './tmpData/spk_tr%s'%(trialId) + '.csv'
    fp = open(filename,'w')
    np.savetxt(fp, out, fmt='%.5f', delimiter = ';')
    fp.close()

if __name__ == '__main__':
    nTrials = 100
    nNeurons = 500
    rate = 2.0
    dt = 0.001 # 1ms
    tStop = 20.0 # in seconds
    t = np.arange(0.0, float(tStop), dt)
    p = Pool(16)
    func = partial(Trials, nNeurons)
    p.map(func, np.arange(nTrials))
    p.close()


    # for kTr in np.arange(nTrials):
    #     out = np.empty((0, 2))
    #     print kTr
    #     for mNeuron in np.arange(nNeurons):
    #         st = SpikeTimes(rate, t, dt)
    #         tmp = np.transpose(np.array([st, np.ones((st.size)) * mNeuron]))
    #         out = np.concatenate((out, tmp), axis = 0)
    #     filename = './tmpData/spk_tr%s'%(kTr) + '.csv'
    #     np.savetxt(filename, out, fmt='%.5f', delimiter = ';')

    

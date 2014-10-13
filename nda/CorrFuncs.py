import MySQLdb as mysql
import numpy as np
import code
import sys
import pylab as plt

#plt.ion()

sys.path.append("/homecentral/srao/Documents/code/mypybox")

import Keyboard as kb

def PadZeros(a, n):
    return np.concatenate((a, np.zeros((n,))))

def autocorrelation(spike_times, bin_width=0.025, width=100, T=None):
    """Given the sorted spike train 'spike_times' return the
    autocorrelation histogram, as well as the bin edges (including the
    rightmost one). The bin size is specified by 'bin_width', and lags are
    required to fall within the interval [-width, width]. The algorithm is
    partly inspired on the Brian function with the same name."""

    d = []                    # Distance between any two spike times
    n_sp = len(spike_times)  # Number of spikes in the input spike train

    i, j = 0, 0
    for t in spike_times:
        # For each spike we only consider those spikes times that are at most
        # at a 'width' time lag. This requires finding the indices
        # associated with the limiting spikes.
        while i < n_sp and spike_times[i] < t - width:
            i += 1
        while j < n_sp and spike_times[j] < t + width:
            j += 1
        # Once the relevant spikes are found, add the time differences
        # to the list
        d.extend(spike_times[i:j] - t)


    n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    b = np.linspace(-width, width, 2 * n_b, endpoint=True)
#    h = np.histogram(d, bins=b, new=True)
    h = np.histogram(d, bins=b)
    H = h[0] # number of entries per bin

    # Compute the total duration, if it was not given
    # (spike trains are assumed to be sorted sequences)
    if T is None:
        T = spike_times[-1] - spike_times[0] # True for T >> 1/r

    # The sample space gets smaller as spikes are closer to the boundaries.
    # We have to take into account this effect.
    W = T - bin_width * abs( np.arange(n_b - 1, -n_b, -1) )
    tmp0 =  H / (bin_width * (T - width))
    kb.keyboard()
 #   return ( H/W - n_sp**2 * bin_width / (T**2), b)
#     return (H / (bin_width * (T - width)), b)
    return (H / (bin_width), b)

def AutoCorr(x, corrLength = "same"):
    # x : spike Times in ms
#    print "cmputing FFT", len(x)
    N = len(x)
    nPointFFT = int(np.power(2, np.ceil(np.log2(len(x)))))
    fftX = np.fft.fft(x, nPointFFT)
 #   print "computing IFFT"
    return np.abs(np.abs(np.fft.ifft(np.multiply(fftX, np.conj(fftX)))))

def SpkTime2Vector(spkTimes, simDT, IF_SUBSTRACT_MEAN = True):
    # spkTimes in ms
    # simulation time step in ms
    print "inside spkTime 2 vector"
    nSpks = len(spkTimes)
    print nSpks, np.ceil((nSpks - 1) / simDT), spkTimes[nSpks - 1]
#    spkVector = np.zeros((np.ceil(spkTimes[nSpks - 1] / simDT) + 1 ,))
    spkVector = np.zeros((np.ceil(spkTimes[nSpks - 1] / simDT) + 1 ,))
    spkBins = (spkTimes / simDT).astype(int);
    if(IF_SUBSTRACT_MEAN):
        spkVector[spkBins] = 1 - (nSpks / np.floor((nSpks - 1) / simDT))
#        print  1 - (nSpks / np.floor((nSpks - 1) / simDT))
    else :
        spkVector[spkBins] = 1
        print 'here'
    return spkVector

def AvgAutoCorr(neuronsList, dbName = "tstDb", theta = 0, simDT = 0.025, minSpks = 0, maxTimeLag = 100, simDuration = 25000, NE = 10000, NI = 10000):
    N = len(neuronsList)
    #avgCorr = list() 
    
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    pcDone = 0
#    avgCorr = np.zeros((int((maxTimeLag * 2)/ simDT) - 1, ))
#    nTimeLagBins = int(np.power(2, np.ceil(np.log2(simDuration + simDT)))) / 2
    nTimeLagBins = int(2 * maxTimeLag) # works if downsample bin size = 1ms otherwise divide by value
    avgCorr = np.zeros((int(np.power(2, np.ceil(np.log2(simDuration + simDT)))), ))
#    avgCorr = np.zeros((nTimeLagBins, ))
    binFlag = 0;
    nValidNeurons = 0;
    downSampleBinSize = 1
    spkBins = np.arange(0, simDuration + simDT, downSampleBinSize)
    nSpkBins = len(spkBins) ;
    #print "MEAN RATE E = ", float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId < %s AND theta = %s", (NE, theta))) / (simDuration * 1e-3 * NE)
    #print "MEAN RATE I = ", float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId > %s AND theta = %s", (NE, theta))) / (simDuration * 1e-3 * NI)
    avgRate = 0
    
    for i, kNeuron in enumerate(neuronsList):
        spkTimes = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s", (kNeuron, theta))
        meanRate = float(spkTimes) / float(simDuration * 1e-3)
#        print "mean rate = ", meanRate
        nValidNeurons += 1
        avgRate += meanRate
        if(spkTimes > minSpks):
#            print "neuron :", kNeuron, "#spks :", spkTimes, "i = ", i
            st = np.squeeze(np.asarray(dbCursor.fetchall()))
            st = np.histogram(np.squeeze(st), spkBins)
#            st = SpkTime2Vector(st, simDT)
            tmpCorr = AutoCorr(st[0])
#            tmpCorr = autocorrelation(st, bin_width = simDT, width = maxTimeLag, T = simDuration)

#            kb.keyboard()

            avgCorr += tmpCorr / ((downSampleBinSize * 1e-3) **2 * nSpkBins * meanRate)
            #avgCorr += tmpCorr 
            #            avgCorr += tmpCorr[0]
#            avgCorr += np.concatenate((tmpCorr[np.arange(nTimeLagBins/ 2 -1)],

            #            print "DONE"

#            tmpCorr = tmpCorr[np.argmax(tmpCorr):]  # right half
#            avgCorr.append(tmpCorr[0]) # zero is correlations and 1 is time lag bins
            # if(~binFlag):
            #     bins = tmpCorr[1][:-1]
            #     binFlag = 1;

            if(len(neuronsList) > 10):
                if(not bool(np.mod(i, 200))):
#                if(~np.mod(i, int(len(neuronsList) * 0.1))):
                    pcDone = 100.0 * float(i) / N
                    print "%s%% " %((pcDone,))
       


    dbCursor.close()
    db.close()
    avgCorr = avgCorr / nValidNeurons
    print "avg rate ", avgRate / nValidNeurons
    print "#valide nuruons = ", nValidNeurons
#    avgCorr = np.array(np.roll(avgCorr, -avgCorr.size/2))
    bins = np.array(downSampleBinSize)
    print avgCorr.size, bins.size
    if(len(avgCorr) > 0):
        filename = 'avgCorr_' + dbName
        print "saving as ", filename
        np.save(filename, avgCorr)
        return avgCorr
    else :
        return 0

if __name__ == "__main__":
    n = 1000
    useDb =  "biII_3"
    N_NEURONS = 10000
    IF_UNIQUE = False
    neuronsList = np.unique(np.random.randint(0, N_NEURONS, size = n + 00))
    # while(not IF_UNIQUE):
    #     neuronsList = np.unique(np.random.randint(0, N_NEURONS, size = n + 100))
    #     print len(neuronsList)
    #     if(len(neuronsList) >= n):
    #         print "TRUE"
    #         IF_UNIQUE = True
    #         neuronsList = neuronsList[0:n]
               
    print "#neurons = ", len(neuronsList)
#    neuronsList = [12]
    ac = AvgAutoCorr(neuronsList, useDb, theta = 0, simDuration = 10000, simDT = 0.05, NE = 19600, NI = 19600)
    ac[np.argmax(ac)] = 0.0
    bins = np.arange(-100, 100, 1)
    ac0 = ac;
    print bins.shape, ac.shape
#    plt.bar(bins, np.concatenate((np.flipud(ac[-1:-1-100:-1]), ac[0:100] )), fc = 'k', edgecolor = 'k')

    plt.plot(bins, np.concatenate((np.flipud(ac[-1:-1-100:-1]), ac[0:100] )), 'k') # E auto-corr
    plt.xlabel('Time lag (ms) ')
#    plt.ylabel('raw correlation : IFFT{ S(f) x S(f)* }')
    plt.ylabel('Firing Rate (Hz)')

    neuronsList = np.unique(np.random.randint(N_NEURONS, 2 * N_NEURONS, size = n + 00))
    ac = AvgAutoCorr(neuronsList, useDb, theta = 0, simDuration = 10000, simDT = 0.05, NE = 19600, NI = 19600)
    ac[np.argmax(ac)] = 0.0
    plt.plot(bins, np.concatenate((np.flipud(ac[-1:-1-100:-1]), ac[0:100] )), 'r') # I auto-corr
    plt.legend(('E', 'I'))
    
#    plt.xlim((min
#    plt.show()
    filename = "avg_autocorr_" + useDb
    plt.savefig(filename + ".png")
 #   plt.waitforbuttonpress()
    



        

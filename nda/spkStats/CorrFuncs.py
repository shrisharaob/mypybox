import MySQLdb as mysql
import numpy as np
import code
import sys
import pylab as plt
from multiprocessing import Pool
from functools import partial 

#plt.ion()

sys.path.append("/homecentral/srao/Documents/code/mypybox")
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")


import Keyboard as kb
from DefaultArgs import DefaultArgs

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
    N = len(x)
    nPointFFT = int(np.power(2, np.ceil(np.log2(len(x)))))
    fftX = np.fft.fft(x, nPointFFT)
    return np.abs(np.abs(np.fft.ifft(np.multiply(fftX, np.conj(fftX)))))

def SpkTime2Vector(spkTimes, simDT, IF_SUBSTRACT_MEAN = True):
    # spkTimes in ms
    # simulation time step in ms
    print "inside spkTime 2 vector"
    nSpks = len(spkTimes)
    print nSpks, np.ceil((nSpks - 1) / simDT), spkTimes[nSpks - 1]
    spkVector = np.zeros((np.ceil(spkTimes[nSpks - 1] / simDT) + 1 ,))
    spkBins = (spkTimes / simDT).astype(int);
    if(IF_SUBSTRACT_MEAN):
        spkVector[spkBins] = 1 - (nSpks / np.floor((nSpks - 1) / simDT))
    else :
        spkVector[spkBins] = 1
        print 'here'
    return spkVector

def AvgAutoCorr(neuronsList, dbName = "tstDb", simDT = 0.025, minSpks = 0, maxTimeLag = 100, simDuration = 25000, NE = 10000, NI = 10000, fileTag = 'E', theta = 0):
    N = len(neuronsList)
    print "theta = ", theta
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
        nValidNeurons += 1
        avgRate += meanRate
        if(spkTimes > minSpks):
            st = np.squeeze(np.asarray(dbCursor.fetchall()))
            st = np.histogram(np.squeeze(st), spkBins)
#            st = SpkTime2Vector(st, simDT)
            tmpCorr = AutoCorr(st[0])
            avgCorr += tmpCorr / ((downSampleBinSize * 1e-3) **2 * nSpkBins * meanRate)
            #avgCorr += tmpCorr 
            #            avgCorr += tmpCorr[0]
#            avgCorr += np.concatenate((tmpCorr[np.arange(nTimeLagBins/ 2 -1)],
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
        filename = 'avgCorr_' + fileTag + '_' + dbName
#        print "saving as ", filename
#        np.save(filename, avgCorr)
        return avgCorr
    else :
        return 0

if __name__ == "__main__":
    computeType = 'plot'
    if(len(sys.argv) > 2):
        computeType = sys.argv[2]
    n = 5000
    useDb = sys.argv[1] #supply dbname
    N_NEURONS = 10000
    NE = 10000
    NI = 10000
    IF_UNIQUE = False
    #useTheta = np.array([0, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    useTheta = np.array([30, 60, 80, 100, 120])
    simT = 100000
    dt = 0.05
    ne = 10000
    ni = 10000
    maxLag = 5000
    bins = np.arange(-500, 500, 1)
    filetag = 'I'    
#    useTheta = np.array([401])
    if(computeType == 'compute'):
        neuronsList = np.unique(np.random.randint(0, N_NEURONS, size = n + 00))
        acMat = np.zeros((useTheta.size, maxLag))
#        acMat[0, :] = np.arange(0, maxLag, 1)
        listOfneurons = np.unique(np.random.randint(NE, NE + NI, size = n + 00))
        #listOfneurons = np.arange(NE, NE + NI)
        print '#neurons = ', len(listOfneurons)
     #   nPools = np.min(10, len(useTheta))
        p = Pool(5)
        result = p.map(partial(AvgAutoCorr, listOfneurons, useDb, dt, 0, 200.0, simT, ne, ni, '_I_tau'), useTheta)
        for kk, kTheta in enumerate(useTheta):
            ac = result[kk]
            ac[np.argmax(ac)] = 0.0
            acMat[kk, :] = ac[:maxLag]
        print "saving as ", '../data/long_tau_vs_ac_mat' + useDb
        np.save('../data/long_tau_vs_ac_mat'+useDb + '_' + filetag, acMat)
        kb.keyboard()
    else :
        ac = np.load('../data/long_tau_vs_ac_mat'+useDb +'_' + filetag +  '.npy')
        plotMaxLag = 5000
        plt.ion()
        tau = [3.0, 6.0, 8.0, 10.0, 12.0]
#	alpha = np.array([0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        alpha = np.array([0.9])
        colormap = plt.cm.gist_rainbow
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(alpha))]
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(tau))]
        plt.figure()
        plt.gca().set_color_cycle(colors)
        print ac.shape
        for kk, kTau in enumerate(tau):
            for tt, kAlpha in enumerate(alpha):
                plt.plot(np.arange(plotMaxLag) / kTau, ac[kk, :plotMaxLag], label =r'$\tau = %s, \; \alpha=%s$'%((kTau, kAlpha)))
        plt.legend()
#        plt.title(r'$\tau = $%s, %s neurons'%(tau, filetag), fontsize = 20)
        plt.title(r'$\alpha = $%s, %s neurons'%(alpha[0], filetag), fontsize = 20)
        plt.xlabel('Time (ms)', fontsize = 20)
        plt.ylabel('Firing rate (Hz)', fontsize = 20)
        plt.tick_params(axis = 'both', labelsize = 16)
        kb.keyboard()





#((240,163,255),(0,117,220),(153,63,0),(76,0,92),(25,25,25),(0,92,49),(43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),(194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),(224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(255,80,5))

#----------------------------------------------    
#     ac[np.argmax(ac)] = 0.0
#     plt.figure()
#     plt.plot(bins, np.concatenate((np.flipud(ac[-1:-1-maxLag:-1]), ac[0:maxLag] )), 'r', label = 'I') # I auto-corr
# #    plt.legend(('E', 'I'))
#     plt.legend()
#     filename = "avg_autocorr_I_" + useDb + "tau12_50s"
#     plt.savefig(filename + ".png")

#AvgAutoCorr(neuronsList, dbName = "tstDb", simDT = 0.025, minSpks = 0, maxTimeLag = 100, simDuration = 25000, NE = 10000, NI = 10000, fileTag = 'E', theta = 0):


    # while(not IF_UNIQUE):
    #     neuronsList = np.unique(np.random.randint(0, N_NEURONS, size = n + 100))
    #     print len(neuronsList)
    #     if(len(neuronsList) >= n):
    #         print "TRUE"
    #         IF_UNIQUE = True
    #         neuronsList = neuronsList[0:n]
               
#     print "#neurons = ", len(neuronsList)
# #    neuronsList = [12]

#     ac = AvgAutoCorr(neuronsList, useDb, theta = useTheta, simDuration = simT, simDT = dt, NE = ne, NI = ne, fileTag = 'E_tau%s'%((useTheta)))
#     ac[np.argmax(ac)] = 0.0


#----------------------------
#     ac0 = ac;
#     print bins.shape, ac.shape
#     plt.plot(bins, np.concatenate((np.flipud(ac[-1:-1-maxLag:-1]), ac[0:maxLag] )), 'k', label='E') # E auto-corr
#     plt.xlabel('Time lag (ms) ')
#     plt.ylabel('Firing Rate (Hz)')
#     plt.legend()
#     filename = "avg_autocorr_E_" + useDb + "tau12_50s"
#     plt.savefig(filename + ".png")

#----------------------------------

#    for kk, kTheta in enumerate(useTheta):
    #     print kTheta
    #     neuronsList = np.unique(np.random.randint(NE, NE + NI, size = n + 00))
    #     ac = AvgAutoCorr(neuronsList, useDb, theta = kTheta, simDuration = simT, simDT = dt, NE = ne, NI = ne, fileTag = 'DEL_I_tau%s'%((kTheta)))
    #     acMat[kk + 1, :] = ac[0:maxLag]
#-------------------------------------
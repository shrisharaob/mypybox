import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
from enum import Enum
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


sys.path.append("/homecentral/srao/Documents/code/mypybox")


def MeanFiringRate(dbName, NE, NI, simDuration, discardTime, theta):
#    print "simDuration = ", simDuration,  "theta = ", theta, 
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId < %s AND theta = %s AND spkTimes > %s", (NE, theta, discardTime))
    print "mean E firing rate = ", float(nSpks) / ((simDuration - discardTime) * NE * 1e-3), "theta = ", theta
    nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId > %s AND theta = %s AND spkTimes > %s", (NE, theta, discardTime))
    print "mean I firing rate = ", float(nSpks) / ((simDuration - discardTime) * NI * 1e-3), "theta = ", theta


def FanoFactor(spkTimes, binSize, simDuration = 10000):
    fanoFac = 0.0
    meanSpkCount = 0.0
    spkCntVar = 0.0
    fr = 0.0
    if(spkTimes.size > 0):
        bins = np.arange(0, simDuration+0.0001, binSize)
        counts, bins = np.histogram(spkTimes, bins)
        counts.astype(float)        
        meanSpkCount = np.mean(counts[1:-1])
        spkCntVar = np.var(counts[1:-1])
        fr = float(np.sum(spkTimes > 2000.0)) / ((simDuration)*1e-3)
        if(meanSpkCount != 0):
            fanoFac = spkCntVar/ meanSpkCount # the first and last bins are discarded
    return meanSpkCount, spkCntVar, fanoFac, fr

def AvgFano(dbName, neuronsList, simDuration, simDT, binSize, theta = 0):
    # binsize in ms
    print "fldr nda/spkStats"
    nNeurons = len(neuronsList)
    avgFanoFactor = 0;
    nValidNeurons = 0;
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    out = np.empty((nNeurons, 2))
    out[:] = np.nan
    fanoFac = np.zeros((nNeurons,))
    dicardT = 2000.0
    validDuration = simDuration - dicardT
    fr = np.zeros(neuronsList.shape)
    for i, kNeuron in enumerate(neuronsList):
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s", (kNeuron, theta))
        nValidNeurons += 1
        if(nSpks > 0):
            spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))
            spkTimes = spkTimes[spkTimes < simDuration]
            spkTimes = spkTimes[spkTimes > dicardT]
            tmp = FanoFactor(spkTimes, binSize, validDuration)
            out[i, 0] = tmp[0]
            out[i, 1] = tmp[1]
            fanoFac[i] = tmp[2]
            fr[i] = tmp[3]
    dbCursor.close()
    db.close()
    return out, np.mean(fanoFac), fr

def ObjQuad(x, a, b, c):
    return a * x**2 + b * x + c

def ObjLin(x, m, l):
    return x * m + l

def FanoFit(dbName, filename):
    meanVar = np.load(filename)
    meanVar = mv[~np.isnan(mv).any(axis = 1)] # remove rows with nans
    out = np.polyfit(meanVar[:, 0], meanVar[:, 1], 2, full = True) #fit quadratic
    coeff = out[0]
    return coeff

def SerialCor(spkTimes):
    out = (0.0, np.nan)
    if(spkTimes.size > 20):
       isi = np.diff(spkTimes)
       out =  stat.spearmanr(isi[0:-1], isi[1:]) # return 1st-order spearman rank correrelation coefficient
    return out

def SerialCorDistr(dbName, neuronsList, theta = 0, simDuration = 50000.0):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    sc = np.zeros((neuronsList.size, ))
    pVal = np.zeros((neuronsList.size, ))
    out = np.zeros((neuronsList.size, 3))
    alpha = 0.05
    for k, kNeuron in enumerate(neuronsList):
        #print k
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s", (kNeuron, theta))
 #       print nSpks
        if(nSpks > 0):
            spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))
            tmp = SerialCor(spkTimes)
            sc[k] = tmp[0]
      #      fr[k] = float(nSpks) / simDuration * 1e-3
            pVal[k] = tmp[1]
            out[k, :] = np.array([tmp[0], tmp[1], float(nSpks) / simDuration * 1e-3])
    dbCursor.close()
    db.close()
    np.save('serialCor', out)
    return (sc, pVal)

if __name__ == "__main__":
    dbName = '' #sys.argv[1]
    argc = len(sys.argv)
    n = 10000
    NE = 10000
    NI = 10000
    simDuration = 6000.0
    simDT = 0.05
    binsize = 2000 #ms

    theta = np.arange(0, 10, 1)

    kTheta = 350
    
    # class Sat2Compute(Enum):
    #     fano = 0
    #     serialCorr = 1
    #     cv = 2
    
    Stat2Compute = Enum('Stat2Compute', fano = 0, serialCorr = 1, cv = 2, isiDistr = 3, fanoVsT = 4, fano_neuronwise = 5)
    computeStatType = Stat2Compute.fano_neuronwise

    if(argc > 1):
        dbName = sys.argv[1]
        print dbName
        print "dbname = ", dbName
    if(argc > 2):
        n = int(sys.argv[2])
        print "n = ", n
    if(argc > 3):
        if(int(sys.argv[3]) == 0):
               computeStatType = Stat2Compute.fano
               print "computing fano factor"
        if(int(sys.argv[3]) == 1):
               computeStatType = Stat2Compute.serialCorr
               print "computing serial correlation"
        if(int(sys.argv[3]) == 2):
               computeStatType = Stat2Compute.cv
               print "computing cv"
        if(int(sys.argv[3]) == 3):
               computeStatType = Stat2Compute.isiDistr
               print "computing isi"
        if(int(sys.argv[3]) == 4):
               computeStatType = Stat2Compute.fanoVsT
               print "computing fano vs sim T"

    print "simDuration = ", simDuration, " bin size = ", binsize, " NE = ", NE, " NI = ", NI, " #neurons in list = ", n 

    if(computeStatType == Stat2Compute.serialCorr):
        neuronsList = np.unique(np.random.randint(NE, NE+NI, size = n))
    #    neuronsList = np.array([1])
        out = SerialCorDistr(dbName, neuronsList, theta = kTheta)
        #    sc = out:, 0]
        #   fr = out[:, 2]
        sc = out[0]
        pVal = out[1]
        print "done"
        sc = sc[~np.isnan(sc)]
    #    kb.keyboard()
        print pVal.shape
        counts, bins, patches = plt.hist(sc, 25)
        counts, bins, patches = plt.hist(sc[pVal<0.05], 25, label='pVal < 0.05')
        #    kb.keyboard()
        #   for kk, kBin in enumerate(bins[:-1]):
        #      plt.text(kBin, counts[kk]+0.1, '%.2f' %((fr[kk],)))
        plt.draw()
        plt.xlabel('Serial Correlation')
        plt.ylabel('Count')
        plt.title('I neurons')
        plt.legend()
        plt.waitforbuttonpress()
        plt.savefig('serialCorr')
        #plt.waitforbuttonpress()

    if(computeStatType == Stat2Compute.fano):
        IF_COMPUTE = False
        if(IF_COMPUTE == True):
            theta = np.array([350.0, 650, 850, 1050, 1250, 2460])
 #           theta = np.array([650])
            tau = np.array([4, 6, 8, 10])
            confidenceLevel = 1.96
        #    print AvgFano(dbName, neuronsList, simDuration, simDT, binsize)
            print "E neurons"
            out = np.empty((0, 2))
            fanofactor = np.zeros((len(theta), ))
            quadParameters = np.zeros((len(theta), 6))
            linParameters = np.zeros((len(theta), 4))
            for kk, kTheta in enumerate(theta):
                neuronsList = np.unique(np.random.randint(0, NE, size = n))
                print "theta = ", np.squeeze(kTheta)
                tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
                out = np.concatenate((out, tmp[0]), axis = 0)
                fanofactor[kk] = tmp[1]
                out = out[~np.isnan(out).any(axis=1)]
                y = tmp[0]; y = y[~np.isnan(y).any(axis=1)]
                popt, pcov = curve_fit(ObjLin, y[:, 0], y[:, 1], p0 = (0.5e-3, 0.5e-3))
                linParameters[kk, :] = np.concatenate((popt, confidenceLevel * np.sqrt(np.diag(pcov))))
                popt, pcov = curve_fit(ObjQuad, y[:, 0], y[:, 1], p0 = (0.5e-3, 0.5e-3, 1.0))
                quadParameters[kk, :] = np.concatenate((popt, confidenceLevel * np.sqrt(np.diag(pcov))))
        
#        fanoFitParams = (linParameters, quadParameters)
#        np.save('fanoFactor_E_' + dbName, fanofactor)
#       np.save('fano_mean_var_E_' + dbName, out)
#                np.savez('FanoFitParams_E_'+dbName, linParameters=linParameters, quadParameters=quadParameters)
        
                print "I neurons"
                f, ax = plt.subplots(2, 3)
                out = np.empty((0, 2))
                fanofactor = np.zeros((len(theta), ))
                quadParameters = np.zeros((len(theta), 6))
                linParameters = np.zeros((len(theta), 4))
                
                for kk, kTheta in enumerate(theta):
                    neuronsList = np.unique(np.random.randint(NE, NE+NI, size = n))
            #           neuronsList = np.array([3486])
                    print "theta = ", np.squeeze(kTheta)
                    tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
#                    kb.keyboard()
                    out = np.concatenate((out, tmp[0]), axis = 0)
#                    out = out[~np.isnan(out).any(axis=1)]
                    fanofactor[kk] = tmp[1]
                    y = tmp[0]; y = y[~np.isnan(y).any(axis=1)]
                    xx = np.linspace(0, np.max(y[:, 0]) + 1, 200)    
                    ax[np.unravel_index(kk, ax.shape)].plot(y[:, 0], y[:, 1], 'k.')
                    
                    popt, pcov = curve_fit(ObjLin, y[:, 0], y[:, 1], p0 = (0.5e-3, 0.5e-3))
                    
                    yy = ObjLin(xx, *popt)
                    ax[np.unravel_index(kk, ax.shape)].plot(xx, yy, 'r')
                    
                    linParameters[kk, :] = np.concatenate((popt, confidenceLevel * np.sqrt(np.diag(pcov))))
                    popt, pcov = curve_fit(ObjQuad, y[:, 0], y[:, 1], p0 = (0.5e-3, 0.5e-3, 1.0))

                    yy = ObjQuad(xx, *popt)
                    ax[np.unravel_index(kk, ax.shape)].plot(xx, yy, 'g')
                    ax[np.unravel_index(kk, ax.shape)].set_title(r'$\tau_{syn}=%s$'%((tau[kk], )))
                    print popt
                    quadParameters[kk, :] = np.concatenate((popt, confidenceLevel * np.sqrt(np.diag(pcov))))

#    np.save('fanoFactor_I_' + dbName, fanofactor)
                np.save('fano_mean_var_I_' + dbName, out)
            
#        fanoFitParams = (linParameters, quadParameters)
                np.savez('FanoFitParams_I_'+dbName, linParameters=linParameters, quadParameters=quadParameters)
                plt.show()
                plt.waitforbuttonpress()

        else :
            print "here else"
            tau = np.array([3, 6, 8, 10, 12, 24])
            y = np.load('FanoFitParams_I_'+dbName+'.npz')
            quadParameters = y['quadParameters']
            linParameters = y['linParameters']
            fig, ax1 = plt.subplots()
            ax1.errorbar(tau, quadParameters[:, 0], ecolor = 'k', yerr=1.96*np.sqrt(quadParameters[:, 3]))
            ax1.plot(tau, quadParameters[:, 0], 'k', label='a')
            ax1.set_ylabel('a', fontsize=20, color='k')
            for t1 in ax1.get_yticklabels():
                t1.set_color('k')
            ax1.xaxis.grid(True)
            ax1.set_xlabel(r'$\tau_{syn}$', fontsize = 20)
            plt.legend(loc=2)

#            plt.show()

#            plt.figure()
            ax2 = ax1.twinx()
            ax2.errorbar(tau, quadParameters[:, 1], ecolor='r', yerr=1.96*np.sqrt(quadParameters[:, 4]))
            ax2.plot(tau, quadParameters[:, 1], 'r', label='b')

            ax2.set_ylabel('b', fontsize=20, color = 'r')
            for t1 in ax2.get_yticklabels():
                t1.set_color('r')

            ax2.xaxis.grid(True)
            #plt.errorbar(tau, quadParameters[:, 2], yerr=1.96*np.sqrt(quadParameters[:, 5]), label='c')
            plt.xticks(tau)
            plt.legend()
            plt.title(r'$ax^{2}+bx+c$', fontsize=20)
            plt.grid()

            fig, ax1 = plt.subplots()
            ax1.errorbar(tau, linParameters[:, 0], ecolor = 'k', yerr=1.96*np.sqrt(linParameters[:, 2]))
            ax1.plot(tau, linParameters[:, 0], 'k', label='m')
            ax1.set_ylabel('m', fontsize=20)
            ax1.xaxis.grid(True)
            ax1.set_xlabel(r'$\tau_{syn}$', fontsize = 20)
            plt.title(r'$mx+l$', fontsize=20)
            plt.legend(loc=2)

            ax2 = ax1.twinx()
            ax2.errorbar(tau, linParameters[:, 1], ecolor='r', yerr=1.96*np.sqrt(linParameters[:, 3]))
            ax2.plot(tau, linParameters[:, 1], 'r', label='l')

            ax2.set_ylabel('l', fontsize=20)
            ax2.xaxis.grid(True)
            #plt.errorbar(tau, quadParameters[:, 2], yerr=1.96*np.sqrt(quadParameters[:, 5]), label='c')
            plt.xticks(tau)
            plt.legend()
            plt.grid()



            plt.show()





            plt.waitforbuttonpress()
#        plt.errorbar(tau, linParameters[:, 0], yerr = 1.96 * np.sqrt
        

            
    if(computeStatType == Stat2Compute.fanoVsT):
        kTheta = 3
        dbName = "tau3_alpha1_100"
        n = 10000
        binSizes = np.arange(2000, 10001, 2000)
        fanoFactor = np.zeros((binSizes.size, 3))
        fanoFactor[:, 2] = binSizes
        def returnFanoFactor(neuronsList, binSize):
            print "in return fano factor"
            dbName = "tau3_alpha1_100"
            simDuration = 100000
            simDT = 0.05
            kTheta = 3
            db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
            dbCursor = db.cursor()
            db.autocommit(True)
            discardTime = 2000.0
            meanSpkCount = np.zeros(neuronsList.shape)
            varSpkCount = np.zeros(neuronsList.shape)
            bins = np.arange(discardTime, simDuration+1, binSize) 
            for i, kNeuron in enumerate(neuronsList):
                nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s", (kNeuron, kTheta))
#                print nSpks
                if(nSpks > 0):
                    spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))
                    spkTimes = spkTimes[spkTimes > discardTime]
                    hist = np.histogram(spkTimes, bins)
                    spkCounts = hist[0]
                    meanSpkCount[i] = np.mean(spkCounts)
                    varSpkCount[i] = np.var(spkCounts)
#            y = tmp[0]; y = y[~np.isnan(y).any(axis=1)]
            popt, pcov = curve_fit(ObjLin, meanSpkCount, varSpkCount, p0 = (0.5e-3, 0.5e-3))
#            kb.keyboard()
            dbCursor.close()
            db.close()
            return popt[0]

        p = Pool(6)
        neuronsList = np.arange(NE) 
#        neuronsList = np.random.randint(0, NE, n) # NE
        result = p.map(partial(returnFanoFactor, neuronsList), binSizes)
        ffE = result
        

 #       neuronsList = np.random.randint(NE, NE+NI, n) # NI
        neuronsList = np.arange(NE, NE+NI, 1)
        result = p.map(partial(returnFanoFactor, neuronsList), binSizes)

        out = np.array([binSizes, ffE, result]) # bins, ffE, ffI
        np.save('fano_vs_T', out)
        ## display
        fig, ax1 = plt.subplots()
        majorFormatter = FormatStrFormatter('%1.3f')
        ax1.plot(binSizes * 1e-3, ffE, 'ks-', label='E')
        ax1.set_ylabel('fano factor')
        ax1.set_xlabel('window size (s)')
        ax1.set_yticks(ffE)
        ax1.yaxis.set_major_formatter(majorFormatter)
        ax1.grid()
        ax1.legend(loc=3)
        ax2 = ax1.twinx()
        ax2.plot(binSizes * 1e-3, result, 'ro-', label='I')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        ax2.set_xticks(binSizes * 1e-3)
        ax2.set_yticks(result)
        ax2.yaxis.set_major_formatter(majorFormatter)
        ax2.set_ylabel('fano factor')
        ax2.set_xlabel('window size (s)')
        ax2.set_title('fano factor vs observation time window length')
        ax2.grid()
        ax2.legend(loc=4)
        plt.ion()
        plt.show()
        kb.keyboard()
#        plt.waitforbuttonpress()


  
        
#        result = p.map(partial(returnFanoFactor, neuronsList), binSizes)
  #      kb.keyboard()
        # for kk, kBinSize in enumerate(binSizes):
        #     print kBinSize
        #     returnFanoFactor(neuronsList, kBinSize)

        #     #neuronsList = np.unique(np.random.randint(0, NE, size = n))
        #     neuronsList = np.arange(0, n, 1)
        #     print "theta = ", np.squeeze(kTheta)
        #     tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
        #     y = tmp[0]; y = y[~np.isnan(y).any(axis=1)]
            
        #     popt, pcov = curve_fit(ObjLin, y[:, 0], y[:, 1], p0 = (0.5e-3, 0.5e-3))
        #     fanoFactor[kk, 0] = popt[0]

        #     # I NEURONS
        #     #neuronsList = np.unique(np.random.randint(NE, NE+NI, size = n))
        #     neuronsList = np.arange(n, 2*n, 1)
        #     tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
        #     y = tmp[0]; y = y[~np.isnan(y).any(axis=1)]
        #     popt, pcov = curve_fit(ObjLin, y[:, 0], y[:, 1], p0 = (0.5e-3, 0.5e-3))
        #     fanoFactor[kk, 1] = popt[0]

        np.save('fano_vs_T', fanoFactor)

    if(computeStatType == Stat2Compute.cv):
        print "COMPUTING CV"
        def ComputeCV(spkTimes):
            cv = 0
            if(len(spkTimes > 2)):
                isi = np.diff(spkTimes)
                if(isi.size > 0):
                    mean_isi = np.mean(isi)
                    if(mean_isi > 0):
                        cv = np.std(isi) / mean_isi
            return cv
        
        def CVDistr(neuronsList, dbName, theta, discardTime):
            db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
            dbCursor = db.cursor()
            db.autocommit(True)
            cv = np.zeros(neuronsList.shape)
            for i, kNeuron in enumerate(neuronsList):
                nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s", (kNeuron, theta))
                if(nSpks > 0):
                    spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))
                    spkTimes = spkTimes[spkTimes > discardTime]            
                    cv[i] = ComputeCV(spkTimes)

            dbCursor.close()
            db.close()
            return cv

        p = Pool(8)

        useDb = dbName
        simDuration = 6000
        simDT = 0.05
        kTheta = 10
        n = 10000
        #neIdx = np.random.randint(0, NE, n) # NE
        #niIdx = np.random.randint(NE, NE+NI, n) # NI
#        rr = CVDistr(neIdx, useDb, kTheta, 2000.0)
#       kb.keyboard()
        niIdx = np.arange(NE, NE+NI, 1)
        neIdx = np.arange(NE) 
        neList = [neIdx[:n/4], neIdx[n/4:n/2], neIdx[n/2:(3*n)/4], neIdx[(3*n)/4:]]
        niList = [niIdx[:n/4], niIdx[n/4:n/2], niIdx[n/2:(3*n)/4], niIdx[(3*n)/4:]]
        poolList = neList + niList
        poollist = np.arange(10000)
        results = p.map(partial(CVDistr, dbName = useDb, theta = kTheta, discardTime = 2500.0), poolList)
        kb.keyboard()
        cnt, bins, patches = plt.hist(results[0], 50)
        plt.setp(patches, 'edgecolor', 'k', 'facecolor', 'k')
        plt.title(r'E neurons $\tau = 3ms, \; \alpha = 0$')
        plt.xlabel('CV')
        plt.ylabel('Counts')
        plt.figure()
        cnt, bins, patches = plt.hist(results[1], 50)
        plt.setp(patches, 'edgecolor', 'k', 'facecolor', 'k')
        plt.title(r'E neurons $\tau = 3ms, \; \alpha = 0$')
        plt.title('I neurons')
        plt.xlabel('CV')
        plt.ylabel('Counts')
        plt.ion()
        plt.show()

        kb.keyboard()
            
        
    if(computeStatType == Stat2Compute.fano_neuronwise):
        IF_COMPUTE = True
        if(IF_COMPUTE == True):
            tau = np.array([3])
            theta = np.array([309])
            alpha = 0.9
            print dbName, simDuration, simDT
            out = np.empty((0, 2))
            neuronsList = np.arange(NE, NE + NI, 1)
            ff_neuronwise = np.empty(neuronsList.shape)
            prows = 2
            pclms = int(np.ceil(len(theta) /float( prows)))
            f0, ax0 = plt.subplots(prows, pclms)
            f1, ax1 = plt.subplots(prows, pclms)
            plt.ion()
            print "simDuration = ", simDuration, " bin size = ", binsize, " NE = ", NE, " NI = ", NI, " #neurons in list = ", n 
            print "I neurons"
            for kk, kTheta in enumerate(theta):
                print "theta = ", np.squeeze(kTheta)
                tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
                meanSpkCnt = tmp[0][:, 0]
                spkVar = tmp[0][:, 1]
                tmpidx = ~(np.isnan(meanSpkCnt))
                tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
                meanSpkCnt = meanSpkCnt[tmpidx]
                spkVar = spkVar[tmpidx]
                ff = spkVar / meanSpkCnt
                print "mean ff = ", np.nanmean(ff)
                cnts, bins = np.histogram(ff[~(np.isnan(ff))], 50)
                #gshape, gloc, gscale = stat.gamma.fit(ff)
                #binCenters = (bins[0:-1] + bins[1:]) / 2
                subscripts = np.unravel_index(kk, (prows, pclms))
                print subscripts
                if(pclms == 1):
                    subscripts = subscripts[0]
                print subscripts
                ax0[subscripts].bar(bins[:-1], cnts, color = 'r', edgecolor = 'r', width = 0.8)
                ax0[subscripts].set_title(r'$\tau = %s \; \alpha = %s$'%(tau[kk], alpha))                
                #plt.plot(binCenters, stat.gamma.pdf(binCenters, gshape, gloc, gscale) *  float(np.sum(cnts)), 'r', linewidth = 2)
                ax1[subscripts].plot(tmp[2][tmpidx] , ff, 'r.') # fr vs ff
                ax1[subscripts].set_title(r'$\tau = %s \; \alpha = %s$'%(tau[kk], alpha))                


            print "E neurons"
            neuronsList = np.arange(NE)
            for kk, kTheta in enumerate(theta):
                print "theta = ", np.squeeze(kTheta)
                tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
                meanSpkCnt = tmp[0][:, 0]
                spkVar = tmp[0][:, 1]
                tmpidx = ~(np.isnan(meanSpkCnt))
                tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
                meanSpkCnt = meanSpkCnt[tmpidx]
                spkVar = spkVar[tmpidx]
                ff = spkVar / meanSpkCnt
                cnts, bins = np.histogram(ff[~(np.isnan(ff))], 50)
                #gshape, gloc, gscale = stat.gamma.fit(ff)
                #binCenters = (bins[0:-1] + bins[1:]) / 2

                subscripts = np.unravel_index(kk, (prows, pclms))
                if(pclms == 1):
                    subscripts = subscripts[0]
                print "kkkkk",  np.sum(cnts)

                ax0[subscripts].bar(bins[:-1], cnts, color = 'k', edgecolor = 'k', width = 0.8)
                ax0[subscripts].set_title(r'$\tau = %s \; \alpha = %s$'%(tau[kk], alpha))                
                #plt.plot(binCenters, stat.gamma.pdf(binCenters, gshape, gloc, gscale) *  float(np.sum(cnts)), 'r', linewidth = 2)
                ax1[subscripts].plot(tmp[2][tmpidx] , ff, 'k.') # fr vs ff
                ax1[subscripts].set_title(r'$\tau = %s \; \alpha = %s$'%(tau[kk], alpha))                
                print "done"
            
            print "pclms = ", pclms, "ax : ", ax0.shape
            if(pclms > 1):
                ax0[prows - 1, 0].set_ylabel('Counts', fontsize = 20)
                ax0[prows - 1, 0].set_xlabel('Fano factor', fontsize = 20)
                ax1[prows - 1, 0].set_ylabel('Fano factor', fontsize = 20)
                ax1[prows - 1, 0].set_xlabel('Firing rate (Hz)', fontsize = 20)


            
            plt.legend(('E', 'I'))
            plt.draw()
            plt.show()

            kb.keyboard()

                # plt.bar(bins[:-1], cnts, color = 'k', edgecolor = 'k')
                # plt.xlabel('fano factor', fontsize = 20)
                # plt.ylabel('counts', fontsize = 20)
                # plt.title('I neurons')
                # #plt.plot(binCenters, stat.gamma.pdf(binCenters, gshape, gloc, gscale) *  float(np.sum(cnts)), 'r', linewidth = 2)
                # plt.figure()
                # plt.plot(tmp[2][tmpidx] , ff, 'k.') # fr vs ff
                # print "done..."

                # plt.ion()
                # plt.show()

                # kb.keyboard()

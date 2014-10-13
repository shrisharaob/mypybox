import MySQLdb as mysql
import numpy as np
import code, sys
import pylab as plt

sys.path.append("/homecentral/srao/Documents/code/mypybox")

def FanoFactor(spkTimes, binSize = 2000, simDuration = 10000):
    fanoFac = 0.0
    meanSpkCount = 0.0
    spkCntVar = 0.0
#    print "simDuration = ", simDuration, binSize 
    if(spkTimes.size > 0):
        bins = np.arange(0, simDuration+0.0001, binSize)
        counts, bins = np.histogram(spkTimes, bins)
  #      print counts, bins
        counts = counts.astype(float)
        meanSpkCount = np.mean(counts[1:-1])
        spkCntVar = np.var(counts[1:-1])
        #print spkTimes.size, meanSpkCount, spkCntVar
#        print np.mean(counts[1:]), np.var(counts[1:]),
        if(meanSpkCount != 0):
            fanoFac = spkCntVar/ meanSpkCount # the first and last bins are discarded
#    print spkTimes.size, meanSpkCount, spkCntVar
    return meanSpkCount, spkCntVar, fanoFac

def AvgFano(dbName, neuronsList, simDuration, simDT, binSize = 2000, theta = 0):
    # binsize in ms
    nNeurons = len(neuronsList)
    avgFanoFactor = 0;
    nValidNeurons = 0;
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    out = np.empty((nNeurons, 2))
    out[:] = np.nan
    fanoFac = np.zeros((nNeurons,))
    for i, kNeuron in enumerate(neuronsList):
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s", (kNeuron, theta))
        nValidNeurons += 1
#        print kNeuron, nSpks
        if(nSpks > 0):
            spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))
            tmp = FanoFactor(spkTimes, binsize, simDuration)
            out[i, 0] = tmp[0]
            out[i, 1] = tmp[1]
            fanoFac[i] = tmp[2]
            #avgFanoFactor += FanoFactor(spkTimes, binsize, simDuration)
            # if(tmp[1] > 2):
            #     plt.loglog(tmp[0], tmp[1], 'ko')

    dbCursor.close()
    db.close()
    return out, np.mean(fanoFac)

def FanoFit(dbName, filename):
    meanVar = np.load(filename)
    meanVar = mv[~np.isnan(mv).any(axis = 1)] # remove rows with nans
    out = np.polyfit(meanVar[:, 0], meanVar[:, 1], 2, full = True) #fit quadratic
    coeff = out[0]
    return coeff
    
if __name__ == "__main__":
    dbName = "bg_II"
    argc = len(sys.argv)
    n = 1000
    NE = 10000
    NI = 10000
    simDuration = 25000
    simDT = 0.025
    binsize = 2000 #ms
    theta = np.arange(0, 10, 1)
    print "simDuration = ", simDuration, " bin size = ", binsize, " NE = ", NE, " NI = ", NI, " #neurons in list = ", n
    if(argc > 1):
        dbName = sys.argv[1]
        print "dbname = ", dbName
    if(argc > 2):
        n = int(sys.argv[2])
        print "n = ", n
    

#    print AvgFano(dbName, neuronsList, simDuration, simDT, binsize)
    print "E neurons"
    out = np.empty((0, 2))
    fanofactor = np.zeros((len(theta), ))
    for kTheta in theta:
        neuronsList = np.unique(np.random.randint(0, NE, size = n))
 #       neuronsList = np.array([3486])
        print "theta = ", np.squeeze(kTheta)
        tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
        out = np.concatenate((out, tmp[0]), axis = 0)
        fanofactor[kTheta] = tmp[1]

        
    np.save('fanoFactor_E_' + dbName, fanofactor)
    np.save('fano_mean_var_E_' + dbName, out)

    print "I neurons"
    out = np.empty((0, 2))
    fanofactor = np.zeros((len(theta), ))
    for kTheta in theta:
        neuronsList = np.unique(np.random.randint(NE, NE+NI, size = n))
#        neuronsList = np.array([3486])
        print "theta = ", np.squeeze(kTheta)
        tmp = AvgFano(dbName, neuronsList, simDuration, simDT, binsize, kTheta)
        out = np.concatenate((out, tmp[0]), axis = 0)
        fanofactor[kTheta] = tmp[1]

    np.save('fanoFactor_I_' + dbName, fanofactor)
    np.save('fano_mean_var_I_' + dbName, out)

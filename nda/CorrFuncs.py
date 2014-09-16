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

def AutoCorr(x, corrLength = "same"):
    return np.correlate(x, x, mode = corrLength)

def AvgAutoCorr(neuronsList, dbName = "tstDb", theta = 0, minSpks = 50):
    N = len(neuronsList)
    avgCorr = list() 
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    for kNeuron in neuronsList:
        spkTimes = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s", (kNeuron, theta))
        if(spkTimes > minSpks):
            print kNeuron, spkTimes
            st = np.squeeze(np.asarray(dbCursor.fetchall()))
            tmpCorr = AutoCorr(st) 
            tmpCorr = tmpCorr[np.argmax(tmpCorr):] / np.var(st) # right half
            avgCorr.append(tmpCorr)

    dbCursor.close()
    db.close()
    if(len(avgCorr) > 0):
        corLengths = np.asarray(map(len, avgCorr))
        maxLength = np.max(corLengths)
        appendLengths = -1 * (corLengths  - maxLength)
        avgCorMat = np.matrix(np.zeros((len(corLengths), maxLength)))
        for i in np.arange(len(corLengths)):
            tmp0 = np.squeeze(PadZeros(avgCorr[i], appendLengths[i]))
            avgCorMat[i, :]  = tmp0
#            plt.plot(tmp0)
        out = np.squeeze(np.mean(avgCorMat, 0))
        np.save('avgCorr', out)
        # if(np.sum(np.isnan(out))):
        #     print "NAN"
        # print "out shape", out.shape
        # plt.plot(np.squeeze(out), 'k')
        return 
    else :
        return 0

if __name__ == "__main__":
    n = 1000
    N_NEURONS = 10000
    neuronsList = np.random.randint(0, N_NEURONS - 1, size = n)
    avgCorr = AvgAutoCorr(neuronsList)

#    plt.show()
#    plt.waitforbuttonpress()
  #  plt.savefig('avg_autocorr.png')

        

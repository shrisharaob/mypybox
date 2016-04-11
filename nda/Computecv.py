import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig

def ComputeCV(spkTimes):
    cv = 0
    if(len(spkTimes > 2)):
        isi = np.diff(spkTimes)
        if(isi.size > 0):
            mean_isi = np.mean(isi)
            if(mean_isi > 0):
                cv = np.std(isi) / mean_isi
    return cv
        
def CVDistr(dbName, discardTime, neuronsList, theta):
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

p = Pool(20)
useDb = sys.argv[1]
simDuration = 6000
simDT = 0.05
theta = np.array([10, 20, 30, 40, 50, 60, 80, 100])
n = 10000
discardTime = 2500.0
poolList = np.arange(10000, 20000, 1)
#poolList = np.arange(10000)
out = p.map(partial(CVDistr, useDb, discardTime, poolList), theta)
print "DONE !"
prows = 3
pclms = 3
#f, ax = plt.subplots(prows, pclms)
for i, ktau in enumerate(theta):
    results = out[i]
    results = results[results != 0]
    #subscripts = np.unravel_index(i, (prows, pclms))
    cnt, bins, patches = plt.hist(results, 50)
    plt.setp(patches, 'edgecolor', 'k', 'facecolor', 'k')
    plt.title(r'I neurons $\tau = 3ms, \; \alpha = 0.1, \; noise \; \tau = %s$'%(ktau))
    plt.xlabel('CV')
    plt.ylabel('Counts')

    # plt.figure()
    # cnt, bins, patches = plt.hist(results[1], 50)
    # plt.setp(patches, 'edgecolor', 'k', 'facecolor', 'k')
    # plt.title(r'E neurons $\tau = 3ms, \; \alpha = 0$')
    # plt.title('I neurons')
    # plt.xlabel('CV')
    # plt.ylabel('Counts')
    plt.ion()
    plt.show()
#    plt.savefig('CV_alpha_1em1_I_ffnoise_pink_tau_%s'%(ktau))
    ReportFig('CV_ff_pink_noise', '<n(t)n(t)> := exp(-1/%s)'%(ktau), 'CV', 'png', 'CV', 'CV_I_tn%s'%(ktau))
    plt.waitforbuttonpress()

import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
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

nTrials = 50

dbName = 'pois991'
db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
dbCursor = db.cursor()
db.autocommit(True)
fanoFactor = np.empty((1, ))
fanoFactor[:] = np.nan
spkTimeStart = 0.0
spkTimeEnd = 1000.0
ne = 10000
#avgSpkCnt = np.zeros((nTrials, ))
avgSpkCnt = np.zeros((ne, nTrials))

for kTrial in np.arange(nTrials):
    tmpCnt = np.zeros((ne, ))
    for mNeuron in np.arange(ne):
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND spkTimes > %s AND spkTimes < %s ", (mNeuron, kTrial, float(spkTimeStart), float(spkTimeEnd)))
        tmpCnt[mNeuron] = float(nSpks)
        print nSpks
    avgSpkCnt[:, kTrial] = tmpCnt
dbCursor.close()
db.close()

#print "FF=", np.var(avgSpkCnt) / np.mean(avgSpkCnt)
#np.save('delthis', [np.var(avgSpkCnt), np.mean(avgSpkCnt)])

plt.ion()
plt.plot(np.mean(avgSpkCnt, 1), np.var(avgSpkCnt, 1), 'ko')
plt.plot(range(19), range(19), 'r')
kb.keyboard()

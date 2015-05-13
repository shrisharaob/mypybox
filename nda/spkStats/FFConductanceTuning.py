
basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import scipy.stats as stat
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from enum import Enum
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import GetPO


def CircVar(firingRate, atTheta):
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(firingRate)

#datafolder = '/homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/testGmean/'
#datafolder = '/homecentral/srao/orm/twopop/f1r9e1/gffmean/'
datafolder = '/homecentral/srao/orm/twopop/ne4ni1f2r19e1/gffmean/'
NTRIALS = 2
NE = 40000
NI = 10000
theta = np.arange(0., 180., 22.5)
#theta = np.arange(0., 2., 22.5)
print theta
gffTuning = np.empty((NE, theta.size))
gffTuning[:] = 0.0
for kk, kTheta in enumerate(theta):
    for jTrial in range(NTRIALS):
        filename = datafolder + 'gffmean_R00.0_theta%s_0.00_3.0_2000_tr%s.csv'%((int(kTheta), jTrial))
        print filename
        gff = np.loadtxt(filename)
        gff = gff[:NE]
        for nNeuron in range(NE):
            gffTuning[nNeuron, kk] += gff[nNeuron]

gffTuning = gffTuning / NTRIALS 

np.save('gffTuning_testGmean', gffTuning)

cv = np.empty((NE, ))
cv[:] = np.nan
for nNeuron in range(NI):
    cv[nNeuron] = CircVar(gffTuning[nNeuron], theta)

np.save('gffCircVar_testGmean', cv)
plt.ioff()
plt.hist(cv[~np.isnan(cv)], 50, fc = 'k', edgecolor = 'w')

plt.savefig('/homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/gffCircVar.png')


plt.figure()
#po = np.argmax(gffTuning, 1)
po = GetPO.POofPopulation(gffTuning)
po = po.reshape((200, 200))
plt.pcolor(po * 180.0 / np.pi, cmap = 'hsv')
plt.ylim((0, 200))
plt.colorbar()
plt.ion()
plt.show()

kb.keyboard()
#plt.waitforbuttonpress()
#plt.savefig('/homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/testGmean/gff_orimap.png')

basefolder = "/homecentral/srao/Documents/code/mypybox"
import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
#import FanoFactorDynamics as ffd
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import GetPO

#dbnames = ['rewiresig5', 'rewiresig75', 'rewiresig1', 'allAnglesa0T4xi12C100Tr100']
dbnames = ['rewireEIsig5', 'rewireEIsig1', 'allAnglesa0T4xi12C100Tr100']
labels = ['0.5 rad', '1.0 rad', 'random']
NE = 10000

for kk, kdb in enumerate(dbnames):
    tc = np.load('/homecentral/srao/db/data/tuningCurves_'+kdb+'.npy')
    x = GetPO.OSIofPopulation(tc, np.arange(0., 180., 22.5))
    x = x[:NE]
    cnt, bins = np.histogram(x[~np.isnan(x)], 100)
#    plt.plot(bins[:-1], cnt.astype(float)/cnt.sum(), linewidth = 2, label = labels[kk])
    plt.plot(bins[:-1], cnt, linewidth = 2, label = labels[kk])

plt.legend(loc=2)
plt.xlim([0.85, 1])
plt.xlabel('OSI')
plt.ion()
plt.show()
#Print2Pdf(plt.gcf(),  'osi_distr_rewireEI_normalized', [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])
Print2Pdf(plt.gcf(),  'osi_distr_rewireEI', [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])        
kb.keyboard()

basefolder = "/homecentral/srao/Documents/code/mypybox"
#import MySQLdb as mysql
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

def POofLoc(x, y):
# return np.fmod(np.arctan(np.divide(float(y), float(x))) + np.pi * 0.5, np.pi)
 return  np.arctan(np.divide(float(y), float(x))) + (np.pi /2)

ne = 10000
patchSize = int(np.sqrt(ne))
L = 1.0
po = np.empty((ne, ))

for i in range(ne):
    tmp = np.unravel_index(i, (ne, ne))
    x = np.fmod(float(i), patchSize) * (L / (patchSize - 1));
    y = np.floor(float(i) / patchSize) * (L / (patchSize - 1.0))
    x = x - (L * 0.5)
    y = y - (L * 0.5)
    po[i] = POofLoc(x, y)
plt.ion()
plt.imshow(po.reshape((patchSize, patchSize)) * 180.0 / np.pi, cmap = 'hsv')
plt.axis('equal')
plt.xlim(0, patchSize)
plt.ylim(0, patchSize)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.title('PO in layer 4', fontsize = 20)

plt.colorbar()
print np.min(po*180.0/np.pi), np.max(po*180.0 / np.pi)
plt.waitforbuttonpress()
                  

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
# return  0.5 * (np.arctan(np.divide(float(y), float(x))))
    return  0.5 * (np.arctan2(float(y), float(x))) + np.pi * 0.5

ne = 10000
#ne = 16
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
#np.save('om', po.reshape((100, 100)))
plt.axis('equal')
plt.xlim(0, patchSize)
plt.ylim(0, patchSize)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.title('PO in layer 4', fontsize = 20)

plt.colorbar()
print np.min(po*180.0/np.pi), np.max(po*180.0 / np.pi)

kb.keyboard()

# plt.ion()
# plt.figure()
# theta = np.pi
# R0 = 0.09
# for n in range(100):
#     spks = np.zeros((ne, ))
#     steps = 50
#     for mStep in range(steps):
#         for i in range(ne):
#             instRate = R0 + .99 * R0 * np.cos(2.0* (theta - po[i]));
#             spks[i] += (instRate * 0.05) >  np.random.rand()
#     plt.imshow(spks.reshape((100, 100)))
#     plt.waitforbuttonpress()
#     plt.clf()
    
    

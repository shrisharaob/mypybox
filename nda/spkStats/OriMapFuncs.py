basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
import scipy.ndimage as ndimage



def CircVar(atTheta):
    zk = 0.0
    firingRate = np.nan
    if(atTheta.size > 1):
        firingRate = np.ones((atTheta.size, ))
        zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(firingRate)


def test_func(values):
#    print values
    out = np.nan
    poAtCenter = values[4] # 8 neighbours and center element 
    idx = [0, 1, 2, 3, 5, 6, 7, 8]
    values = values[idx]
    neighPOs = values[~np.isnan(values)]
    if(neighPOs.size > 0):
        out = np.abs(np.cos(2.0 * (neighPOs - poAtCenter))).mean()
        #out = CircVar(neighPOs - poAtCenter)
    return out




#x = np.array([[1,2,3],[4,5,6],[7,8,9]])
#x = np.random.rand(9)

dbName = sys.argv[1]
NE = 10000
tuningCurves = np.load(basefolder + '/db/tuningCurves_' + dbName + '.npy')

po = np.argmax(tuningCurves, 1)
poe = po[:NE]
x = poe.reshape((100, 100))
x = x * (np.pi / 8.0)
#x = np.random.rand(10000)
#x = x.reshape((100, 100))
footprint = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])

results = ndimage.generic_filter(x, test_func, footprint=footprint, mode = 'constant', cval=np.nan)
#print results

plt.ion()
plt.imshow(x, cmap = 'hsv')
plt.ylim((0, 100))
#plt.pcolor(x, vmin = 0, vmax = np.max(x[:]), cmap =  'hsv')
plt.colorbar()
plt.title('PO')
plt.figure()
#plt.imshow(results, cmap = 'rainbow', interpolation='gaussian')
plt.pcolor(results, vmin = 0.0, vmax = 1.0, cmap = 'rainbow')
plt.colorbar()
plt.title('orimap local correlation')
plt.show()


kb.keyboard()




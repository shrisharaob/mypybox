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
from Print2Pdf import Print2Pdf
import scipy.ndimage as ndimage
import GetPO

def CircVarNeig(atTheta):
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
        out = np.cos(2.0 * (neighPOs - poAtCenter)).mean()
        #out = CircVarNeig(neighPOs - poAtCenter)
    return out

#x = np.array([[1,2,3],[4,5,6],[7,8,9]])
#x = np.random.rand(9)
dbName = sys.argv[1] #"omR020a0T3Tr15" #
NE = 4
tuningCurves = np.load(basefolder + '/db/data/tuningCurves_' + dbName + '.npy')
po = GetPO.POofPopulation(tuningCurves)
kb.keyboard()
#po = np.argmax(tuningCurves, 1)

neuronType = 'I'
if(neuronType == 'E'):
    poe = po[:NE]
else:
    poe = po[NE:]
x = poe.reshape((100, 100))
#x = x * (np.pi / 8.0)
footprint = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
results = ndimage.generic_filter(x, test_func, footprint=footprint, mode = 'constant', cval=np.nan)
#print results
#plt.rcParams['figure.figsize'] = 4, 3
plt.ioff()
plt.ion()
plt.figure()
#plt.imshow(x, cmap = 'hsv')
print np.min(x), np.max(x)
plt.pcolor(x * 180.0 / np.pi, cmap = 'hsv')
plt.axis('equal')
plt.xlabel('x cordinate')
plt.ylabel('y cordinate')
plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
plt.ylim((0, 100))
kb.keyboard()
#plt.pcolor(x, vmin = 0, vmax = np.max(x[:]), cmap =  'hsv')
#plt.colorbar()
plt.title('PO %s neurons'%(neuronType))
#figFolder = '/homecentral/srao/Documents/cnrs/figures/feb28/'
figFolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/figs/mar19/'
filename = 'OriMapFunc_' + dbName + '_POmap_%s'%(neuronType)  
plt.show()
L = 100.0
radii = np.arange(0, L * 0.5 + 0.001, L * 0.5 / 5.0) # left limits of the set
axHandle = plt.gca()
for kk, kRadius in enumerate(radii[1:]):
    circObj = plt.Circle((50, 50), kRadius, color = 'w', fill = False, linewidth = 2)
    axHandle.add_artist(circObj)
    if(kk > 0 and kk < 5):
        plt.text(50, 50 + (radii[kk] + radii[kk+1])*0.5 - 0.025, '%s'%(kk), color='w', weight = 'bold')
    if(kk == 0):
        plt.text(50, 50, '0', color='w', weight = 'bold')
plt.draw()
plt.waitforbuttonpress()

Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png') #, tickFontsize=14, paperSize = [4.0, 3.0])
plt.figure()
#plt.imshow(results, cmap = 'rainbow', interpolation='gaussian')
#plt.pcolor(results, vmin = 0.0, vmax = 1.0, cmap = 'rainbow')
plt.pcolor(results, vmin = -1.0, vmax = 1.0, cmap = 'rainbow')
plt.colorbar()
plt.title('orimap local correlation')
plt.xlabel('x cordinate')
plt.ylabel('y cordinate')
plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
filename = 'OriMapFunc_' + dbName + '_POmapLocalCorr_%s'%(neuronType)
Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png') #, tickFontsize=14, paperSize = [4.0, 3.0])

#plt.waitforbuttonpress()

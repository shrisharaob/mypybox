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
NE = 40000
NI = 10000
tuningCurves = np.load(basefolder + '/db/data/tuningCurves_' + dbName + '.npy')
po = GetPO.POofPopulation(tuningCurves)
np.save('./data/po_' + dbName, po)
#po = np.argmax(tuningCurves, 1)

neuronType = 'I'
if(neuronType == 'E'):
    poe = po[:NE]
    xDim = int(np.sqrt(NE))
    yDim = xDim
else:
    poe = po[NE:]
    xDim = int(np.sqrt(NI))
    yDim = xDim
x = poe.reshape((xDim, xDim))
#x = x * (np.pi / 8.0)
footprint = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
results = ndimage.generic_filter(x, test_func, footprint=footprint, mode = 'constant', cval=np.nan)
#print results
#plt.rcParams['figure.figsize'] = 4, 3
plt.ioff()
#plt.ion()
plt.figure()
#plt.imshow(x, cmap = 'hsv')
print np.min(x), np.max(x)
plt.pcolor(x * 180.0 / np.pi, cmap = 'hsv')
plt.axis('equal')
plt.xlabel('x cordinate')
plt.ylabel('y cordinate')
plt.ylim((0, yDim))
plt.xlim((0, xDim))
plt.gca().set_xticks(np.arange(0, xDim+1, xDim/5))
plt.gca().set_yticks(np.arange(0, yDim+1, xDim/5))
plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
plt.colorbar()
#kb.keyboard()
#
#plt.pcolor(x, vmin = 0, vmax = np.max(x[:]), cmap =  'hsv')
#plt.colorbar()
plt.title('PO %s neurons'%(neuronType))
#figFolder = '/homecentral/srao/Documents/cnrs/figures/feb28/'
figFolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/figs/'
filename = 'OriMapFunc_' + dbName + '_POmap_%s'%(neuronType)  
#plt.show()
L = float(xDim)
nRings = 3.0
radii = np.arange(0, L * 0.5 + 0.001, L * 0.5 / nRings) # left limits of the set
axHandle = plt.gca()
for kk, kRadius in enumerate(radii[1:]):
    circObj = plt.Circle((L*0.5, L*0.5), kRadius, color = 'w', fill = False, linewidth = 2)
    axHandle.add_artist(circObj)
    if(kk > 0 and kk < 5):
        plt.text(L*0.5, L*0.5 + (radii[kk] + radii[kk+1])*0.5 - 0.025, '%s'%(kk), color='w', weight = 'bold')
    if(kk == 0):
        plt.text(L*0.5, L*0.5, '0', color='w', weight = 'bold')
plt.draw()
#plt.waitforbuttonpress()

#Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png') #, tickFontsize=14, paperSize = [4.0, 3.0])
ftsize = 12.0
Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png',  paperSize = [5.26/ 1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)
plt.figure()
#plt.imshow(results, cmap = 'rainbow', interpolation='gaussian')
#plt.pcolor(results, vmin = 0.0, vmax = 1.0, cmap = 'rainbow')
plt.pcolor(results, vmin = -1.0, vmax = 1.0, cmap = 'rainbow')
plt.colorbar()
plt.title('orimap local correlation')
plt.xlabel('x cordinate')
plt.ylabel('y cordinate')
plt.gca().set_xticks(np.arange(0, xDim+1, xDim/5))
plt.gca().set_yticks(np.arange(0, yDim+1, xDim/5))
plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
filename = 'OriMapFunc_' + dbName + '_POmapLocalCorr_%s'%(neuronType)
Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png',  paperSize = [5.26/ 1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)
plt.clf()
poCnt, poBins = np.histogram(poe * 180.0 / np.pi, 10)
print poCnt.sum()
#kb.keyboard()
plt.bar(poBins[:-1], poCnt, color = 'k', edgecolor='w', width = np.mean(np.diff(poBins)))
plt.xlabel('Preffered orientation')
plt.ylabel('Counts')
plt.title('PO distribution in I neurons')
plt.draw()
filename = 'OriMapFunc_' + dbName + '_POhist_%s'%(neuronType)
Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', paperSize = [6.0, 4.56])
#plt.waitforbuttonpress()

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
from scipy.interpolate import interp1d
from scipy import interpolate
import GetPO

def CircularMean(a):
#     # a is the vector in radians
#     # return value in degrees
    mu = 0.5 * np.angle(np.sum(np.exp(2j * a)), deg = True)
    if(mu < 0):
        mu = mu + 180.0
    return mu

def PlotMaskedMap(omMap, invalidNeurons, n=8500):
    rndmask = np.random.rand(*omMap.shape) > float(omMap.size - n) / float(omMap.size)
    rndmask = np.logical_or(rndmask, invalidNeurons)
    maskedOm = np.ma.array(omMap, mask = rndmask)
    cmap = plt.get_cmap('hsv')
    cmap.set_bad(color = 'w', alpha = 1.)
    print 'printing masked ori map, discarding %s pixels'%(n)
    print 'min max angles:', np.min(omMap), np.max(omMap)
    plt.imshow(maskedOm, cmap = cmap, vmin = 0.0, vmax = 180.0)
    #plt.pcolor(maskedOm, cmap = cmap, vmin = 0.0, vmax = 180.0)
    plt.colorbar()
    filename = 'masked_om_discarded%spix_%s_'%(n, neuronType) + dbName
    ftsize = 12.0
    plt.title('PO %s, %s neurons removed'%(neuronType, n))
    plt.axis('equal')
    plt.xlabel('x cordinate')
    plt.ylabel('y cordinate')
    plt.ylim((0, yDim))
    plt.xlim((0, xDim))
    plt.gca().set_xticks(np.arange(0, xDim+1, xDim/5))
    plt.gca().set_yticks(np.arange(0, yDim+1, xDim/5))
    plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
    plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
    prps = [5.26/ 1.2, 4.26/1.2]  #[3.38, 2.74]
#    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png',  paperSize = prps, labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)
 #   plt.clf()


    
figFolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/figs/'
# [neuronType, NE, NI] = DefaultArgs(sys.argv[1:4], ['I', 40000, 10000])
# nPinwheels = len(sys.argv[4:])
# dbNames = []
[neuronType, NE, NI, nPinwheels] = DefaultArgs(sys.argv[1:], ['I', 40000, 10000, 8])
nPinwheels = int(nPinwheels)
print '#pin wheels = ', nPinwheels
dbNames = ['pinwheel%sTr50'%(x) for x in range(nPinwheels)]
firingThresh = 5.0 # cells with value above are selected
cvThresh = 0.6 # cells with value below are selected
if(neuronType == 'E'):
    nNeurons = NE
else:
    nNeurons = NI
pinWheels = np.zeros((nNeurons, nPinwheels))
avgPinWheel = np.zeros((nNeurons, ))
invalidNeuronsInPool = np.empty((nNeurons, ))
invalidNeuronsInPool[:] = False
print '#invalid neurons in pool ', np.sum(invalidNeuronsInPool)
for i in range(nPinwheels):
    dbName = dbNames[i] #sys.argv[4+i]
#    dbNames.append(dbName)
    print dbName,
    tuningCurves = np.load(basefolder + '/db/data/tuningCurves_' + dbName + '.npy')
    circVar = np.load(basefolder + '/db/data/Selectivity_' + dbName + '.npy')
    try:
        po = np.load('./data/po_' + dbName + '.npy')
    except IOError:
        print 'computing po ...'
        po = GetPO.POofPopulation(tuningCurves)
        np.save('./data/po_' + dbName, po)
    if(neuronType == 'E'):
        poe = po[:NE]
        xDim = int(np.sqrt(NE))
        yDim = xDim
        invalidNeurons = np.logical_or(np.max(tuningCurves[:NE, :], 1) < firingThresh, circVar[:NE] > cvThresh)
    else:
        poe = po[NE:]
        xDim = int(np.sqrt(NI))
        yDim = xDim
        invalidNeurons = np.logical_or(np.max(tuningCurves[NE:, :], 1) < firingThresh, circVar[NE:] > cvThresh)
    invalidNeuronsInPool = np.logical_or(invalidNeuronsInPool, invalidNeurons)
    invalidNeurons = np.reshape(invalidNeurons, (xDim, yDim))
    pinWheels[:, i] = poe 
    print ' #valid neurons ', np.sum(np.logical_not(invalidNeurons))
print ''
invalidNeuronsInPool[:] = False
print '#invalid neurons in pool ', np.sum(invalidNeuronsInPool)
invalidNeuronsInPool = np.reshape(invalidNeuronsInPool, (xDim, yDim))
for j in range(nNeurons):
    avgPinWheel[j] = CircularMean(pinWheels[j, :])
#    if(j > 5000 and j < 5010):
#        print pinWheels[j, :] * 180.0 / np.pi, avgPinWheel[j]
pclmns = int(np.ceil(nPinwheels/3.0))
fg, ax = plt.subplots(3,  int(np.ceil(nPinwheels / 3.0)))
plt.ion()
for i in range(nPinwheels):
    subscripts = np.unravel_index(i, (3, int(np.ceil(nPinwheels / 3.0))))
#    print subscripts
    if(pclmns == 1):
        subscripts = subscripts[0]
    im = ax[subscripts].imshow(np.reshape(pinWheels[:, i] * 180.0 / np.pi, (np.sqrt(nNeurons), np.sqrt(nNeurons))), cmap = 'hsv')
plt.colorbar(im, ax=ax[subscripts])
#im = ax[-1].imshow(np.reshape(avgPinWheel, (np.sqrt(nNeurons), np.sqrt(nNeurons))), cmap = 'hsv')
plt.figure()
immean = plt.imshow(np.reshape(avgPinWheel, (np.sqrt(nNeurons), np.sqrt(nNeurons))), cmap = 'hsv')
#cb = plt.colorbar()
cb = plt.colorbar(immean,fraction=0.03, pad=0.02)
cb.set_ticks(np.array([0, 90, 180]))
plt.title(neuronType)
if neuronType == 'E':
    circ0 = (100., 100.)
    radius0 = 100
    circ1 = (100., 100.)
    radius1 = 50
    plt.text(100, 100, '0', fontsize = 20)
    plt.text(130, 50, '1', fontsize = 20)
    plt.xticks(np.array([0, 100, 200]))
    plt.yticks(np.array([0, 100, 200]))
else:
    circ0 = (50., 50.)
    radius0 = 50
    circ1 = (50., 50.)
    radius1 = 25
    plt.text(50, 50, '0', fontsize = 20)
    plt.text(65, 25, '1', fontsize = 20)
    plt.xticks(np.array([0, 50, 100]))
    plt.yticks(np.array([0, 50, 100]))    
circleObj = plt.Circle(circ0, radius0, color = 'w', fill = False)
plt.gca().add_artist(circleObj)
circleObj = plt.Circle(circ1, radius1, color = 'w', fill = False)
plt.gca().add_artist(circleObj)
plt.show()
figFormat = 'eps'

paperSize = [3.5, 3.]
axPosition = [0.1, 0.15, .6, 0.6]        
Print2Pdf(plt.gcf(), 'mean_pinwheel_N%s_%s'%(nPinwheels, neuronType), paperSize, figFormat=figFormat, labelFontsize = 8, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
for i in range(2):
    plt.figure()
    PlotMaskedMap(np.reshape(avgPinWheel, (np.sqrt(nNeurons), np.sqrt(nNeurons))), invalidNeuronsInPool, 10000)


plt.figure()
plt.hist(avgPinWheel, 8, facecolor = 'k', edgecolor='w')
paperSize = [2., 1.6]
axPosition = [0.3, 0.23, .6, 0.6]
plt.xlabel('PO (deg)')
plt.ylabel('Count')
plt.title('%s'%(neuronType))
plt.xticks(np.array([0, 90, 180]))
if neuronType == 'E':
    plt.yticks(np.array([0.0, 3000, 6000]))
else:
    plt.yticks(np.array([0.0, 800, 1600]))
Print2Pdf(plt.gcf(), 'PO_hist_mean_pinwheel_N%s_%s'%(nPinwheels, neuronType), paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)

kb.keyboard()

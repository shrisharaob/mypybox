basefolder = "/homecentral/srao/Documents/code/mypybox"
import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from enum import Enum
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf

def CircVar(firingRate, atTheta):
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(firingRate)

def NeuronInRing(neuronIdx, radius0, radius1, nNeuronsInPatch, patchSize):
    tmp = np.sqrt(float(nNeuronsInPatch))
    yCor = np.floor(float(neuronIdx)/ tmp) * (1.0 / tmp) - (patchSize * 0.5)
    xCor = np.fmod(float(neuronIdx), tmp) * (1.0 / tmp) - (patchSize * 0.5)
    neuronRadius = np.sqrt(xCor **2 + yCor **2)
    IF_NEURON_WITHIN_RING = False
    if(neuronRadius >= radius0 and neuronRadius < radius1):
        IF_NEURON_WITHIN_RING = True
    return IF_NEURON_WITHIN_RING

dbName = sys.argv[1]  #"omR020a0T3Tr15" 
dataFolder = "/homecentral/srao/Documents/code/mypybox/nda/spkStats/data/"
NE = 4
NI = 10000
patchSize = 1.0
L = 1.0
nRings = 5.0
radii = np.arange(0, L * 0.5 + 0.001, L * 0.5 / nRings) # left limits of the set
#theta = np.arange(0.0, 180.0, 22.5)
theta = np.arange(-90.0, 90.0, 22.5)
ffVsOri = np.load(dataFolder + 'FFvsOri_' + dbName + '.npy')
print ffVsOri.shape
tc = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, )));
circVar = np.load('/homecentral/srao/Documents/code/mypybox/db/data/Selectivity_' + dbName + '.npy')
neuronType = 'I'
if(neuronType == 'E'):
    nNeurons = NE
    ffVsOri = ffVsOri[:NE, :]
    tc = tc[:NE, :]
    circVar = circVar[:NE]
else:
    nNeurons = NI
    ffVsOri = ffVsOri[NE:, :]
    tc = tc[NE:, :]
    circVar = circVar[NE:]

ffVsRadius = np.zeros((radii.size - 1, theta.size))
firingRateVsRadius = np.zeros((radii.size - 1, theta.size))
nNeuronsInRing = np.zeros((radii.size - 1, ))
neuronIds = np.arange(nNeurons)
firingThresh = 0.0
circThresh = 1.0
#validNeuronIdx = np.empty((neuronIds.size, )) #np.max(tc, 1) > firingThresh
#validNeuronIdx[:] = True
validNeuronIdx = np.max(tc, 1) > firingThresh
validNeuronIdx = np.logical_and(validNeuronIdx, circVar < circThresh)
#neuronIds = neuronIds[validNeuronIdx]
print "# valid neurons: ", np.sum(validNeuronIdx)
prefferedOri = np.argmax(tc, 1)
nValidNeuronsInRing = np.zeros((radii.size, ))
#cvInRing = np.zeros((radii.size, 25))
for kk, kRadius in enumerate(radii[:-1]):
    radius0 = radii[kk]
    radius1 = radii[kk + 1]
    tmpFr = np.empty((1, theta.size))
    tmpFF = np.empty((1, theta.size))
    tmpcv = []
    for i, iNeuron in enumerate(neuronIds):
        if(NeuronInRing(iNeuron, radius0, radius1, nNeurons, patchSize)):
            if(validNeuronIdx[iNeuron]):
                iFFvsOri = np.roll(ffVsOri[iNeuron, :], -1 * prefferedOri[iNeuron]) 
                iFiringRateVsOri = np.roll(tc[iNeuron, :], -1 * prefferedOri[iNeuron])
#                tmpcv
#            print iFiringRateVsOri.shape
                iFFvsOri.shape = 1, theta.size
                iFiringRateVsOri.shape = 1, theta.size
                tmpFF = np.concatenate((tmpFF, iFFvsOri), axis = 0)
                tmpFr = np.concatenate((tmpFr, iFiringRateVsOri), axis = 0)
                nValidNeuronsInRing[kk] += 1
            neuronIds = np.setdiff1d(neuronIds, iNeuron)
    tmpFF = tmpFF[1:, :]
    tmpFr = tmpFr[1:, :]
    ffVsRadius[kk, :] = np.nanmean(tmpFF, 0) 
    firingRateVsRadius[kk, :] = np.nanmean(tmpFr, 0) 
print "# neurons in ring: ", nValidNeuronsInRing
print "others =  ", neuronIds.size
np.save('ffinRing', ffVsRadius)
np.save('tcinRing', firingRateVsRadius)


tmpOthers = np.empty((1, theta.size))
for ii, iiNeuron in enumerate(neuronIds):
    mytmp = np.roll(ffVsOri[iiNeuron, :], -1 * prefferedOri[iiNeuron])
    mytmp.shape = 1, theta.size
    tmpOthers = np.concatenate((tmpOthers, mytmp), axis = 0)
tmpOthers = tmpOthers[1:, :]
meanTmpOthers = np.nanmean(tmpOthers, 0)

figFolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/figs/mar19/'
plt.ion()
f0 = plt.figure()
for kk, kRadius in enumerate(radii[:-1]):
    plt.plot(theta, np.roll(ffVsRadius[kk, :], 4), 'o-', label='ring %s (n=%s)'%(kk, nValidNeuronsInRing[kk]))

plt.xlabel('stimulus orientation (deg)')
plt.ylabel('Mean fano factor')
plt.title('Mean Fano factor of %s neurons in ring'%((neuronType)))
plt.legend(loc=0, prop={'size':10})
plt.grid()
plt.draw()
Print2Pdf(f0, figFolder + 'FFInRing_' + neuronType + '_'  + dbName, figFormat='png', paperSize=[6.26, 4.26])

f1 = plt.figure()
for kk, kRadius in enumerate(radii[:-1]):
    plt.plot(theta, np.roll(firingRateVsRadius[kk, :], 4), 'o-', label='ring %s (n=%s)'%(kk, nValidNeuronsInRing[kk]))
plt.xlabel('stimulus orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title('Mean firing rate of %s neurons in ring'%((neuronType)))
plt.legend(loc=0, prop={'size':10})
plt.grid()
plt.draw()
Print2Pdf(f1, figFolder + 'FiringRateInRing_' + neuronType + '_' + dbName , figFormat='png', paperSize=[6.26, 4.26])


plt.figure()
plt.plot(theta, meanTmpOthers, 's-')

kb.keyboard()

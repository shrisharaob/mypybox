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

def CircVar(firingRate):
    atTheta = np.arange(0.0, 180.0, 22.5)
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

def PlotFFvsFr(axisHdl, atThetaIndex):
    frAtPO = np.empty((nNeurons, ))
    frAtPO[:] = np.nan
    ffAtPO = np.empty((nNeurons, ))
    ffAtPO[:] = np.nan
    ffbins = np.linspace(0.0, np.nanmax(ffVsOri), 26)
    for jk in np.arange(nNeurons):
        jkTuningCurve = tc[jk, :]
        jkff = ffVsOri[jk, :]
        jkPO = np.argmax(jkTuningCurve) 
        jkTuningCurve = np.roll(jkTuningCurve, -1 * jkPO)   # shift 2 zero
        jkff = np.roll(jkff, -1 * jkPO)   # shift 2 zero
        frAtPO[jk] = jkTuningCurve[atThetaIndex]
        ffAtPO[jk] = jkff[atThetaIndex]
    #ax_scatter.plot(frAtPO, ffAtPO, '.')
    #ax_scatter.hist(ffAtPO[~np.isnan(ffAtPO)], 26, alpha=0.2)
    fcnt, fbins = np.histogram(ffAtPO[~np.isnan(ffAtPO)], ffbins)
    axisHdl.plot(fbins[:-1], fcnt / float(fcnt.sum()), 'ko-', label = '%s'%(atThetaIndex))
    axisHdl.axvline(np.nanmean(ffAtPO), color = 'k')
    print atThetaIndex, '=', np.nanmean(ffAtPO)
    return fcnt/float(fcnt.sum()), np.nanmean(ffAtPO), np.nanstd(ffAtPO)

dbName = sys.argv[1]  #"omR020a0T3Tr15" 
dataFolder = "/homecentral/srao/Documents/code/mypybox/nda/spkStats/data/"
NE = 40000
NI = 10000
patchSize = 1.0
L = 1.0
nRings = 3.0
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
firingThresh = 10.0
circThresh = 0.5
#validNeuronIdx = np.empty((neuronIds.size, )) #np.max(tc, 1) > firingThresh
#validNeuronIdx[:] = True
validNeuronIdx = np.max(tc, 1) > firingThresh
validNeuronIdx = np.logical_and(validNeuronIdx, circVar < circThresh)
#neuronIds = neuronIds[validNeuronIdx]
print "# valid neurons: ", np.sum(validNeuronIdx)
prefferedOri = np.argmax(tc, 1)
nValidNeuronsInRing = np.zeros((radii.size, ))
nCvBins = 25
cvBins = np.linspace(0.0, 1.0, nCvBins+1); # bin edges
tmpBins = cvBins
tmpBins.shape = nCvBins + 1, 1
cvBinCenters = np.mean(np.concatenate((tmpBins[:-1], tmpBins[1:]), axis = 1), 1)
cvBins.shape = nCvBins + 1, 
cvInRing = np.zeros((radii.size, nCvBins))
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
                tmpcv.append(CircVar(tc[iNeuron, :]))
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
    tmpcv = np.asarray(tmpcv)
    circVarCnt, _ = np.histogram(tmpcv[~np.isnan(tmpcv)], cvBins)
    cvInRing[kk, :] = circVarCnt
    tmpcv = []
print "# neurons in ring: ", nValidNeuronsInRing
print "others =  ", neuronIds.size
np.save('ffinRing', ffVsRadius)
np.save('tcinRing', firingRateVsRadius)

figFolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/figs/'
f00 = plt.figure()
plt.ion()
for kk, kRadius in enumerate(radii[:-1]):
    plt.plot(cvBinCenters, cvInRing[kk, :],'.-', label='ring %s (n=%s)'%(kk, nValidNeuronsInRing[kk]))
plt.xlabel('Circular variance')
plt.ylabel('Count')
plt.title('Circular variance of %s neurons in ring'%((neuronType)))
plt.legend(loc=0, prop={'size':10})
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim((ymin - ymax/50.0, ymax))
labels = [item.get_text() for item in f00.gca().get_xticklabels()]
labels[0] = "0.0\nhighly tuned"
labels[-1] = "1.0\nnot selective"
f00.gca().set_xticklabels(labels)
plt.draw()
ftsize = 10.0
Print2Pdf(f00, figFolder + 'CircVarInRing_' + neuronType + '_'  + dbName, figFormat='png', paperSize=[6.26/1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)
plt.clf()
for kk, kRadius in enumerate(radii[:-1]):
    plt.plot(cvBinCenters, cvInRing[kk, :] / cvInRing[kk, :].sum() ,'.-', label='ring %s (n=%s)'%(kk, nValidNeuronsInRing[kk]))
plt.xlabel('Circular variance')
plt.ylabel('Normalized count')
plt.title('Circular variance of %s neurons in ring'%((neuronType)))
plt.legend(loc=0, prop={'size':10})
plt.grid()
ymin, ymax = plt.ylim()
plt.ylim((ymin - ymax/50.0, ymax))
labels = [item.get_text() for item in f00.gca().get_xticklabels()]
labels[0] = "0.0\nhighly tuned"
labels[-1] = "1.0\nnot selective"
f00.gca().set_xticklabels(labels)
plt.draw()
Print2Pdf(f00, figFolder + 'CircVarInRing_normalized_' + neuronType + '_'  + dbName, figFormat='png', paperSize=[6.26/1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)

tmpOthers = np.empty((1, theta.size))
for ii, iiNeuron in enumerate(neuronIds):
    mytmp = np.roll(ffVsOri[iiNeuron, :], -1 * prefferedOri[iiNeuron])
    mytmp.shape = 1, theta.size
    tmpOthers = np.concatenate((tmpOthers, mytmp), axis = 0)
tmpOthers = tmpOthers[1:, :]
meanTmpOthers = np.nanmean(tmpOthers, 0)


#plt.ion()
f0 = plt.figure()
for kk, kRadius in enumerate(radii[:-1]):
    plt.plot(theta, np.roll(ffVsRadius[kk, :], 4), 'o-', label='ring %s (n=%s)'%(kk, nValidNeuronsInRing[kk]))

plt.xlabel('stimulus orientation (deg)')
plt.ylabel('Mean fano factor')
plt.title('Mean Fano factor of %s neurons in ring'%((neuronType)))
plt.legend(loc=0, prop={'size':10})
plt.grid()
plt.draw()
Print2Pdf(f0, figFolder + 'FFInRing_' + neuronType + '_'  + dbName, figFormat='png', paperSize=[6.26/1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)

f1 = plt.figure()
for kk, kRadius in enumerate(radii[:-1]):
    plt.plot(theta, np.roll(firingRateVsRadius[kk, :], 4), 'o-', label='ring %s (n=%s)'%(kk, nValidNeuronsInRing[kk]))
plt.xlabel('stimulus orientation (deg)')
plt.ylabel('Firing rate (Hz)')
plt.title('Mean firing rate of %s neurons in ring'%((neuronType)))
plt.legend(loc=0, prop={'size':10})
plt.grid()
plt.draw()
Print2Pdf(f1, figFolder + 'FiringRateInRing_' + neuronType + '_' + dbName , figFormat='png', paperSize=[6.26/1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)


plt.figure()
plt.plot(theta, meanTmpOthers, 's-')

figHdl_ffdistr, axHdl_ffdistr = plt.subplots(8)
tmpffm = np.zeros((8, 25))
meanFFAtPO = np.zeros((8, ))
FFstdAtPO = np.zeros((8, ))
for ii, lk in enumerate(np.roll(range(8), 4)):
    tmp00 = PlotFFvsFr(axHdl_ffdistr[ii], lk)
    tmpffm[ii, :] = tmp00[0]
    meanFFAtPO[ii] = tmp00[1]
    FFstdAtPO[ii] = tmp00[2]
#-------------------------------------------------------
# MAKE XLIMS SAME ACROSS SUBPLOTS
tmpxlimmax = np.zeros((8, ))
for lk in range(8):
    tmpxlimmax[lk] = axHdl_ffdistr[lk].get_xlim()[1]
for lk in range(8):
    axHdl_ffdistr[lk].set_xlim((0, 2.0)) #np.max(tmpxlimmax)))
    axHdl_ffdistr[lk].set_ylabel(theta[lk])
    zed = [tick.label.set_fontsize(8) for tick in axHdl_ffdistr[lk].yaxis.get_major_ticks()]
#---------------------------------------------------------
# KEEP ONLY THE XTICKS IN THE LAST SUBPLOT FOR ASTHETIC REASONS
for lk in range(7):
    labels = [item.get_text() for item in axHdl_ffdistr[lk].get_xticklabels()]
    labels[0] = ''
    axHdl_ffdistr[lk].set_xticklabels(labels)
    # ylabels = [item.get_text() for item in axHdl_ffdistr[lk].get_yticklabels()]
    # tmplabels = np.array(ylabels)
    # tmplabels[1:-2] = ''
    # axHdl_ffdistr[lk].set_yticklabels(tmplabels)
    
axHdl_ffdistr[4].set_ylabel('PO')
#------------------------------------------------------------
axHdl_ffdistr[0].set_title('Normalized Fano factor distr at stimulus angles, %s neurons'%(neuronType))
axHdl_ffdistr[-1].set_xlabel('Fano Factor')
plt.draw()    
plt.ion()
plt.show()
#plt.plot(range(8), tmpffm, 'ko-')
ffbins = np.linspace(0.0, np.nanmax(ffVsOri), 26)
# FFdistr_vs_ori_
#plt.imshow(tmpffm)

Print2Pdf(figHdl_ffdistr, figFolder + 'FFDistrInRing_' + neuronType + '_' + dbName , figFormat='png', paperSize=[6.26, 12.3], labelFontsize = 10, tickFontsize = 8, titleSize = 12)


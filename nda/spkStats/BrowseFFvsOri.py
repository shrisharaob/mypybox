
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
from Print2Pdf import Print2Pdf
import GetPO

def PlotFFGivenCordinates(x, y, ax_tc, ax_ff, nNeuronsInPatch, neuronType = 'E'):
#    fg_tc, ax_tc = plt.subplots()
 #   fg_ff, ax_ff = plt.subplots()
    ax_tc.clear()
    ax_ff.clear()
    xAxis = np.arange(-90, 90.0, 22.5)
    patchSize = int(np.sqrt(nNeuronsInPatch))
    print patchSize
    if(neuronType == 'E'):
        neuronId = np.ravel_multi_index((x, y), (patchSize, patchSize))
    else :
        print x, y, patchSize
        neuronId = np.ravel_multi_index((x, y), (patchSize, patchSize))
        neuronId = neuronId + NE # since I indexing starts from fron NE 
    xyTuningCurve = tuningCurves[neuronId, :]
    xyFanoFactor =  fanoFactor[neuronId, :]
    xyPO = np.argmax(xyTuningCurve) 
    xyTuningCurve = np.roll(xyTuningCurve, -1 * xyPO)   # shift 2 zero
    xyFanoFactor = np.roll(xyFanoFactor, -1 * xyPO)
    ax_tc.plot(xAxis, np.roll(xyTuningCurve, 4), 'ko-') # center
    ax_ff.plot(xAxis, np.roll(xyFanoFactor, 4), 'ko-')
    ax_tc.set_title('Tuning Curve, neuron id: %s'%(neuronId))    
    ax_tc.set_xlabel('Stimulus orientation')
    ax_tc.set_ylabel('Firing rate (Hz)')
    ax_ff.set_title('Fano factor, neuron id: %s'%(neuronId))    
    ax_ff.set_xlabel('Stimulus orientation')
    ax_ff.set_ylabel('Fano factor')
    ax_ff.set_xlim((xAxis[0], xAxis[-1]))
    ax_tc.set_xlim((xAxis[0], xAxis[-1]))
    plt.draw()
    plt.show()
    return 0

def PlotFFvsFr(ax_scatter, atIndex):
    frAtPO = np.empty((neuronsOnPatch, ))
    frAtPO[:] = np.nan
    ffAtPO = np.empty((neuronsOnPatch, ))
    ffAtPO[:] = np.nan
    for jk in np.arange(neuronsOnPatch):
        if(neuronType == 'E'):
            jkTuningCurve = tuningCurves[jk, :]
            jkff = fanoFactor[jk, :]
            jkPO = np.argmax(jkTuningCurve) 
            jkTuningCurve = np.roll(jkTuningCurve, -1 * jkPO)   # shift 2 zero
            jkff = np.roll(jkff, -1 * jkPO)   # shift 2 zero
            frAtPO[jk] = jkTuningCurve[atIndex]
            ffAtPO[jk] = jkff[atIndex]
        else:
            jkTuningCurve = tuningCurves[jk+NE, :]
            jkff = fanoFactor[jk+NE, :]
            jkPO = np.argmax(jkTuningCurve) 
            jkTuningCurve = np.roll(jkTuningCurve, -1 * jkPO)   # shift 2 zero
            jkff = np.roll(jkff, -1 * jkPO)   # shift 2 zero
            frAtPO[jk] = jkTuningCurve[atIndex]
            ffAtPO[jk] = jkff[atIndex]
    #ax_scatter.plot(frAtPO, ffAtPO, '.')
    #ax_scatter.hist(ffAtPO[~np.isnan(ffAtPO)], 26, alpha=0.2)
    fcnt, fbins = np.histogram(ffAtPO[~np.isnan(ffAtPO)], 26)
    ax_scatter.plot(fbins[:-1], fcnt / float(fcnt.sum()), label = '%s'%(atIndex))
    print atIndex, '=', np.nanmean(ffAtPO)
    return np.nanmean(ffAtPO)
#    ax_scatter_xlim = ax_scatter.get_xlim()
#    ax_scatter_ylim = ax_scatter.get_ylim()
#    ax_scatter.plot(ax_scatter_ylim, ax_scatter_ylim, 'r')

# def onclick(event):
#     x = int(event.xdata)
#     y = int(event.ydata)
#     neuronId = np.ravel_multi_index((x, y), (100, 100))
#     PlotFFGivenCordinates(x, y, ax_tc, ax_ff, NE)

def keyp(event):
    global figCounter
    if event.key == "f":
        figname = dbName + '%s'%(figCounter)
        figCounter += 1
        print "saving figure as", figname, ' in', figFolder
        Print2Pdf(plt.gcf(), figFolder + figname, figFormat='png', tickFontsize=10)        

if __name__ == "__main__":
    dbName = sys.argv[1]
    thetaStep = 22.5
    thetas = np.arange(0, 180., thetaStep)
    NE = 40000
    NI = 10000
    neuronType = 'E'

    figCounter = -1
    figFolder = basefolder + '/nda/spkStats/figs_browse_orimap/' + dbName + '/'
    if not os.path.isdir(figFolder):
        os.makedirs(figFolder)
    tuningCurves = np.load(basefolder + '/db/data/tuningCurves_' + dbName + '.npy')
    fanoFactor = np.load(basefolder + '/nda/spkStats/data/FFvsOri' + '_' + dbName + '.npy')
    preferredOri = np.argmax(tuningCurves, 0) 
    firingThresh = 2.0
#    invalidId = np.max(tuningCurves, 1) < firingThresh
#    po = np.argmax(tuningCurves, 1)
    po = GetPO.POofPopulation(tuningCurves)
    if(neuronType == 'E'):
        poe = po[:NE]
        neuronsOnPatch = NE
        xDim = int(np.sqrt(NE))
        yDim = xDim
    else :
        poe = po[NE:]
        neuronsOnPatch = NI
        xDim = int(np.sqrt(NI))
        yDim = xDim
 #   poe[invalidId] = np.nan
    print poe.shape, tuningCurves.shape
    fg = plt.figure(0)
    ax_po = plt.subplot2grid((2,2), (0, 0), aspect = 'equal') #, rowspan=2)
    ax_po.set_xlim((0, xDim))
    ax_po.set_ylim((0, yDim))
    ax_po.set_title('PO in layer 2/3 %s neurons'%(neuronType))
    ax_po.set_xlabel('x')
    ax_po.set_ylabel('y')
    ax_po.set_position([0.125, 0.58, 0.3, 0.3])
    ax_tc = plt.subplot2grid((2,2), (1, 0))
    ax_ff = plt.subplot2grid((2,2), (1, 1))
    ax_scatter = plt.subplot2grid((2,2), (0, 1))
    plt.draw()
    print ax_tc.get_position()
    print ax_ff.get_position()
    ax_tc.set_position([0.125, 0.1, 0.363636, 0.363636])
    ax_ff.set_position([0.57, 0.1, 0.363636, 0.363636])
    plt.ion()
    im = ax_po.imshow(poe.reshape((xDim, yDim)) * 22.5, cmap='hsv')
    tmpfm = np.zeros((8, ))
    for lk in range(8):
        tmpfm[lk] = PlotFFvsFr(ax_scatter, lk)
    plt.figure()
    plt.plot(tmpfm, 'ko-')
    kb.keyboard()
    ax_scatter.legend()
    plt.draw()
    plt.show()
    plt.colorbar(im, ax=ax_po)
    fg.canvas.mpl_connect('key_press_event', keyp)
    while(plt.fignum_exists(0)):
        plt.figure(0)
        xyCordinates = plt.ginput(1)
        neuronId = np.ravel_multi_index((int(xyCordinates[0][0]), int(xyCordinates[0][1])), (xDim, yDim))
        if(np.max(tuningCurves[neuronId, :]) >= firingThresh):
            #circleObj = plt.Circle((int(xyCordinates[0][0]), int(xyCordinates[0][1])), 1.0, color = 'w', fill = False)
            #ax_po.add_artist(circleObj)
            ax_tc.clear()
            ax_ff.clear()
            if(len(ax_po.lines)):
                ax_po.lines.pop(0)
#                ax_po.show()
            ax_po.plot(int(xyCordinates[0][0]), int(xyCordinates[0][1]), 'wo', markersize=10)
            out = PlotFFGivenCordinates(int(xyCordinates[0][0]), int(xyCordinates[0][1]), ax_tc, ax_ff, neuronsOnPatch, neuronType)

        else:
            print "peak firing rate is below threshold of ", firingThresh




# masked_array = np.ma.array (a, mask=np.isnan(a))
# cmap = matplotlib.cm.jet
# cmap.set_bad('w',1.)
# ax.imshow(masked_array, interpolation='nearest', cmap=cmap)

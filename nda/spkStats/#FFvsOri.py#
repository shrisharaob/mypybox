#script to compute fano factor as a function of bi-directional connectivity alpha

#basefolder  = "/home/shrisha/Documents/code/mypybox"

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

def CircVar(fanofactor, atTheta):
    zk = np.dot(fanofactor, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(fanofactor)

def ComputeCV(spkTimes):
    cv = np.nan
    if(spkTimes.size > 2):
        isi = np.diff(spkTimes)
        if(isi.size > 0):
            mean_isi = np.mean(isi)
            if(mean_isi > 0):
                cv = np.std(isi) / mean_isi
    return cv

def CVInIntervalForAllTheta(dbName, spkTimeStart, spkTimeEnd, nTrials, neuronId):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    thetas = np.arange(0., 180., 22.5)
    cvVsTheta = np.empty((thetas.size, ))
    thetas = thetas.astype(int)
    for mm, mTheta in enumerate(thetas):
        cv = np.empty((nTrials, 1))
        cv[:] = np.nan
        for kTrial in np.arange(nTrials):
            kTheta = int((kTrial+1) * 1000) + mTheta
            nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND spkTimes > %s AND spkTimes < %s ", (neuronId, kTheta, float(spkTimeStart), float(spkTimeEnd)))
            if(nSpks > 0):
                spkTimes = np.squeeze(np.asarray(dbCursor.fetchall()))
                cv[kTrial] = ComputeCV(spkTimes)
        cvVsTheta[mm] = np.nanmean(cv) 
    dbCursor.close()
    db.close()
    return  cvVsTheta

def FanoInIntervalForAllTheta(dbName, spkTimeStart, spkTimeEnd, nTrials, neuronId):
    avgSpkCnt = np.zeros((nTrials, ))
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    fanoFactor = np.empty((1, ))
    fanoFactor[:] = np.nan
#    print "neuronId", neuronId
    thetas = np.arange(0., 180., 22.5)
    ffvsTheta = np.empty((thetas.size, ))
    thetas = thetas.astype(int)
    for mm, mTheta in enumerate(thetas):
        for kTrial in np.arange(nTrials):
            kTheta = int((kTrial+1) * 1000) + mTheta
            nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s AND theta = %s AND spkTimes > %s AND spkTimes < %s ", (neuronId, kTheta, float(spkTimeStart), float(spkTimeEnd)))
            avgSpkCnt[kTrial] = float(nSpks)
        ffvsTheta[mm] = np.var(avgSpkCnt) / np.mean(avgSpkCnt) 
    dbCursor.close()
    db.close()
    return  ffvsTheta

if __name__ == "__main__":
    alpha = np.array([0.0])
#    NE = 10000
    NE = 4
    NI = 40000
    simDuration = 2000
    spkTimeStart = 1000.0
    spkTimeEnd = 2000.0
    simDT = 0.05
    tau = 3.0
    binSize = 50.0  # in ms fano factor observation window
    print alpha
    [dbName, computeType, nTrials, alpha, simDuration, simDT, NE, NI, tau] = DefaultArgs(sys.argv[1:], ['', 'plot', 3, alpha, simDuration, simDT, NE, NI, tau])
    try:
        nTrials = int(nTrials)
    except ValueError:
        print 'ntrials not an interetr !'
        raise
    thetas = np.arange(0., 180., 22.5)
    filebase = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/data/'
    if(computeType == 'compute'):
        neuronsList = np.arange(NE + NI)
        ffMat = np.empty((NE+NI, len(thetas))) # matrix N_NEURONS-by-theta with each element containing the fano factor 
        ffMat[:] = np.nan
        func = partial(FanoInIntervalForAllTheta, dbName, spkTimeStart, spkTimeEnd, nTrials)
        p = Pool(20)
        result = p.map(func, neuronsList) 
        p.close()
        ffMat = np.asarray(result)
        filename =  filebase + 'FFvsOri' #os.path.splitext(sys.argv[0])[0]
#        print 'asdfasddfasdf =', os.path.splitext(sys.argv[0])[0] 
        np.save(filename + '_' + dbName, ffMat)
        
    elif(computeType == 'cv'):
        IF_PLOT = True
        if(not IF_PLOT):
            neuronsList = np.arange(NE + NI)
            cvMat = np.empty((NE+NI, len(thetas))) # matrix N_NEURONS-by-theta with each element containing the fano factor 
            cvMat[:] = np.nan
            func = partial(CVInIntervalForAllTheta, dbName, spkTimeStart, spkTimeEnd, nTrials)
            print len(neuronsList)
            p = Pool(16)
            result = p.map(func, neuronsList) 
            p.close()
            cvMat = np.asarray(result)
            filename = filebase + os.path.splitext(sys.argv[0])[0]
            np.save(filename + '_cv_' + dbName, cvMat)
#            kb.keyboard()
        if(IF_PLOT):
            print "plotting cv"
            tc = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, ))); 
            filename = filebase + os.path.splitext(sys.argv[0])[0]
            cv = np.load(filename + '_' + dbName + '.npy')
            prefferedOri = np.argmax(tc, 1)
            cvMat = np.empty((NE+NI, len(thetas)))
            for kNeuron in np.arange(NE + NI):
                cvMat[kNeuron, :] = np.roll(cv[kNeuron, :], -1 * prefferedOri[kNeuron])
            plt.ion()
            circVar = np.load('/homecentral/srao/Documents/code/mypybox/db/data/Selectivity_'+ dbName +'.npy') #allAnglesa0T4xi12C100Tr100.npy')
            nid = np.arange(NE + NI)
            circVarThresh = 0.5
            firingRateThresh = 10.0
            plotId = np.max(tc, 1) > firingRateThresh
            plotId = np.logical_and(circVar < circVarThresh, plotId)
            tmpId = np.arange(NE+NI)
            plotId = tmpId[plotId]
            print "# valid neurons E: ", np.sum(plotId < NE), ", I: ", np.sum(plotId > NE)
            tmpE = cvMat[plotId[plotId < NE], :]
            tmpI = cvMat[plotId[plotId > NE], :]
            meanE = np.nanmean(tmpE * tmpE, 0)
            meanI = np.nanmean(tmpI * tmpI, 0)
            meanE = np.roll(meanE, 4)
            meanI = np.roll(meanI, 4)
            thetas = np.arange(-90, 90, 22.5)
            plt.plot(thetas, meanE, 'ko-', label='E (N = %s)'%(np.sum(plotId < NE)))
            plt.plot(thetas, meanI, 'ro-', label='I (N = %s)'%(np.sum(plotId > NE)))
            plt.xlabel(r'Stimulus orientation ($\deg$)', fontsize = 20)
            plt.ylabel(r'Mean $CV^2$', fontsize = 20)
      #      plt.title(r'$\alpha = 0.0,\; \tau = 3.0,\; \xi = 1.2,\; fr_{thresh} = %sHz, \; CircVar_{thersh} = %s$'%(firingRateThresh, circVarThresh), fontsize = 16)
            plt.title(r'$\alpha = 0.0,\; \tau = 3.0,\; fr_{thresh} = %sHz, \; CircVar_{thersh} = %s$'%(firingRateThresh, circVarThresh), fontsize = 16)
            plt.grid()
            plt.legend(loc = 0)
            plt.ion()
            plt.show()
            filename = 'cv_vs_ori_frft_%s_cvlt_%s.png'%(firingRateThresh, circVarThresh)
            print "saving figures as", filename
            plt.savefig('./figs/' + filename, format='png')
    elif(computeType == 'ff_circvar'):
        filename = os.path.splitext(sys.argv[0])[0]
        filename = filebase + filename + '_' + dbName + '.npy'
        ff =  np.load(filename) 
        circVariance = np.zeros((NE + NI, ))
        theta = np.arange(0.0, 180.0, 22.5)
        circVarThresh = 0.4
        maxFiringRateThresh = 2.0
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, )))
        ccv = np.load('/homecentral/srao/Documents/code/mypybox/db/data/Selectivity_' + dbName + '.npy')
        maxRates = np.max(tc, 1)
        nid = np.arange(NE + NI) # id vector 
        useNeuronId = nid[np.logical_and(ccv < circVarThresh, maxRates> maxFiringRateThresh)]
        print "valid Neuons NE = ", len(useNeuronId < NE), " NI = ", len(useNeuronId > NE)
        for i, kNeuron in enumerate(nid):
            circVariance[i] = CircVar(ff[kNeuron, :], theta)
        filename = filebase + filename + computeType
        print " saving as ", filename
        np.save(filename, circVariance)
        fcve = circVariance[useNeuronId[useNeuronId < NE]]
        fcvi = circVariance[useNeuronId[useNeuronId > NE]]
        nBins = 100
        cntE, binsE = np.histogram(fcve[~np.isnan(fcve)], nBins)
        cntI, binsI = np.histogram(fcvi[~np.isnan(fcvi)], nBins)
        plt.bar(binsI[:-1], cntI, color = 'r', edgecolor = 'r', width = np.mean(np.diff(binsI)))
        plt.bar(binsE[:-1], cntE, color = 'k', edgecolor = 'k', width = np.mean(np.diff(binsE)))
        plt.legend(('I', 'E'), loc=0)
        plt.xlabel('Circular variance of fano factor', fontsize = 20)
        plt.ylabel('Counts')
        plt.title(r'Distribution of circular variance of Fano factor, $\alpha = %s$'%(alpha))
        plt.ion()
        plt.show()
        kb.keyboard()
    elif(computeType == 'ff_po_scatter'):
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_bidirII_%s.npy'%((dbName, )))
        filename = os.path.splitext(sys.argv[0])[0]
        filename = filebase + filename + '_' + dbName + '.npy'
        ff =  np.load(filename) 
#        ff = np.load(filename + '_' + dbName + '.npy')
        prefferedOri = np.argmax(tc, 1)
        ffMat = np.empty((NE+NI, len(thetas)))
        for kNeuron in np.arange(NE + NI):
            ffMat[kNeuron, :] = np.roll(ff[kNeuron, :], -1 * prefferedOri[kNeuron])
        plt.plot(ffMat[NE:,0], ffMat[NE:, 4], '.r')
        plt.plot(ffMat[:NE,0], ffMat[:NE, 4], '.k')
        plt.ion()
        kb.keyboard()

    else:
        print "plotting ", "here"
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, ))); 
        print tc.shape
        #tc = np.load('/home/shrisha/Documents/cnrs/tmp/jan30/tuningCurves_allAnglesa0T4xi12C100Tr100.npy')
        filename = 'FFvsOri' #os.path.splitext(sys.argv[0])[0]
        ff = np.load(filebase + filename + '_' + dbName + '.npy')
        print tc.shape, ff.shape
        #ff = np.load('/home/shrisha/Documents/cnrs/tmp/jan30/FFvsOri_allAnglesa0T4xi12C100Tr100.npy')
        prefferedOri = np.argmax(tc, 1)
        ffMat = np.empty((NE+NI, len(thetas)))
        tcMat = np.empty((NE + NI, len(thetas)))
        for kNeuron in np.arange(NE + NI):
            ffMat[kNeuron, :] = np.roll(ff[kNeuron, :], -1 * prefferedOri[kNeuron])
            tcMat[kNeuron, :] = np.roll(tc[kNeuron, :], -1 * prefferedOri[kNeuron])
        plt.ioff()
        circVar = np.load('/homecentral/srao/Documents/code/mypybox/db/data/Selectivity_' + dbName + '.npy')
        print "CIRC VAR", circVar.shape
        nid = np.arange(NE + NI)
        circVarThresh = 0.5
        firingRateThresh = 5.0
        plotId = np.max(tc, 1) > firingRateThresh
        plotId = np.logical_and(circVar < circVarThresh, plotId)
        tmpId = np.arange(NE+NI)
        plotId = tmpId[plotId]
        print "# valid neurons E: ", np.sum(plotId < NE), ", I: ", np.sum(plotId > NE)
        meanE = np.nanmean(ffMat[plotId[plotId < NE], :], 0)
        meanI = np.nanmean(ffMat[plotId[plotId > NE], :], 0)
        meanFrE = np.mean(tcMat[plotId[plotId < NE], :], 0)
        meanFrI = np.mean(tcMat[plotId[plotId > NE], :], 0)
#        kb.keyboard()
        meanE = np.roll(meanE, 4)
        meanI = np.roll(meanI, 4)
        meanFrE = np.roll(meanFrE, 4)
        meanFrI = np.roll(meanFrI, 4)
        thetas = np.arange(-90, 90, 22.5)

 #       plt.plot(thetas, meanE, 'ko-', label='E (N = %s)'%(np.sum(plotId < NE)))

        plt.plot(thetas, meanI, 'ro-', label='I (N = %s)'%(np.sum(plotId > NE)))
        plt.xlabel(r'Stimulus orientation ($\deg$)', fontsize = 20)
        plt.ylabel('Mean fano factor', fontsize = 20)
  #      plt.title(r'$\alpha = 0.0,\; \tau = 3.0,\; \xi = 1.2,\; fr_{thresh} = %sHz, \; CircVar_{thersh} = %s$'%(firingRateThresh, circVarThresh), fontsize = 16)
        plt.title(r'$\alpha = 0.0,\; \tau = 3.0,\; fr_{thresh} = %sHz, \; CircVar_{thersh} = %s$'%(firingRateThresh, circVarThresh), fontsize = 16)
        plt.grid()
        plt.legend(loc=0, prop={'size':10})
        filename = 'ff_vs_ori_frft_%s_cvlt_%s_'%(firingRateThresh, circVarThresh) + dbName + '.png'
        print "saving figures as", filename
        #figFolder = '/homecentral/srao/Documents/cnrs/figures/feb28/'
        figFolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/figs/'
        Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png') #, tickFontsize=12, paperSize = [4.0, 3.0])
#        plt.ion(); plt.show(); plt.waitforbuttonpress()
#        plt.savefig(filename, format='png')
        plt.clf()
 
#        plt.plot(thetas, meanFrE, 'ko-', label='E (N = %s)'%(np.sum(plotId < NE)))

        plt.plot(thetas, meanFrI, 'ro-', label='I (N = %s)'%(np.sum(plotId > NE)))
        plt.grid()
        

        plt.xlabel('Stimulus orientation (deg)', fontsize = 20)
        plt.ylabel('Mean firing rate (Hz)', fontsize = 20)
        #plt.title(r'$\alpha = 0.0,\; \tau = 3.0,\; \xi = 1.2,\; fr_{thresh} = %sHz, \; CircVar_{thersh} = %s$'%(firingRateThresh, circVarThresh), fontsize = 16)      
        plt.title(r'$\alpha = 0.0,\; \tau = 3.0,\; \; fr_{thresh} = %sHz, \; CircVar_{thersh} = %s$'%(firingRateThresh, circVarThresh), fontsize = 16)      
        filename = 'tuning_curves_frft_%s_cvlt_%s_'%(firingRateThresh, circVarThresh) + dbName + '.png'
        print "saving mean tuningcurve as", filename
        #plt.savefig(filename)
        plt.legend(loc=0, prop={'size':10})

#    figFolder = '/homecentral/srao/Documents/code/mypybox/db/'
        Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png') #, tickFontsize=14, paperSize = [4.0, 3.0])
 #       plt.ion(); plt.show()
        
        # plt.figure()
        # plt.plot(np.transpose(tcMat), color = [0.9, 0.9, 0.9])
        # plt.ion()
        # plt.show()
#        plt.waitforbuttonpress()

#        plt.waitforbuttonpress()
 #       kb.keyboard()

        #f, ax = plt.subplots(2, 4)
        #f.set_size_inches(26.5,10.5)
        # print ax.shape
        # for i in np.arange(len(thetas)):
        #     subscripts = np.unravel_index(i, (2, 4))
        #     tmp = ffMat[NE:, i]
        #     tmp = tmp[tmp != 0]
        #     cnts, bins = np.histogram(tmp, 50)
        #     #cnts=cnts.astype(float)
        #     ax[subscripts].bar(bins[:-1], cnts, color = 'r', edgecolor = 'r', width = bins[1]-bins[0])
        #     tmp = ffMat[:NE, i]
        #     tmp = tmp[tmp != 0]
        #     cnts, bins = np.histogram(tmp, 50)
        #     #cnts=cnts.astype(float)
        #     ax[subscripts].bar(bins[:-1], cnts, color = 'k', edgecolor = 'k', width = bins[1]-bins[0])
        #     ax[subscripts].set_title(r'$\theta = %s$'%(thetas[i]))

#        ReportFig('FFvsOri_%s'%(dbName), 'alpha = %s, tau = 3ms, T = 1sec <br> average over 16 trials'%(alpha[0]), 'Fano factor vs Orientation', 'png', '', 'summary_alpha%s'%(alpha[0]))
 #           tmp[:, i] = cnts / np.sum(cnts)

        

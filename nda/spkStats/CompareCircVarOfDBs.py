basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
import MySQLdb as mysql
sys.path.append(basefolder)
import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import GetPO

# def CircVar(firingRate, atTheta):
#     zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
#     return 1 - np.absolute(zk) / np.sum(firingRate)

rho = '1'
xi = '_xi0.8'

print 'COMPUTING FOR RHO  = ', rho

def CircVar(firingRate, atTheta):
    out = np.nan
    zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    if(firingRate.mean() > 5.0):
        out = 1 - np.absolute(zk) / np.sum(firingRate)
    return out


def PopulationCircVar(tuningCurves, theta, NE, NI):
    circVariance = np.zeros((NE + NI,))
    neuronIdx = np.arange(NE + NI)
    for i, kNeuron in enumerate(neuronIdx):
        circVariance[i] = CircVar(tuningCurves[kNeuron, :], theta)
    return circVariance

def PlotCircVars(circVariance, NE, NI, axE, axI, legendLabel, nBins = 26):
    cvE = circVariance[:NE]
    cvI = circVariance[NE:]
    cvE = cvE[np.logical_not(np.isnan(cvE))]
    cvI = cvI[np.logical_not(np.isnan(cvI))]
    cveCnt, cvEbins = np.histogram(cvE, nBins)
    cviCnt, cvIbins = np.histogram(cvI, nBins)
#    axE.plot(cvEbins[:-1], cveCnt / float(cveCnt.sum()), '.-', label = legendLabel + r'($\mu$: %.4s)'%(cvE.mean()))
 #   axI.plot(cvIbins[:-1], cviCnt / float(cviCnt.sum()), '.-', label = legendLabel + r'($\mu$: %.4s)'%(cvI.mean()))
    axE.plot(cvEbins[:-1], cveCnt / float(cveCnt.sum()), '.-', label = legendLabel)
    axI.plot(cvIbins[:-1], cviCnt / float(cviCnt.sum()), '.-', label = legendLabel)
    print 'mean circ var: ', cvE.mean(), cvI.mean()
    return np.array([cvE.mean(), cvI.mean()])

def PlotPopAvgTuningCurves(tuningCurves, nTheta, NE, NI, axE, axI, legendLabel):
    prefferedOri = np.argmax(tuningCurves, 1)
    tcMat = np.empty((NE + NI, nTheta))
    for kNeuron in np.arange(NE + NI):
        tcMat[kNeuron, :] = np.roll(tuningCurves[kNeuron, :], -1 * prefferedOri[kNeuron])
    meanE = np.nanmean(tcMat[:NE, :], 0)
    meanI = np.nanmean(tcMat[NE:, :], 0)
    print  meanE.mean(), meanI.mean()
    rotateMeanBy = 4
    meanE = np.roll(meanE, rotateMeanBy)
    meanI = np.roll(meanI, rotateMeanBy)
    #    theta = np.linspace(-90, 90, nTheta)
    theta = np.arange(-90, 90, 22.5)    
    axE.plot(theta, meanE, 'o-', label = legendLabel) #'p: %s'%(legendLabel))
    axI.plot(theta, meanI, 'o-', label = legendLabel) #'p: %s'%(legendLabel))
    return meanE.mean(), meanI.mean()
    
def main2(dbnames, legendLabels, xtickLabels, nTheta = 8):
    # PLOTS MEAN TUNING CURVES
    plt.ioff()
    fgtcE, axtcE = plt.subplots()
    fgtcI, axtcI = plt.subplots()
    fgfr, axfr = plt.subplots()
    colormap = plt.cm.gist_rainbow
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(dbNames))]
    axtcE.set_color_cycle(colors)
    axtcI.set_color_cycle(colors)
    fr = np.zeros((2, len(dbNames)))
    print len(dbNames), len(legendLabels)
    for k, kDb in enumerate(dbNames):
        print "Processint db:", kDb
        print "loading file: tuningCurves_" + kDb + 'rho' + rho + xi + '.npy'
        tc = np.load(basefolder + '/db/data/tuningCurves_' + kDb + 'rho' + rho + xi + '.npy')
        theta = np.arange(thetaStart, thetaEnd, thetaStep)
        if(kDb == 'a1b25te6T10' or kDb == 'a1b1T10te6rho1'):
#            legendLabel = 1
            ne = 20000
            ni = 20000
        else:
 #           legendLabel = int(kDb[6:])
            ne = NE
            ni = NI
        print ne, ni
        fr[:, k] = PlotPopAvgTuningCurves(tc, nTheta, ne, ni, axtcE, axtcI, legendLabels[k])
    xaxis = np.arange(1, len(dbNames) + 1, 1)
    axfr.plot(xaxis, fr[0, :], 'o-k', label = 'E')
    axfr.plot(xaxis, fr[1, :], 'o-r', label = 'I')
    axfr.grid('on')
    axtcE.set_title('Mean Population tuning curves, E')
    axtcE.set_xlabel('Orientation (deg)')
    axtcE.set_ylabel('Firing Rate (Hz)')
    axtcE.grid('on')
    axtcI.set_title('Mean Population tuning curves, I')
    axtcI.set_xlabel('Orientation (deg)')
    axtcI.set_ylabel('Firing Rate (Hz)')
    axtcI.grid('on')
    axfr.set_ylabel('Mean population activity (Hz)')
    axfr.set_xlabel('p')
    axfr.set_xticks(xaxis)
    axfr.set_xticklabels(xtickLabels)
    plt.draw()
    plt.figure(fgtcE.number)   
    plt.legend(prop={'size': 10}, loc = 0)
    plt.figure(fgtcI.number)   
    plt.legend(prop={'size': 10}, loc = 0)    
#    figFolder = basefolder + '/nda/spkStats/figs/broad_i_tuning/rho' + rho + '/'
#    figFolder = basefolder + '/nda/spkStats/figs/broad_i_tuning/rho'
    figFolder = '/homecentral/srao/cuda/data/poster/figs/'
    np.save(figFolder + 'meanActivities', fr)
    #kb.keyboard()
    figName = 'meanTuningCurves_E'
    pageSizePreFactore = 1.
    Print2Pdf(fgtcE,  figFolder + figName, [pageSizePreFactore*4.6,  pageSizePreFactore*3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])
    figName = 'meanTuningCurves_I'
    pageSizePreFactore = 1.
    Print2Pdf(fgtcI,  figFolder + figName, [pageSizePreFactore*4.6,  pageSizePreFactore*3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])
    figName = 'p_vs_meanActivity'
    Print2Pdf(fgfr,  figFolder + figName, [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.15, 0.15, .78, .75])
 #   plt.ion()
  #  plt.show()
   # plt.draw()
#    kb.keyboard()
    
def main(dbNames, xtickLabels, rhos, legendLabels):
    plt.ioff()
    if(rho == '1'):
        # figE, axE = plt.subplots()
        # figI, axI = plt.subplots()
        # figm, axm = plt.subplots()
        colormap = plt.cm.gist_rainbow
        if(len(dbNames) > len(rhos)):
            colors = [colormap(i) for i in np.linspace(0, 0.9, len(dbNames))]
        else:
            colors = [colormap(i) for i in np.linspace(0, 0.9, len(rhos))]
        axE.set_color_cycle(colors)
        axI.set_color_cycle(colors)
    cvMeans = np.zeros((len(dbNames), 2))
    print "RHO = ", rho
    for k, kDb in enumerate(dbNames):
        print "Processint db:", kDb
        tc = np.load(basefolder + '/db/data/tuningCurves_' + kDb + 'rho' + rho + xi +'.npy')
        theta = np.arange(thetaStart, thetaEnd, thetaStep)
        if(k == 0):
 #           legendLabel = 0
            ne = NE #20000
            ni = NI #20000
#            theta = np.arange(0.0, 180.0, 22.5/2)
        else:
#            legendLabel = k #float(kDb[6:])/10.0
            ne = NE
            ni = NI
        circVariance = PopulationCircVar(tc, theta, ne, ni)
        cvMeans[k, :] = PlotCircVars(circVariance, ne, ni, axE, axI, legendLabels[k])
    plt.figure(figE.number)   
    plt.legend(prop={'size': 10}, loc = 0)
    plt.xlim([0.0, 1.0])
    plt.xlabel('Circular Variance')
    plt.ylabel('Normalized count')
    plt.title('Distribution of circular variance, E')
    plt.figure(figI.number)   
    plt.legend(prop={'size': 10}, loc = 0)
    plt.xlabel('Circular Variance')
    
    plt.ylabel('Normalized count')
    plt.title('Distribution of circular variance, I')
    plt.xlim([0.0, 1.0])
    labels = [item.get_text() for item in axE.get_xticklabels()]
    xticks = axI.get_xticks()
    tmp = int(len(labels) / 2.0)
    labels[tmp] = xticks[tmp]
    labels[0] =  "%s\nhighly tuned"%(xticks[0])
    labels[-1] =  "%s\nnot selective"%(xticks[-1])
    axE.set_xticklabels(labels)
    labels = [item.get_text() for item in axI.get_xticklabels()]
    xticks = axI.get_xticks()
    tmp = int(len(labels) / 2.0)
    labels[tmp] = xticks[tmp]
    labels[0] =  "%s\nhighly\ntuned"%(xticks[0])
    for ii, i in enumerate(np.arange(.2, .9, .2)):
#        print ii+1, i
        labels[ii+1] = '%s'%(i)
    labels[-1] =  "%s\nnot\nselective"%(xticks[-1])
#    labels[-1] = 1.0
    axE.set_xticklabels(labels)
    axI.set_xticklabels(labels)
#    figFolder = basefolder + '/nda/spkStats/figs/broad_i_tuning/rho' + rho + '/'
#    figFolder = basefolder + '/nda/spkStats/figs/broad_i_tuning/rho_compare/'
    figFolder = '/homecentral/srao/cuda/data/poster/figs/'    
    axE.xaxis.set_label_coords(0.5, -0.1);
    axI.xaxis.set_label_coords(0.5, -0.1);
    figName = 'ori_cvDistr_E_diffFFK'
    pageSizePreFactore = 1.
    Print2Pdf(figE,  figFolder + figName, [pageSizePreFactore*5.25,  pageSizePreFactore*4.], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.145, 0.15, .78, .75])
    figName = 'ori_cvDistr_I_diffFFK'
    Print2Pdf(figI,  figFolder + figName, [pageSizePreFactore*5.25,  pageSizePreFactore*4.], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.145, 0.15, .78, .75])

    xaxis = np.arange(1, len(dbNames) + 1, 1)
 #   xaxis = np.concatenate(([1], np.delete(np.arange(0.4, 2.1, .2), 3)))
    axm.plot(xaxis, cvMeans[:, 0], 'ko-', label = 'E')
    axm.plot(xaxis, cvMeans[:, 1], 'ro-', label = 'I')
    axm.set_xticks(xaxis)
    axm.set_xticklabels(xtickLabels)
    axm.grid('on')
    plt.draw()
    np.save(figFolder + 'cvmeans', cvMeans)
    plt.figure(figm.number)
    plt.xlabel('p')
    plt.ylabel('Mean CV')
    figName = 'cv_vs_cff'
    print 'saving in folder', figFolder

    Print2Pdf(figm,  figFolder + figName, [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.15, 0.15, .78, .75])

    # plt.ion()
    # plt.show()
    # kb.keyboard()

# ===== GLOBAL VARIABLES ==============
[NE, NI, thetaStart, thetaEnd, thetaStep] = DefaultArgs(sys.argv[1:7], [20000, 20000, 0, 180, 22.5])
NE = int(NE)
NI = int(NI)
thetaStart = float(thetaStart)
thetaEnd = float(thetaEnd)
thetaStep = float(thetaStep)
#dbNames = ['p1'] #g1xT3theta16Tr16'] # the first one is the control db
dbNames = []
for i in range(len(sys.argv[6:])):
    dbName = sys.argv[6+i]
    dbNames.append(dbName)
    print dbName
print [NE, NI, thetaStart, thetaEnd, thetaStep]
#=======================================


if __name__ == '__main__':
#    alphaBetas = [[1.0, 1.0], [0.5, 0.5], [0.75, 0.75], [1.5, 1.5], [2.0, 1.0], [1.0, 0.5]]
#     alphaBetas = [ [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 1.0], [1.0, 0.5]]
#    alphaBetas = [[1.0, .5], [1.0, 1.0], [1.0, 2.0]] 
#    alphaBetas = [[0.5, 0.5], [1.0, 1.0], [2.0, 1.0], [4.0, 1.0], [8.0, 1.0], [10.0, 1.0]]
#    alphaBetas = [[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]
#    alphaBetas = [[1, 1], [2, 1], [4, 1], [8, 1]]
    alphaBetas = [[8, 1], [12, 1.5], [16, 2]]
    alphaBetas = [[16, 2]]
    legendLabels = []
    xticklabels = []
    for i in range(len(alphaBetas)):
        legendLabels.append(r'$\alpha = %s, \beta = %s$'%(alphaBetas[i][0], alphaBetas[i][1]))
        xticklabels.append(r'$\alpha = %s$'%(alphaBetas[i][0]) + '\n' + r'$\beta = %s$'%(alphaBetas[i][1]))
    #    print xticklabels
    if(len(dbNames) != len(alphaBetas)):
        print 'len of dbnames and alphabetas do not match'
        sys.exit(0)
    else:
        main2(dbNames, legendLabels = legendLabels, xtickLabels = xticklabels)
#        rhos = [0.1, 0.5, 0.6, 0.7]
        rhos = [0.5]
#        legendLabels = [r'$\rho = %s$'%(float(xx[1])) for xx in enumerate(rhos)]
        legendLabels = ['cntrl', 'kff_I = 400', 'Kff_I = 800', 'kff_EI = 200']
        print legendLabels
        figE, axE = plt.subplots()
        figI, axI = plt.subplots()
        figm, axm = plt.subplots()
        for kk, krho in enumerate(rhos):
            rho = 0.5 #'%s'%(int(krho*10))
            main(dbNames, xticklabels, rhos, [legendLabels[kk]])        
        plt.ion()
        plt.show()
        plt.waitforbuttonpress()
    
        

    
    

    
#python CompareCircVarOfDBs.py [] [] [] [] [] '0' p4 p6 p8 p12 p14 p16 p18 p20


#        alphaBetas = [[1, 1], [1, .8], [1, .5], [.8, .8], [.5, .5]]
        # alphaBetas = [[1.0, 1.0], [1.0, 0.5], [1.0, 0.25]] 
        # legendLabels = []
        # for i in range(len(alphaBetas)):
        #     legendLabels.append(r'$\alpha = %s, \beta = %s$'%(alphaBetas[i][0], alphaBetas[i][1]))

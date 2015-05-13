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
#import scipy.stats as stats
#from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf

def keyp(event):
    global figCounter
    if event.key == "f":
        figCounter += 1
        figname = 'crf_' + '%s'%(figCounter)
        print "saving figure as", figname, ' in', figFolder
        Print2Pdf(plt.gcf(),  figFolder + figname,  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])        #[6.3,  5.27], figFormat='png', labelFontsize = 10, tickFontsize=10, titleSize = 10.0)
#        kb.keyboard()

def dummy():
    a = 1;

def ObjHratio(cntrst, rmax, n, c50):
    # hyperbolic objective function
    # rmax * C^n / (C ^n + C50 ^n)
    return rmax * cntrst **n / (cntrst **n + c50 **n)

def ObjLog(cntrst, a, b):
    # log10 objective function
    # a + b * log10(C)
    return a + b * np.log10(cntrst)

def FitHratio(cntrst, firingRate, IF_PLOT = False):
    popt = np.empty((3, ))
    estimateSD = np.empty((3, ))
    popt[:] = np.nan
    estimateSD = np.nan
    residual = np.nan
    try:
        popt, pcov = curve_fit(ObjHratio, cntrst, firingRate, p0 = (firingRate[-1], 1.1, 1e-3))
        estimateSD = np.sqrt(np.diag(pcov))
        residual = np.sum((firingRate - ObjHratio(cntrst, *popt)) **2)
#        print 'squared residual = ', residual
        if(IF_PLOT):
            plt.figure(12)
            xx = np.arange(1.1, 100.0, 0.1)
            yy = ObjHratio(xx, *popt)
            plt.loglog(cntrst, firingRate, 'ko')
            plt.loglog(xx, yy, 'k')
            #plt.plot(cntrst, firingRate, 'ko')
            #plt.plot(xx, yy, 'k')
            #kb.keyboard()
            #plt.text(plt.xlim()[1] + 2.0, plt.ylim()[1] * 0.45, r'$R_{max} \frac{C^n}{C^n + C_{50}^n}$' + '\n\n Rmax = %.4s +- %.4s \n n = %.4s +- %.4s \n C50 = %.4s  +- %.4s'%(popt[0], estimateSD[0], popt[1], estimateSD[1], popt[2], estimateSD[2]), ha='center', va='center', fontsize = 12)
            plt.text(3.0, 0.5 * plt.ylim()[1] , r'$R_{max} \frac{C^n}{C^n + C_{50}^n}$' + '\n\n Rmax = %.4s +- %.4s \n n = %.4s +- %.4s \n C50 = %.4s  +- %.4s'%(popt[0], estimateSD[0], popt[1], estimateSD[1], popt[2], estimateSD[2]), ha='center', va='center', fontsize = 10)
            plt.show()
    except:
        dummy()
    return popt, estimateSD, residual

def FitLog(cntrst, firingRate, IF_PLOT = False):
    popt = np.empty((2, ))
    estimateSD = np.empty((2, ))
    popt[:] = np.nan
    estimateSD = np.nan
    residual = np.nan
    try:
        popt, pcov = curve_fit(ObjLog, cntrst, firingRate, p0 = (1.1, 5.1))
        estimateSD = np.sqrt(np.diag(pcov))
        residual = np.sum((firingRate - ObjLog(cntrst, *popt)) **2)
        #print 'squared residual = ', resudial
        if(IF_PLOT):
            plt.figure(12)
            xx = np.arange(1.1, 100.0, 0.1)
            yy = ObjLog(xx, *popt)
            plt.loglog(cntrst, firingRate, 'ko')
            plt.loglog(xx, yy, 'k')
            #plt.text(plt.xlim()[0] + 2.0, plt.ylim()[1] * 0.45, r'$a + b * log_{10}(C)$' + '\n \n a = %.4s +- %.4s \n b = %.4s +- %.4s'%(popt[0], estimateSD[0], popt[1], estimateSD[1]), ha='center', va='center', fontsize = 12)
            plt.text(3.0, plt.ylim()[1] * 0.5, r'$a + b * log_{10}(C)$' + '\n \n a = %.4s +- %.4s \n b = %.4s +- %.4s'%(popt[0], estimateSD[0], popt[1], estimateSD[1]), ha='center', va='center', fontsize = 10)
            plt.show()
    except:
        dummy()
    return popt, estimateSD, residual


figCounter = -1
contrasts = np.array(sys.argv[1:], dtype = 'int')
NE = 10000
NI = 10000
nNeurons = NE + NI
K = 1000
figFolder = basefolder + '/nda/spkStats/figs/crf/k%s'%(K) + '/'
nTheta = 8
if(K == 1000):
    dbNameBase = 'crfT30C'
elif(K == 400):
    dbNameBase = 'crfT30K400C'
elif(K == 100):
    dbNameBase = 'crfT25K100C'    
contrastLevels = np.array([1.5, 2.5, 4, 5, 6, 10, 20, 30, 50, 60, 70, 80, 90, 100]) # cntrst levels for k = 1000
#contrastLevels = np.array([1.4, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 60.0, 100.0]) # cntrst levels for k = 400
#contrastLevels = np.array([1.4, 2.0, 3.0, 4.0, 6.0, 60.0, 100.0]) # cntrst levels for k = 100
nContrastLevels = contrasts.size
nNeurons = 20000
nTheta = 8
IF_PLOT = True
if IF_PLOT:
    fg12 = plt.figure(12)
    fg12.canvas.mpl_connect('key_press_event', keyp)
fitFunc = 'hratio' #'log' 
tuningCurves = np.zeros((nNeurons, nTheta, nContrastLevels))
circVars = np.zeros((nNeurons, nContrastLevels))
dataFolder = basefolder + '/db/data/'
for kk, kCntrst in enumerate(contrasts):
    kdbName = dbNameBase + '%s'%(kCntrst)
    print 'processing db: ', kdbName
    tc = np.load(dataFolder + 'tuningCurves_' + kdbName + '.npy')
    # mtc = np.max(tc, 1)
    # c, b = np.histogram(mtc, 200)
    # plt.plot(c, b[:-1], label = '%s'%(kCntrst))
    po = np.argmax(tc, 1)
#    plt.plot(np.roll(tc[111, :], -3), label = '%s'%(kCntrst))
 #   plt.waitforbuttonpress()
    for mNeuron in range(nNeurons):
        tuningCurves[mNeuron, :, kk] = np.roll(tc[mNeuron, :], -1 * po[mNeuron])
    circVars[:, kk] = np.load(dataFolder + 'Selectivity_' + kdbName + '.npy')
plt.ion()
plt.plot(contrastLevels,np.nanmean(circVars[:NE, :], 0), 'ko-', label='E')
plt.plot(contrastLevels,np.nanmean(circVars[NE:, :], 0), 'ro-', label='I')
plt.xlabel('Contrast')
plt.ylabel('Circular variance')
plt.title('Mean population cv, K = %s'%(K)) 
Print2Pdf(plt.gcf(),  figFolder + 'mean_circvar_',  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])
kb.keyboard()
cvThreshold = 0.5
theta = np.arange(-90, 90.0, 22.5)
# colormap = plt.cm.gist_rainbow
# colors = [colormap(i) for i in np.linspace(0, 0.9, nContrastLevels)]
# fgE, axE = plt.subplots()
# fgI, axI = plt.subplots()
# axE.set_color_cycle(colors)
# axI.set_color_cycle(colors)
# for kk, kCntrst in enumerate(contrasts):
#     kdbName = dbNameBase + '%s'%(kCntrst)
#     tmpCV = np.load(dataFolder + 'Selectivity_' + kdbName + '.npy')
#     validIdx = np.logical_and(~np.isnan(tmpCV), tmpCV < cvThreshold)
#     tc = np.roll(np.mean(tuningCurves[validIdx[:NE], :, kk], 0), 4)
#     axE.plot(theta, tc/tc.max(), 'o-', label = 'C %s'%(contrastLevels[kk]))
#     tc = np.roll(np.mean(tuningCurves[validIdx[NE:], :, kk], 0), 4)
#     axI.plot(theta, tc/tc.max(), 'o-', label = 'C %s'%(contrastLevels[kk]))
# axI.set_title('Normalized population tuning curve, I neurons')
# axE.set_title('Normalized population tuning curve, E neurons')
# axE.set_xlabel('PO (degrees)')
# axI.set_xlabel('PO (degrees)')
# axE.legend(loc=0, prop={'size':10})
# axI.legend(loc=0, prop={'size':10})
# plt.show()
# Print2Pdf(fgE,  figFolder + 'mean_normalized_tuningCurves_E_cvgt8',  [5.8,  5.0], figFormat='png', labelFontsize = 14, tickFontsize=14, titleSize = 14.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])
# Print2Pdf(fgI,  figFolder + 'mean_normalized_tuningCurves_I_cvgt8',  [5.8,  5.0], figFormat='png', labelFontsize = 14, tickFontsize=14, titleSize = 14.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])
plt.figure()
mtc = np.nanmean(tuningCurves[:NE, 0, :], 0)
plt.loglog(contrastLevels, mtc, 'ko-', label = 'E')
mtc = np.nanmean(tuningCurves[NE:, 0, :], 0)
plt.loglog(contrastLevels, mtc, 'ro-', label = 'I')
plt.xlabel('Contrast')
plt.ylabel('Firing rate (Hz)')
plt.title('Mean CRFs, K = %s'%(K))
plt.legend(loc=2)
Print2Pdf(plt.gcf(),  figFolder + 'mean_crf_K%s'%(K), [6.3,  5.27], figFormat='png', labelFontsize = 10, tickFontsize=10, titleSize = 10.0)        
kb.keyboard()
if(fitFunc == 'hratio'):
    paramfits = np.empty((nNeurons, 3))
elif(fitFunc == 'log'):
    paramfits = np.empty((nNeurons, 2))
paramfits[:] = np.nan
residuals = np.empty((nNeurons, ))
residuals[:] = np.nan
maxFiringRate = np.max(tc, 1)
firingThresh = 0.0
for l in range(nNeurons):
    if(IF_PLOT):
        neuronIdx = np.random.randint(0000,10000, 1)[0]
    else:
        neuronIdx = l
    if(not np.fmod(100.0 * neuronIdx / nNeurons, 10)):
        print "%s%% done"%(np.fmod(100.* float(neuronIdx) / nNeurons, 100.0))
    if(maxFiringRate[neuronIdx] >= firingThresh):
        try:
            if(fitFunc == 'hratio'):
                tmpparam = FitHratio(contrastLevels, tuningCurves[neuronIdx, 0, :], IF_PLOT)
            elif(fitFunc == 'log'):
                tmpparam = FitLog(contrastLevels, tuningCurves[neuronIdx, 0, :], IF_PLOT)
            paramfits[neuronIdx, :] = tmpparam[0]
            residuals[neuronIdx] = tmpparam[2]
        except RuntimeError:
            dummy()
        if(IF_PLOT):
           plt.title('K = %s, neuron# %s'%(K, neuronIdx))
           plt.xlabel('Contrast')
           plt.ylabel('Firing rate (Hz)')
           plt.show()
           if(plt.fignum_exists(fg12.number)):
              plt.figure(fg12.number)
              plt.waitforbuttonpress()
              plt.clf()
#              tmp = plt.ginput(1) # quick fix to make the figure wait here
    else:
        if(IF_PLOT):
            print "neuron discarded!"
print "Done"


if(fitFunc == 'hratio'):
    plt.figure()
    rmax = paramfits[:, 0]
    n = paramfits[:, 1]
    c50 = paramfits[:, 2]
    plt.hist(rmax[~np.isnan(rmax)], 50)
    plt.title(r'$R_{max}$ distribution')
    plt.xlabel(r'$R_{max}$')
    plt.ylabel('Counts')
    Print2Pdf(plt.gcf(),  figFolder + 'crf_rmaxDistr_K%s'%(K),  [4.3,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])        
    plt.figure()
    n = n[~np.isnan(n)]
    n = n[n < 5]
    n = n[n>0]
    plt.hist(n, 100)
    plt.xlabel('n')
    plt.ylabel('Counts')
    plt.title('n distribution')
    Print2Pdf(plt.gcf(),  figFolder + 'crf_n_K%s'%(K), [4.3,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .8, .75])        
    plt.figure()
    c50 = c50[~np.isnan(c50)]
    c50 = c50[c50 < 100]
    plt.hist(c50, 100)
    plt.title(r'$C_{50}$ distribution')
    plt.xlabel(r'$C_{50}$')
    plt.ylabel('Counts')
    Print2Pdf(plt.gcf(),  figFolder + 'crf_c50_K%s'%(K),  [4.3,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])        
elif(fitFunc == 'log'):
    plt.figure()
    a = paramfits[:, 0]
    b = paramfits[:, 1]
    plt.hist(a[~np.isnan(a)], 50)
    plt.title('Distribution of a')
    plt.xlabel('a')
    plt.ylabel('Counts')
    Print2Pdf(plt.gcf(),  figFolder + 'crf_a_logfit_K%s'%(K), [4.3,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .8, .75])        
    plt.figure()
    plt.hist(b[~np.isnan(b)], 50)
    plt.title('Distribution of b')
    plt.xlabel('b')
    plt.ylabel('Counts')
    Print2Pdf(plt.gcf(),  figFolder + 'crf_b_logfit_%s'%(K),  [4.3,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .8, .75])        
kb.keyboard()
#plt.figure()
#plt.hist(residuals[~np.isnan(residuals)], 50)
plt.title('Distr of resudials, K = %s, fit func: %s'%(K, fitFunc))
#plt.text(30, 3000, 'Mean = %.4s'%(np.nanmean(residuals)))
plt.xlabel('Sum of squared residuals')
plt.ylabel('Counts')
Print2Pdf(plt.gcf(),  figFolder + 'residuals_crf_k%s_%s'%(K, fitFunc), [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])        
#    plt.loglog(contrastLevels, tuningCurves[np.random.randint(0, 10000, 1)[0], 0, :], 'ko-')
kb.keyboard()

    

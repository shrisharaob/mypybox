basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
from functools import partial
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

def MyErrorbar(x, y, yerr, color, label, marker = '.-'):
     validIdx = ~np.logical_or(np.isnan(yerr), np.isinf(yerr))
     plt.errorbar(x[validIdx], y[validIdx], yerr = yerr[validIdx], ecolor = color, color = color, label = label,  fmt = '.-', linewidth = 0.2, markersize = 1.5)

def ObjFunc(x, tau, amp, offset):
     return amp * np.exp(-x / tau) + offset

def Objline(x, m, c):
     return (-1.0 * m * x) + c

def ExponentialFit(x, y, mean):
   IF_CONVERGE = False
   varLevel = 1e-2;
   maxIterations = 1
   iterationCount = 0;
   pini = np.array([10.0, 5.0])
   while(not IF_CONVERGE):
       iterationCount += 1
       popt, pcov = curve_fit(partial(ObjFunc, offset = mean), x, y, p0 = list(pini))
       pini = pini + 0.01 * np.random.rand(pini.size)
       IF_CONVERGE = True
       if(np.all(np.diag(pcov) < 1e-2)):
           IF_CONVERGE = True
       if(iterationCount > maxIterations):
           IF_CONVERGE = True
           break
   return popt, pcov

def ExpFitFile(filename, fr, neuronType = 1, IF_PLOT = False):
     print filename
     z = np.load(filename)[:, neuronType]
     neuronlabel = 'I'
     if neuronType == 0:
          neuronlabel = 'E'
     nDatapoints = z.shape
     x = np.arange(1000)
     tau = np.array([1.]) #, 6., 8., 10., 12.])
#     fr = np.mean(z[-200:]) # fix firing rates
     for k, kTau in enumerate(tau):
         idx = np.argmax(z)
         xAx = x[0:1000-idx]
         out = ExponentialFit(xAx, z[idx:], fr)
         popt = out[0]
         pcov = out[1]
         if IF_PLOT:
              plt.plot(z[idx:], 'ko-', markerfacecolor = 'none')        
              plt.xlabel('Time lag (ms)', fontsize=20)
              plt.ylabel('Firing rate (Hz)', fontsize=20)
              plt.title(neuronlabel + ' Population')
     return out + (fr, )

if __name__ == "__main__":
    #---- filenames = [sys.argv[1]]
    NE = 20000
    NI = 20000
    # p =     [1, 3, 4, 5, 8, 9, 95,  97]
    # plotP = [1, 3, 4, 5,  8, 9, 9.5, 9.7]
    p =     [5, 7,  8, 9, 95,  97]
    plotP = [5, 7,  8, 9, 9.5, 9.7]
    # p =     [0, 1, 2, 3, 4, 5, 6, 7, 75,  8, 85,  9, 95, 97]
    # plotP = [0, 1, 2, 3, 4, 5, 6, 7, 7.5, 8, 8.5, 9, 9.5, 9.7]    
#    tau_syn = [3, 6, 12, 24, 48]
    tau_syn = [12]
    filenames = ['long_tau_vs_ac_mat_bidir_I2I_p%s.npy'%(ip) for ip in p]
#   figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/i2i/'
    figFolder = '/homecentral/srao/cuda/data/powerlaw/figs/'
    figFormat = sys.argv[1] #'png'
    taudecayFiletag = sys.argv[2] #'png'    
    paramEstimates = np.empty((len(p), 2, 2)) 
    estimateStd = np.empty((len(p), 2, 2))
    xx = np.linspace(0, 2000, 2000)
    decayTimesVsTauE = []
    decayTimesVsTauE_error = []
    decayTimesVsTauI = []
    decayTimesVsTauI_error = []
    frOffsetFolder = '/homecentral/srao/cuda/data/poster/cntrl/'
    if taudecayFiletag == 'NI2E4':
        NE = 2
        NI = 20000
    elif taudecayFiletag == 'NI5E4':
        NE = 2
        NI = 50000
    elif taudecayFiletag == 'NI6E4':
        NE = 2
        NI = 60000
    elif taudecayFiletag == 'NI1E4':
        NE = 2
        NI = 10000        

    print '------------------------------------'
    print '             I NEURONS              '
    print '------------------------------------'
    filenames = []
    folderbase = '/homecentral/srao/cuda/data/powerlaw/'
    #---- I POPULATION
    for ll, lTau in enumerate(tau_syn):
        if taudecayFiletag == 'def':
            filenames = ['long_tau_vs_ac_mat_bidir_I2I_p%s.npy'%(ip) for ip in p]
        else:
#            filenames = ['long_tau_vs_ac_mat_bidir_%s_bidirI2I_tau%s_p%s.npy'%(taudecayFiletag, lTau, ip) for ip in p]
            filenames = ['long_tau_vs_ac_mat_bidir_%s_bidirI2I_tau%s_p%s.npy'%(taudecayFiletag, lTau, ip) for ip in p]            
            
        decayTimes = []
        decayTimesEstError = []
        print 'xxxxxxxxxxxxxxxxx  tau: ', lTau, 'ms   xxxxxxxxxxxxxxxxxxxxx'
        if lTau != 3:
            filenames = []
#            filenames = ['long_tau_vs_ac_mat_bidir_bidirI2I_p%s_tau%s_p%s.npy'%(ip, lTau, ip) for ip in p]
            filenames = ['long_tau_vs_ac_mat_bidir_%s_bidirI2I_tau%s_p%s.npy'%(taudecayFiletag, lTau, ip) for ip in p]            
        for kk, kFile in enumerate(filenames):
            print p[kk]
            frOffsetFolder = folderbase + '%s/tau%s/p%s/'%(taudecayFiletag, lTau, int(p[kk]));
            try:
                frOffset = np.loadtxt(frOffsetFolder + 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
            except IOError:
                try:
                    frOffset = np.loadtxt(frOffsetFolder + 'firingrates_xi0.8_theta0_0.%s_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
                except:
                    raise
            # FIT EXPONENTIAL
            out = ExpFitFile(kFile, frOffset[NE:].mean(), 1, True)
            paramEstimates[kk, :, 1] = out[0]
            estimateStd[kk, :, 1] = np.sqrt(np.diagonal(out[1]))
            decayTimes.append(out[0])
            decayTimesEstError.append(estimateStd[kk, :, 1])
            plt.plot(xx, ObjFunc(xx, *paramEstimates[kk, :, 1], offset = frOffset[NE:].mean()), '-',  label = 'p = 0.%s'%(p[kk]))
            plt.title('%s'%(p[kk]))
            # kb.keyboard()
            plt.ion()
            plt.waitforbuttonpress()
            plt.clf()
        decayTimesVsTauI.append(decayTimes)
        decayTimesVsTauI_error.append(decayTimesEstError)
    plt.legend()
    plt.xlim(0, 200)
    filename = 'Expfit1t_I_bidirI2I_'
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
    plt.close('all')
    #--- PLOT ESTIMATES
    MyErrorbar(np.array(plotP) * 0.1, paramEstimates[:, 0, 1], yerr = estimateStd[:, 0, 1], color = 'r', label = 'I')
    plt.xlabel('p')
    plt.ylabel(r'$\tau_{dec}(ms)$')
    plt.legend(frameon = False, numpoints = 1, loc = 2)
    plt.ylim(-50, 800)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0.0, 1.1, 0.5))
    plt.yticks(np.arange(0, 801, 400))
    filename = 'summary_Expfit_bidirI2I_'
    paperSize = [2.0, 1.5]
    axPosition = [0.27, 0.23, .65, 0.65]    
    Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) 
    # PLOT P VS TAU_DEC 
    plt.figure(1114)
    estColors = ['b', 'g', 'r', 'k', 'c']
    loglogdataI = []
    for kk, kTau in enumerate(tau_syn):
        kTauEstimates = np.array([decayList[0] for decayList in decayTimesVsTauI[kk]])
        kTauEstimateErrors = np.array([estError[0] for estError in decayTimesVsTauI_error[kk]])
        MyErrorbar(np.array(plotP)*.1, kTauEstimates / kTau, yerr = kTauEstimateErrors/kTau, label = r'$\tau = %sms$'%(kTau), color = estColors[kk])
        loglogdataI.append(kTauEstimates)
    plt.xticks(np.arange(0.0, 1.1, 0.5))
    plt.yticks(np.arange(0, 401, 200))
    plt.ylim(-50, 400)        
    plt.xlabel('p')
    plt.ylabel(r'$\frac{\tau_{dec}}{ \tau_{s}}$')
    filename = 'summary_Expfit_bidirI2I_I_'
    paperSize = [2.0, 1.5]
    Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) 
    fglogI, axlogI = plt.subplots()
    fglogEI, axlogEI = plt.subplots()
    slopesI = []    
    for kk, kTau in enumerate(tau_syn):
         fileptr = open('tau_dacay_taus%s.txt'%(kTau), 'w')
         poptI, pcov = curve_fit(Objline, np.log10(1.0 - np.array(plotP[0:])*.1), np.log10(loglogdataI[kk][0:] / float(kTau)))
         print "========= tau: %sms ========="%(kTau)
         print 'a     : %.5s +- %.5s'%(poptI[1], np.sqrt(np.diag(pcov)[1]))
         print 'gamma : %.5s +- %.5s'%(poptI[0], np.sqrt(np.diag(pcov)[0]))
         print "============================"         
         axlogI.loglog(1.0 - np.array(plotP)[0:]*.1, loglogdataI[kk][0:] / float(kTau), '.-', label = r'$\gamma = %.5s$'%(poptI[0]), linewidth = .2, markersize = 1.0)
         axlogEI.loglog(1.0 - np.array(plotP)[0:]*.1, loglogdataI[kk][0:] / float(kTau), 'r.-', label = r'I, $\gamma = %.4s$'%(poptI[0]), linewidth = .2, markersize = 4.0,  mfc = 'none', markeredgewidth = .2)
         for mm, mp in enumerate(plotP):
              fileptr.write("%s;%s\n"%(1.0 - np.array(mp)*.1, loglogdataI[kk][mm]))              
         fileptr.close()
    xxx = np.arange(0., .99, .1)
    np.save('loglog_I_' + taudecayFiletag, np.array([1-np.array(plotP) * 0.1, loglogdataI]))
    float(np.abs(poptI[1])) / ((1-xxx)**poptI[0])
    axlogEI.loglog((1-xxx), poptI[1] / ((1-xxx)**poptI[0]) , 'g', label = 'fit', linewidth = 0.5)
    axlogI.legend(frameon = False, loc = 0, prop={'size':5}, numpoints = 1, ncol = 2)
    axlogI.set_yticks(np.array([  1.00000000e+0, 1.00000000e+02, 1.00000000e+04]))    
    axlogI.set_xlabel(r'$1 - p$')
    axlogI.set_ylabel(r'$\frac{\tau_{dec}}{ \tau_{s}}$')
    axPosition = [0.3, 0.28, .6, 0.6]        
    filename = 'summary_loglog_bidirI2I_I_'
    Print2Pdf(fglogI,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    # plt.ion()
    # plt.show()
    # kb.keyboard()    

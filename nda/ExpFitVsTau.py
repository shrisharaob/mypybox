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
  #   print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
     print 'tau  : ', y
     print 'sigma: ', yerr
     validIdx = ~np.logical_or(np.isnan(yerr), np.isinf(yerr))
     plt.errorbar(x[validIdx], y[validIdx], yerr = yerr[validIdx], ecolor = color, color = color, label = label,  fmt = '.-', linewidth = 0.2, markersize = 1.5)

def ObjFunc(x, tau, amp, offset):
     return amp * np.exp(-x / tau) + offset

def Objline(x, m, c):
     return m*x + c

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
    p =     [0, 1, 2, 3, 4, 5, 6, 7, 75,  8, 85,  9, 95]
    plotP = [0, 1, 2, 3, 4, 5, 6, 7, 7.5, 8, 8.5, 9, 9.5]

    # p =     [0, 1, 2, 3, 4, 5, 6, 7, 75,  8, 85,  9]
    # plotP = [0, 1, 2, 3, 4, 5, 6, 7, 7.5, 8, 8.5, 9]    
    tau_syn = [3, 6, 12, 24, 48]
#    tau_syn = [3, 6, 12, 24]
    filenames = ['long_tau_vs_ac_mat_bidir_I2I_p%s.npy'%(ip) for ip in p]
    figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/i2i/'
    figFormat = sys.argv[1] #'png'
    paramEstimates = np.empty((len(filenames), 2, 2))
    estimateStd = np.empty((len(filenames), 2, 2))
    xx = np.linspace(0, 2000, 2000)
    decayTimesVsTauE = []
    decayTimesVsTauE_error = []
    decayTimesVsTauI = []
    decayTimesVsTauI_error = []
    frOffsetFolder = '/homecentral/srao/cuda/data/poster/cntrl/'
#    plt.ion()
    #---- E POPULATION    
    print 'E NEURONS'
    for ll, lTau in enumerate(tau_syn):
        decayTimes = []
        decayTimesEstError = []
        print 'xxxxxxxxxxxxxxxxxxxx  ', lTau, '   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        if lTau != 3:
            frOffsetFolder = '/homecentral/srao/cuda/data/poster/bidir/i2i/'            
            filenames = []
            filenames = ['long_tau_vs_ac_mat_bidir_bidirI2I_p%s_tau%s_p%s.npy'%(ip, lTau, ip) for ip in p]
            filenames1 = ['long_tau_vs_ac_mat_bidir_bidirI2I_tau%s_p%s.npy'%(lTau, ip) for ip in p]
#            filenames.remove(0)
 #           filename.insert(0, 'long_tau_vs_ac_mat_bidir_I2I_p%s_tau%s_p%s.npy'%(ip, ltau, ip))
        for kk, kFile in enumerate(filenames):
            filenames1 = ['long_tau_vs_ac_mat_bidir_bidirI2I_tau%s_p%s.npy'%(lTau, ip) for ip in p]
            print p[kk]
            if p[kk] == 0:
                frOffsetFolder = '/homecentral/srao/cuda/data/poster/cntrl/'                 
            else:
                frOffsetFolder = '/homecentral/srao/cuda/data/poster/bidir/i2i/p%s/'%(int(p[kk]))                       
            try:
                 frOffset = np.loadtxt(frOffsetFolder + 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
            except IOError:
                 try:
                      print 'trying0'
                      frOffset = np.loadtxt(frOffsetFolder + 'firingrates_xi0.8_theta0_0.%s_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
                 except IOError:
                      print 'trying1'
                      frOffset = np.loadtxt(frOffsetFolder + 'K500/firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
                 


            try:
                 out = ExpFitFile(kFile, frOffset[:NE].mean(), 0, True)
            except IOError:
                 out = ExpFitFile(filenames1[kk], frOffset[:NE].mean(), 0, True)
      #      print out[0], np.sqrt(np.diagonal(out[1]))
            paramEstimates[kk, :, 0] = out[0]
            estimateStd[kk, :, 0] = np.sqrt(np.diagonal(out[1]))
            decayTimes.append(out[0])
            decayTimesEstError.append(estimateStd[kk, :, 0])
            plt.plot(xx, ObjFunc(xx, *paramEstimates[kk, :, 0], offset = frOffset[:NE].mean()), '-',  label = 'p = 0.%s'%(p[kk]))#linewidth=2,
            # plt.show()
            # plt.waitforbuttonpress()
            # plt.clf()
        decayTimesVsTauE.append(decayTimes)
        decayTimesVsTauE_error.append(decayTimesEstError)
    plt.legend()
    plt.xlim(0, 200)
    filename = 'Expfit1t_E_bidirI2I_' 
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
    #    plt.show()
    plt.close('all')
    print '-------------------'
    print 'I NEURONS'
    filenames = ['long_tau_vs_ac_mat_bidir_I2I_p%s.npy'%(ip) for ip in p]    
    #---- I POPULATION
    for ll, lTau in enumerate(tau_syn):
        decayTimes = []
        decayTimesEstError = []
        print 'xxxxxxxxxxxxxxxxxxxx  ', lTau, '   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        if lTau != 3:
            filenames = []
            filenames = ['long_tau_vs_ac_mat_bidir_bidirI2I_p%s_tau%s_p%s.npy'%(ip, lTau, ip) for ip in p]    
        for kk, kFile in enumerate(filenames):
            print p[kk]                
            if p[kk] == 0:
                frOffsetFolder = '/homecentral/srao/cuda/data/poster/cntrl/'                 
            else:
                frOffsetFolder = '/homecentral/srao/cuda/data/poster/bidir/i2i/p%s/'%(int(p[kk]))
            print kFile
#            frOffset = np.loadtxtlong_tau_vs_ac_mat_bidir_I2I_tau6_p75_p7(frOffsetFolder + 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
            try:
                 frOffset = np.loadtxt(frOffsetFolder + 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
            except IOError:
                 try:
                      print 'trying0'
                      frOffset = np.loadtxt(frOffsetFolder + 'firingrates_xi0.8_theta0_0.%s_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
                 except IOError:
                      print 'trying1'
                      frOffset = np.loadtxt(frOffsetFolder + 'K500/firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_100000_tr0.csv'%(int(p[kk]), int(lTau)))
            
                 
#            out = ExpFitFile(kFile, frOffset[NE:].mean(), 1, True)
       #     print out[0], np.sqrt(np.diagonal(out[1]))
            try:
                 out = ExpFitFile(kFile, frOffset[NE:].mean(), 0, True)
            except IOError:
                 out = ExpFitFile(filenames1[kk], frOffset[NE:].mean(), 0, True)       
            paramEstimates[kk, :, 1] = out[0]
            estimateStd[kk, :, 1] = np.sqrt(np.diagonal(out[1]))
            decayTimes.append(out[0])
            decayTimesEstError.append(estimateStd[kk, :, 1])
            plt.plot(xx, ObjFunc(xx, *paramEstimates[kk, :, 1], offset = frOffset[NE:].mean()), '-',  label = 'p = 0.%s'%(p[kk]))#linewidth=2,
            # plt.ion()
            # plt.waitforbuttonpress()
            # plt.clf()        
        decayTimesVsTauI.append(decayTimes)
        decayTimesVsTauI_error.append(decayTimesEstError)         
    plt.legend()
    plt.xlim(0, 200)
    filename = 'Expfit1t_I_bidirI2I_'
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
#    plt.show()
    plt.close('all')
    
    #--- PLOT ESTIMATES
#    print paramEstimates[:, 0, 1]
    MyErrorbar(np.array(plotP) * 0.1, paramEstimates[:, 0, 0], yerr = estimateStd[:, 0, 0], color = 'k', label = 'E')
    MyErrorbar(np.array(plotP) * 0.1, paramEstimates[:, 0, 1], yerr = estimateStd[:, 0, 1], color = 'r', label = 'I')
    # plt.errorbar(p, paramEstimates[:, 0, 0], yerr = estimateStd[:, 0, 0], ecolor = 'k', color = 'k', label = 'E',  fmt = '.-')
    # plt.errorbar(p, paramEstimates[:, 0, 1], yerr = estimateStd[:, 0, 1], ecolor = 'r', color = 'r', label = 'I',  fmt = '.-')
#    plt.title('Exopnenial fit')
    plt.xlabel('p')
    plt.ylabel('Decay time(ms)')
    plt.legend(frameon = False, numpoints = 1, loc = 2)
#    plt.xlim(0, 0.85)
    # plt.xlim(0, 1.0)    
    # plt.xticks(np.arange(0, .81, 0.4))
    # plt.yticks(np.arange(0, 501, 250))
    plt.ylim(-50, 800)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0.0, 1.1, 0.5))
    plt.yticks(np.arange(0, 801, 400))
    
    filename = 'summary_Expfit_bidirI2I_'
#    paperSize = [1.71*1.65, 1.15*1.65]
    paperSize = [2.0, 1.5]
    axPosition = [0.27, 0.23, .65, 0.65]    
    Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.25, 0.26, .77, .5 * 0.961]) #[0.142, 0.15, .77, .74])
 #   plt.show()
    # plt.ion()
    # plt.show()
    # kb.keyboard()
    plt.figure(1113)
    estColors = ['b', 'g', 'r', 'k', 'c']
    loglogdataE = []    
    for kk, kTau in enumerate(tau_syn):
        kTauEstimates = np.array([decayList[0] for decayList in decayTimesVsTauE[kk]])
        kTauEstimateErrors = np.array([estError[0] for estError in decayTimesVsTauE_error[kk]])
        MyErrorbar(np.array(plotP)*.1, kTauEstimates, yerr = kTauEstimateErrors, label = r'$\tau = %sms$'%(kTau), color = estColors[kk])
        loglogdataE.append(kTauEstimates)
#    plt.ion()
#    plt.legend(loc = 2, frameon = False, numpoints = 1, ncol = 3)
 #   plt.xticks(np.arange(0.0, 0.81, 0.4))
    plt.ylim(-50, 1600)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0.0, 1.1, 0.5))
    plt.yticks(np.arange(0, 1801, 900))
#    plt.yticks(np.arange(0, 501, 250))
    plt.xlabel('p')
    plt.ylabel('Decay time(ms)')
    plt.legend([r'$\tau_{s}=3ms$', r'$\tau_s=6ms$', r'$\tau_s=12ms$'], frameon = False, ncol = 1, loc = 2, numpoints = 1, prop={'size':8}); plt.draw()
#    plt.title('E population')

    filename = 'summary_Expfit_bidirI2I_E_'
#    paperSize = [1.71*1.65, 1.15*1.65]
#    paperSize = [1.15*1.65, 1.15*1.65]
    paperSize = [2.0, 1.5]
    axPosition = [0.3, 0.3, 0.6, 0.6]
    Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.142, 0.15, .77, .74])    



    plt.figure(1114)
    estColors = ['b', 'g', 'r', 'k', 'c']
    loglogdataI = []
    for kk, kTau in enumerate(tau_syn):
        kTauEstimates = np.array([decayList[0] for decayList in decayTimesVsTauI[kk]])
        kTauEstimateErrors = np.array([estError[0] for estError in decayTimesVsTauI_error[kk]])
    #    print kTau, 'ms: ', kTauEstimates
        MyErrorbar(np.array(plotP)*.1, kTauEstimates, yerr = kTauEstimateErrors, label = r'$\tau = %sms$'%(kTau), color = estColors[kk])
        loglogdataI.append(kTauEstimates)
#        axlog.loglog(1.0 - np.array(p)*.1, kTauEstimates)        
#    plt.ion()
#    plt.xticks(np.arange(0.0, .81, 0.4))
    plt.xticks(np.arange(0.0, 1.1, 0.5))
    plt.yticks(np.arange(0, 1801, 900))
    plt.ylim(-50, 1800)        
    plt.xlabel('p')
    plt.ylabel('Decay time(ms)')    
 #   plt.title('I population')
#    plt.legend(loc = 2, frameon = False, numpoints = 1)
    filename = 'summary_Expfit_bidirI2I_I_'
   # paperSize = [1.71*1.65, 1.15*1.65]
    paperSize = [2.0, 1.5]
#    axPosition = [0.3, 0.224, .65, 0.65]    
    Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.2, 0.26, .77, .5 * 0.961]) #[0.142, 0.15, .77, .74])        
#    plt.show()
  #  kb.keyboard()
    fglogE, axlogE = plt.subplots()
    fglogI, axlogI = plt.subplots()
    slopesE = []
    slopesI = []
    
    for kk, kTau in enumerate(tau_syn):
         fileptr = open('tau_dacay_taus%s'%(kTau), 'w')
         poptE, pcov = curve_fit(Objline, np.log10(1.0 - np.array(plotP[3:])*.1), np.log10(loglogdataE[kk][3:] / float(kTau)))         
         poptI, pcov = curve_fit(Objline, np.log10(1.0 - np.array(plotP[3:])*.1), np.log10(loglogdataI[kk][3:] / float(kTau)))         
         axlogE.loglog(1.0 - np.array(plotP[3:])*.1, loglogdataE[kk][3:] / float(kTau), '.-', label = r'$m = %.5s$'%(poptE[0]), linewidth = .2, markersize = 1.0)
         axlogI.loglog(1.0 - np.array(plotP)[3:]*.1, loglogdataI[kk][3:] / float(kTau), '.-', label = r'$m = %.5s$'%(poptI[0]), linewidth = .2, markersize = 1.0)
         for mm, mp in enumerate(plotP):
              fileptr.write('%.5s; %.5s'%(1.0 - np.array(mp)*.1, loglogdataE[kk][mm],  loglogdataI[kk][mm]))
         fileptr.close()
          
         
#    plt.show()
    axlogE.legend(frameon = False, loc = 0, prop={'size':6}, numpoints = 1)
#    axlogE.set_title('E neurons')
    axlogI.legend(frameon = False, loc = 3, prop={'size':6}, numpoints = 1)
 #   axlogI.set_title('I neurons')    

    axlogE.set_yticks(np.array([  1.00000000e+0, 1.00000000e+02, 1.00000000e+04]))
    axlogI.set_yticks(np.array([  1.00000000e+0, 1.00000000e+02, 1.00000000e+04]))    
    axlogE.set_xlabel(r'$1 - p$')
    axlogE.set_ylabel('Decay time(ms)')    
    axlogI.set_xlabel(r'$1 - p$')
    axlogI.set_ylabel('Decay time(ms)')    

    filename = 'summary_loglog_bidirI2I_E_'
#    paperSize = [4.0*1.6, 4.0]
    axPosition = [0.27, 0.28, .65, 0.65]    
    Print2Pdf(fglogE,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)

    filename = 'summary_loglog_bidirI2I_I_'
    Print2Pdf(fglogI,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)

#    kb.keyboard()    

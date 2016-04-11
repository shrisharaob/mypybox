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
     print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
     print 'tau  : ', y
     print 'sigma: ', yerr
     validIdx = ~np.logical_or(np.isnan(yerr), np.isinf(yerr))
     plt.errorbar(x[validIdx], y[validIdx], yerr = yerr[validIdx], ecolor = color, color = color, label = label,  fmt = '.-', linewidth = 0.2, markersize = 1.0)


def ObjFunc(x, tau, amp, offset):
     return amp * np.exp(-x / tau) + offset

    
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

def ExpFitFile(filename, neuronType = 1, IF_PLOT = False):
     z = np.load(filename)[:, neuronType]
     neuronlabel = 'I'
     if neuronType == 0:
          neuronlabel = 'E'
     nDatapoints = z.shape
     x = np.arange(1000)
     tau = np.array([1.]) #, 6., 8., 10., 12.])
     fr = np.mean(z[-200:]) # fix firing rates
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
#    filenames = [sys.argv[1]]
    p = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    filenames = ['long_tau_vs_ac_mat_bidir_I2I_p%s.npy'%(ip) for ip in p]
    figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/i2i/'
    figFormat = sys.argv[1] #'png'
    paramEstimates = np.empty((len(filenames), 2, 2))
    estimateStd = np.empty((len(filenames), 2, 2))
    xx = np.linspace(0, 1000, 1000)
   #---- E POPULATION
    print 'E NEURONS'
    for kk, kFile in enumerate(filenames):
         out = ExpFitFile(kFile, 0, True)
         print out[0], np.sqrt(np.diagonal(out[1]))
         paramEstimates[kk, :, 0] = out[0]
         estimateStd[kk, :, 0] = np.sqrt(np.diagonal(out[1]))
         plt.plot(xx, ObjFunc(xx, *paramEstimates[kk, :, 0], offset = out[2]), '-',  label = 'p = 0.%s'%(p[kk]))#linewidth=2,
    plt.legend()
    plt.xlim(0, 200)
    filename = 'Expfit1t_E_bidirI2I_' 
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
#    plt.show()
    plt.close('all')
    print '-------------------'
    print 'I NEURONS'             
   #---- I POPULATION
    for kk, kFile in enumerate(filenames):
         out = ExpFitFile(kFile, 1, True)
         print out[0], np.sqrt(np.diagonal(out[1]))         
         paramEstimates[kk, :, 1] = out[0]
         estimateStd[kk, :, 1] = np.sqrt(np.diagonal(out[1]))
         plt.plot(xx, ObjFunc(xx, *paramEstimates[kk, :, 1], offset = out[2]), '-',  label = 'p = 0.%s'%(p[kk]))#linewidth=2,
    plt.legend()
    plt.xlim(0, 200)
    filename = 'Expfit1t_I_bidirI2I_' 
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
#    plt.show()
    plt.close('all')
    #--- PLOT ESTIMATES
    print paramEstimates[:, 0, 1]
    MyErrorbar(np.array(p) * 0.1, paramEstimates[:, 0, 0], yerr = estimateStd[:, 0, 0], color = 'k', label = 'E')
    MyErrorbar(np.array(p) * 0.1, paramEstimates[:, 0, 1], yerr = estimateStd[:, 0, 1], color = 'r', label = 'I')    
    # plt.errorbar(p, paramEstimates[:, 0, 0], yerr = estimateStd[:, 0, 0], ecolor = 'k', color = 'k', label = 'E',  fmt = '.-')
    # plt.errorbar(p, paramEstimates[:, 0, 1], yerr = estimateStd[:, 0, 1], ecolor = 'r', color = 'r', label = 'I',  fmt = '.-')
 #   plt.title('Exopnenial fit')
    plt.xlabel('p')
    plt.ylabel('Decay fit(ms)')
    plt.legend(frameon = False, numpoints = 1, loc = 2)
    plt.ylim(0, 50)
    plt.xticks(np.arange(0, .9, 0.4))
    plt.yticks(np.arange(0, 51, 25))               
    filename = 'summary_Expfit_bidirI2I_'
#    paperSize = [1.71*1.65, 1.15*1.65]
    paperSize = [2.0, 1.5]
    axPosition = [0.26, 0.25, .66, 0.6]    
    Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.2, 0.26, .77, .5 * 0.961]) #[0.142, 0.15, .77, .74])    
    
 #   plt.show()

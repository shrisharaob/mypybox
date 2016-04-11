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

def ObjFunc(x, tau0, tau1, amp1, amp0, offset):
     return amp0 * np.exp(-x / tau0) + amp1 * np.exp(-x / tau1) + offset
    
def ExponentialFit(x, y, mean):
   IF_CONVERGE = False
   varLevel = 1e-2;
   maxIterations = 1
   iterationCount = 0;
   pini = np.array([10.0, 5.0, 10.0, 1.0])
#   pini = np.array([10.0, 20.0, 1.0])
   while(not IF_CONVERGE):
       iterationCount += 1
       popt, pcov = curve_fit(partial(ObjFunc, offset = mean), x, y, p0 = list(pini))
#       popt, pcov = curve_fit(partial(ObjFunc, offset = mean, amp0 = amptau0), x, y, p0 = list(pini))       
#       print iterationCount
       pini = pini + 0.01 * np.random.rand(pini.size)
       IF_CONVERGE = True
       if(np.all(np.diag(pcov) < 1e-2)):
           IF_CONVERGE = True
       if(iterationCount > maxIterations):
           IF_CONVERGE = True
           break
   return popt, pcov

def MyErrorbar(x, y, yerr, color, label, marker = '.-'):
     print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
     print 'tau  : ', y
     print 'sigma: ', yerr
     validIdx = ~np.logical_or(np.isnan(yerr), np.isinf(yerr))
     plt.errorbar(x[validIdx], y[validIdx], yerr = yerr[validIdx], ecolor = color, color = color, label = label,  fmt = '.-')

     
def ExpFitFile(filename, neuronType = 1, IF_PLOT = False):
     z = np.load(filename)[:, neuronType]
     print  z.shape
     neuronlabel = 'I'
     if neuronType == 0:
          neuronlabel = 'E'
#     print neuronlabel
     nDatapoints = z.shape
     x = np.arange(1000)
     tau = np.array([1.]) #, 6., 8., 10., 12.])
     fr = np.mean(z[-200:]) # fix firing rates
     slowTime = np.zeros((tau.size,))
     fastTime = np.zeros((tau.size, ))
     firingrate = np.zeros((tau.size, ))
     for k, kTau in enumerate(tau):
         idx = np.argmax(z)
#         print idx
         xAx = x[0:1000-idx]
         out = ExponentialFit(xAx, z[idx:], fr)
         popt = out[0]
         pcov = out[1]
         if IF_PLOT:
 #             print 'estimates: ', out[0]
#              print 'std of estimate : ', np.sqrt(np.diagonal(out[1]))
              plt.plot(z[idx:], 'k', markerfacecolor = 'none')        
              plt.xlabel('Time lag (ms)', fontsize=20)
              plt.ylabel('Firing rate (Hz)', fontsize=20)
              plt.title(neuronlabel + ' Population')
     return out + (fr, )

if __name__ == "__main__":
#    filenames = [sys.argv[1]]
#    p = [0, 2, 4, 5, 6, 7, 8]
    p = [0, 1, 2, 3, 4,  5, 6, 7, 8]    
    filenames = ['long_tau_vs_ac_mat_bidir_I2I_p%s.npy'%(ip) for ip in p]
    figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/i2i/'
    figFormat = 'png'
    paramEstimates = np.empty((len(filenames), 4, 2))
    estimateStd = np.empty((len(filenames), 4, 2))
    xx = np.linspace(0, 1000, 1000)
   #---- E POPULATION
    print 'E NEURONS'
    for kk, kFile in enumerate(filenames):
         out = ExpFitFile(kFile, 0, True)
         taus = out[0][:2]
         idxsort = np.argsort(taus)
         print taus[idxsort], np.sqrt(np.diagonal(out[1])[idxsort])
         paramEstimates[kk, :, 0] = out[0]
         estimateStd[kk, :, 0] = np.sqrt(np.diagonal(out[1]))
         plt.plot(xx, ObjFunc(xx, *paramEstimates[kk, :, 0], offset = out[2]), '-',  label = 'p = 0.%s'%(p[kk]))#linewidth=2,
    plt.legend()
    plt.xlim(0, 200)
    filename = 'Expfit_E_bidirI2I_' 
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
#    plt.show()
    plt.close('all')
    print '-------------------'
    print 'I NEURONS'             
   #---- I POPULATION
    for kk, kFile in enumerate(filenames):
         out = ExpFitFile(kFile, 1, True)
         taus = out[0][:2]
         idxsort = np.argsort(taus)

         print taus, taus[idxsort], np.sqrt(np.diagonal(out[1])[idxsort])
         paramEstimates[kk, :, 1] = out[0]
         estimateStd[kk, :, 1] = np.sqrt(np.diagonal(out[1]))
         plt.plot(xx, ObjFunc(xx, *paramEstimates[kk, :, 1], offset = out[2]), '-',  label = 'p = 0.%s'%(p[kk]))#linewidth=2,
    plt.legend()
    plt.xlim(0, 200)
    filename = 'Expfit_I_bidirI2I_' 
    Print2Pdf(plt.gcf(),  figFolder + filename,  [4.6,  4.0], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
#    plt.show()
    plt.close('all')

    #--- PLOT ESTIMATES
    print paramEstimates[:, 0, 1]
    print p
    p1 = [4, 5, 7, 8]
    MyErrorbar(np.array(p1) * 0.1, paramEstimates[p1, 0, 0], yerr = estimateStd[p1, 0, 0], color = 'k', label = 'E')
    MyErrorbar(np.array(p1) * 0.1, paramEstimates[p1, 0, 1], yerr = estimateStd[p1, 0, 0], color = 'r', label = 'I')
    plt.xlim(0.3, .85)
    plt.title('Sum of exponetials fit')
    plt.xlabel('p')
    plt.ylabel('Slower time(ms)')
    plt.legend(frameon = False, numpoints = 1, loc = 2)
    filename = 'summary_sumExpfit_slow_bidirI2I_'
    paperSize = [1.71*1.65, 1.21*1.65]            
    Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.18, 0.26, .77, .6 * 0.961]) #[0.142, 0.15, .77, .74])    

    plt.figure()
    MyErrorbar(np.array(p) * 0.1, paramEstimates[:, 1, 0], yerr = estimateStd[:, 1, 0], color = 'k', label = 'E')
    MyErrorbar(np.array(p) *0.1, paramEstimates[:, 1, 1], yerr = estimateStd[:, 1, 0], color = 'r', label = 'I')         
    plt.xlim(-0.1, 0.85)
    plt.title('Sum of exponetials fit')
    plt.xlabel('p')
    plt.ylabel('Faster time(ms)')
    plt.legend(frameon = False, numpoints = 1, loc = 2)
    filename = 'summary_sumExpfit_fast_bidirI2I_' 
    Print2Pdf(plt.gcf(),  figFolder + filename, paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.18, 0.26, .77, .6 * 0.961]) #[0.142, 0.15, .77, .74])    

    plt.show()
         
              
    


   

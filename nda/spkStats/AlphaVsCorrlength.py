basefolder = "/homecentral/srao/Documents/code/mypybox"
import MySQLdb as mysql
import numpy as np
import scipy.stats as stat
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
import FanoFactorDynamics as ffd
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf


def ObjFunc(x, tau0, tau1, amp0, amp1, offset):
     return amp0 * np.exp(-x / tau0) + amp1 * np.exp(-x / tau1) + offset
    
def ExponentialFit(x, y, mean):
   IF_CONVERGE = False
   varLevel = 1e-2;
   maxIterations = 1
   iterationCount = 0;
   pini = np.array([1.0, 1e-3, 1.0, 1.0, 1.0])
#   pini = np.array([10.0, 20.0, 1.0, 1.0])
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

if __name__ == "__main__":
    filename = sys.argv[1]
    z = np.load(filename)
    print z.shape
    _, nAlpha, nDatapoints = z.shape
    neuronType = 'I'
    if(neuronType == 'I'):
        z = z[1, :, :]
    else:
        z = z[0, :, :]
    x = np.arange(2000)
    xx = np.linspace(0, 2000, 1000)
    tau = np.array([3.0])
    alpha = np.arange(0., 1.1, 0.1)
    #fr = np.array([2.936012, 2.909356, 2.896436, 2.874876, 2.86848 ,2.832106])
    fr = np.mean(z[:, 500:-1], 1) # fix firing rates
    print fr, z.shape
    slowTime = np.zeros((tau.size,))
    fastTime = np.zeros((tau.size, ))
    firingrate = np.zeros((tau.size, ))
    for k, kAlpha in enumerate(alpha):
        print kAlpha
        plt.plot(x[10:200], z[k, 10:200]) #, '.', label=r'$\alpha_{syn}=%s$'%((kAlpha, )))
        out = ExponentialFit(x[10:200], z[k, 10:200], fr[k])
        popt = out[0]
        amps = np.array([popt[2], popt[3]])
        consts = np.array([popt[0], popt[1]])
        gConst = np.max(consts)
        gAmp = amps[np.argmax(consts)]
        print "estimated decat time = ", consts
        plt.plot(xx, ObjFunc(xx, *popt, offset = fr[k]), 'k', linewidth=2)
        fastTime[k] = np.min(consts)
        firingrate[k] = popt[2]
        slowTime[k] = gConst
    plt.legend(prop={"size":16})
    plt.xlabel(r'Time lag (ms)', fontsize=20)
#    plt.xlabel(r'Time lag (ms)', fontsize=20)
    plt.ylabel('Firing rate (Hz)', fontsize=20)

#    plt.savefig('fit_two_time_scales02.png')
    
    plt.figure()
    plt.plot(tau, slowTime, 'ko-', label='slow time')
    plt.plot(tau, fastTime, 'ks-', label='fast time')
    plt.legend()
    plt.xlabel(r'$\tau_{syn}$ (ms)', fontsize=14)
    plt.ylabel(r'Estimated auto correlation decay ($\tau_{syn}$)', fontsize=14)
 #   plt.ylabel(r'Estimated auto correlation decay (ms)', fontsize=14)
    plt.xticks(tau)
    plt.xlim((2.9, 25))
    plt.grid()
 
#   plt.savefig('tau_vs_two_time_scales02.png') 

    print "done"
    plt.show()
    plt.waitforbuttonpress()

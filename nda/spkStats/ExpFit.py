import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
from functools import partial
import sys

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
#       print iterationCount
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
    nTau, nDatapoints = z.shape
    nTau = nTau - 1
#    x = z[0, :]
    x = np.arange(5000)
    xx = np.linspace(0, 6000, 1000)
    tau = np.array([3., 6., 8., 10., 12.])
    #fr = np.array([2.936012, 2.909356, 2.896436, 2.874876, 2.86848 ,2.832106])
    fr = np.mean(z[:, 4000:-1], 1) # fix firing rates
    print tau.shape
    slowTime = np.zeros((tau.size,))
    fastTime = np.zeros((tau.size, ))
    firingrate = np.zeros((tau.size, ))
    for k, kTau in enumerate(tau):
        print kTau
        idx = ((x / kTau) > 10)
        plt.plot(x[idx]/ kTau, z[k, idx], '.', label=r'$\tau_{syn}=%s$'%((kTau, )))
        out = ExponentialFit(x[idx]/kTau, z[k, idx], fr[k])
#        plt.plot(x[idx], z[k+1, idx], label=r'$\tau_{syn}=%s$'%((kTau, )))
#        out = ExponentialFit(x[idx], z[k+1, idx], fr[k])
        popt = out[0]
        amps = np.array([popt[2], popt[3]])
        consts = np.array([popt[0], popt[1]])
#        print amps
 #       print consts, np.argmin(consts)
        gConst = np.max(consts)
        gAmp = amps[np.argmax(consts)]
        print "estimated decat time = ", consts
        #plt.plot(xx / kTau, ObjFunc(xx / kTau, *popt, offset = fr[k]), 'k')
        plt.plot(xx, ObjFunc(xx, *popt, offset = fr[k]), 'k', linewidth=2)
        fastTime[k] = np.min(consts)
        firingrate[k] = popt[2]
        slowTime[k] = gConst
    
    plt.legend(prop={"size":16})
    plt.xlabel(r'Time lag ($\tau_{syn}$)', fontsize=20)
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

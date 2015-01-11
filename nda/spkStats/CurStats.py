import numpy as np
import pylab as plt
import os, sys
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
from multiprocessing import Pool
from functools import partial
from scipy.optimize import curve_fit

def CurrentAC(alpha, tau, maxLag = 1000, dt = 0.05):
#alpha = 0.0
#tau = 8
    print tau
    if(alpha == 1.0):
        fb = '/homecentral/srao/Documents/code/cuda/tmp/pc83/' 
    else :
        fb = '/homecentral/srao/Documents/code/cuda/cudanw/'

    filename = fb + 'currents_%s_%s.csv' %((alpha, tau)) 
#    maxLag = 2000
    cur = np.loadtxt(filename)
    nTimesteps, nNeurons = cur.shape
    ac = np.zeros((maxLag, nNeurons))
   # plt.ion()
    for kk in range(nNeurons):
        x = cur[4000:, kk]
        X = np.fft.fft(x)
        y = np.multiply(X, np.conj(X))
        y = np.abs(np.fft.ifft(y))
        y = y / y.size
        y = y / dt
        ac[:, kk] = y[:maxLag]
    acm = np.mean(ac, 1)
    np.save('avg_cur_tau%s_alpha%s'%((tau, alpha)), acm)
    return acm

alpha = np.array([0.0, 1.0])
tau = np.array([3, 6, 8, 10])
maxLag = 500
tt = np.arange(maxLag, dtype=float)
dt = 0.05 

def ObjFun(x, a, b, c):
    return np.exp(- x / a) * b + c

if __name__ == "__main__":
    f, ax = plt.subplots(tau.size)
    IF_COMPUTE = False
    if(IF_COMPUTE):
        p = Pool(4)
        acm = [] # list
        for ii, kAlpha in enumerate(alpha):
            print "alpha = ", kAlpha
            results = p.map(partial(CurrentAC, kAlpha), tau)
            acm.append(results)
        np.save('avg_cur_corr_summary', acm)
    else :
        results = np.load('avg_cur_corr_summary.npy')
#            alpha0 = results[0]
#            alpha1 = results[1]
        f, ax = plt.subplots(len(tau))
        tauFit = np.zeros((tau.size, alpha.size))
        for ii, kAlpha in enumerate(alpha):
            for jj, kTau in enumerate(tau):
                acm = results[ii][jj, :]
                yy = acm[:maxLag]
                yyt = np.arange(yy.size, dtype=float)
                popt, pcov = curve_fit(ObjFun, yyt/kTau, yy, (1.0, 1.0, 1.0)) 
                tauFit[jj, ii] = popt[0]                
                ax[jj].plot(yyt/ kTau, yy, '.', label=r'$\tau = %s\;, \alpha = %s, \; \tau_{fit} = %.2f$'%((kTau, kAlpha, popt[0])))
                ax[jj].plot(yyt/kTau, ObjFun(yyt / kTau, *popt), 'r')
                ax[jj].set_xlim((0, 180))
                ax[jj].legend()
            
    ax[jj].set_xlabel(r'time lag ($\tau$)')
    ax[0].set_title('avg net current correlation')
    plt.ion()
    plt.show()
    plt.figure()
    plt.plot(tau, tauFit[:, 0], 'o-', label=r'$\alpha = 0$')
    plt.plot(tau, tauFit[:, 1], 'o-', label=r'$\alpha = 1$')
    plt.show()
kb.keyboard()
                

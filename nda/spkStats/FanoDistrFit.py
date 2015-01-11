import numpy as np
import scipy.stats as stat
import code, sys
import pylab as plt
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
from enum import Enum
from scipy.optimize import curve_fit
import scipy.stats as stats
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append("/homecentral/srao/Documents/code/mypybox/nda")
import SpkStats
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig


def LognormalPDF(x, mu, sigma):
    x[x <= 0] = np.inf
    return np.exp(-0.5 * ((np.log(x) - mu) / sigma) **2 ) / (x * np.sqrt(2 * np.pi) * sigma);


def FitLogNorm(data, alpha = 0.05):
    data = data[~(np.isnan(np.log(data)))]
    ldata = np.log(data)
    dataLen = len(data)
    fit_mu = np.mean(ldata)
    fit_sigma = np.std(ldata)
    # confidence intervals 
    t_crit = stats.t.ppf([alpha / 2, 1 - alpha / 2],  dataLen - 1)
    fit_mu_ci = np.array([fit_mu + t_crit[0] * fit_sigma / np.sqrt(dataLen), fit_mu + t_crit[1] * fit_sigma / np.sqrt(dataLen)])
    chi2_crit = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], dataLen - 1)
    fit_sigma_ci = np.array([fit_sigma * np.sqrt((dataLen - 1) / chi2_crit[1]), fit_sigma * np.sqrt((dataLen - 1) / chi2_crit[0])])
    return fit_mu, fit_sigma, fit_mu_ci, fit_sigma_ci

def FitGamma(data):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)
    return fit_alpha, fit_loc, fit_beta

def LogNormMu(normmu, normsig):
    return np.exp(normmu +  0.5 * normsig ** 2) 

def LogNormSig(normmu, normsig):
    a = np.exp(normsig **2) - 1
    b = np.exp(2.0 * normmu + normsig ** 2)
    return np.sqrt(a * b)

print sys.argv

fp = open(sys.argv[1])
fn_generator = (line.split() for line in fp)

filenames = [tmp for tmp in fn_generator]
filenames = sum(filenames, [])

print filenames
fp.close()
allFilenames = filenames
print allFilenames
for ll in range(2):
    filenames = allFilenames[4*ll:4*(ll+1)]
    nfiles = len(filenames)
    logMu = np.zeros(nfiles)
    logSigma = np.zeros(nfiles)
    mu = np.zeros(nfiles)
    sig = np.zeros(nfiles)
    for kf, filename in enumerate(filenames):
        print 'processing file : ', filename
        ff = np.load(filename)
        #gammaParams = FitGamma(ff)
        lognParams = FitLogNorm(ff)
        print kf, lognParams
        mu[kf] = lognParams[0]
        sig[kf] = lognParams[1]
        logMu[kf]= LogNormMu(lognParams[0], lognParams[1])
        logSigma[kf]= LogNormSig(lognParams[0], lognParams[1])
        logMuCI = LogNormMu(lognParams[2], logSigma[kf])
        logSigmaCI = LogNormSig(logMu[kf], lognParams[3])
#        kb.keyboard()
        # plt.ion()
        # plt.hist(ff, 50)
        # xx = np.linspace(0.0, np.max(ff), 100)
        # plt.plot(xx, stats.gamma.pdf(xx, *gammaParams), 'k', linewidth=2)
        # plt.plot(xx, LognormalPDF(xx, *lognParams), 'r', linewidth=2)
        #plt.plot(kf, lognParams[0], 'ko', label=r'$\mu$,' + legend_label, markersize = 10.0)
#        plt.errorbar(kf, lognParams[0], yerr=[[lognParams[2][0]], [lognParams[2][1]]])
        plt_color = 'k'
        if(ll == 1):
            plt_color = 'r'
#        plt.errorbar(kf, lognParams[1], yerr=[[lognParams[3][0]], [lognParams[3][1]]], ecolor=plt_color)
        plt.errorbar(kf, lognParams[0], yerr=[[lognParams[2][0]], [lognParams[2][1]]], ecolor=plt_color)
#plt.errorbar(kf, lognParams[0], yerr=[np.matlib.repmat(logMuCI[0], 1, len(kf)), np.matlib.repmat(logMuCI[1], 1, len(kf))]
#plt.plot(kf, lognParams[0], 'ks', label=r'$\mu$,' + legend_label, markersize = 10.0)

    print "DONE ! "
    legend_label = 'E'
    plt_color_mu = 'ks-'
    plt_color_sig = 'ko-'
    if(ll == 1):
        legend_label = 'I' 
        plt_color_mu = 'rs-'
        plt_color_sig = 'ro-'
    plt.ion()
#    plt.plot(np.arange(nfiles), logMu, plt_color_mu, label=r'$\mu$,' + legend_label, markersize = 10.0)
 #   plt.plot(np.arange(nfiles), logSigma, plt_color_sig, label=r'$\sigma$,'+ legend_label, markersize = 10.0)
    plt.plot(np.arange(nfiles), mu, plt_color_mu, label=r'$\mu$,' + legend_label, markersize = 10.0)
#    plt.plot(np.arange(nfiles), sig, plt_color_sig, label=r'$\sigma$,'+ legend_label, markersize = 10.0)
plt.setp(plt.gca(), xticks = range(nfiles), xticklabels=['1x', '2x', '4x', '8x'])
plt.xlim((-1, 5))
plt.xlabel('bg input', fontsize = 20)
#plt.ylabel('Estimated params of the corresponding normal distr', fontsize = 14)
plt.ylabel('Estimated std of the corresponding normal distr', fontsize = 18)
plt.title(r'$\tau = 3ms, \; \alpha = 0.0,\;$ fano distr (lognorm fit)', fontsize = 18)
plt.legend(prop={"size":18})
kb.keyboard()

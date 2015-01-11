#script to compute fano factor as a function of bi-directional connectivity alpha
import MySQLdb as mysql
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


def FitLogNorm(data):
#    fit_sigma, fit_loc, fit_scale = stats.lognorm.fit(data)
    data = data[~(np.isnan(np.log(data)))]
    ldata = np.log(data)
    fit_mu = np.mean(ldata)
    fit_sigma = np.std(ldata)
#    return fit_sigma, fit_loc, fit_scale
    return fit_mu, fit_sigma
def FitGamma(data):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)
    return fit_alpha, fit_loc, fit_beta

if __name__ == "__main__":
#    alpha = np.array([0., .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    varyingParamName = 'alpha'
#    varyingParam = np.array([4., 4., 4., 4., 6., 6., 6., 6])
    varyingParam = np.array([0.0, 0.0, 0.5, 0.5])
    if(varyingParamName == 'tau'):
        tau = varyingParam
        alpha = 0.5        
    else:
        alpha = varyingParam
        tau = 3.0
    NE = 10000
    NI = 10000
    simDuration = 50000
    simDT = 0.05
    binSize = 2000.0  # fano factor observation window

    [dbName, alpha, simDuration, simDT, NE, NI, tau] = DefaultArgs(sys.argv[1:], ['', alpha, simDuration, simDT, NE, NI, tau])
#    print "alpha = ", alpha
    print dbName, alpha, simDuration, simDT, NE, NI, tau
    neuronsList = np.arange(NE + NI)
   # thetas = np.array([400, 412, 424, 448, 600, 612, 624, 648]) # corresponds to the theta index as stored in the datbase
    thetas = np.array([30, 35, 60, 65])
    p = Pool(4)
    ffunc = partial(SpkStats.AvgFano, dbName, neuronsList, simDuration, simDT, binSize)
    result = p.map(ffunc, thetas) 

    for ll, lParam in enumerate(thetas):
        print varyingParamName, ' = ', lParam
        if(varyingParamName == 'tau'):
#            theta = thetas[ll]
            tau = varyingParam[ll]
            lAlpha = alpha
        else:
            lAlpha = varyingParam[ll]
            

        out = result[ll]

        # plot mean spk count vs spk count var

        # # plt.ioff()
        # # plt.plot(out[0][NE:, 0], out[0][NE:, 1], 'r.', label = 'I')
        # # plt.plot(out[0][:NE, 0], out[0][:NE, 1], 'k.', label = 'E')
        # # plt.xlabel('Mean spike count')
        # # plt.ylabel('Spike count variance')
        # # plt.title(r'$\tau = %s \; \alpha = %s$'%(tau, alpha, ))
        # # plt.legend()
        # # fig_name_tag =  '' #str((ll+1)) + 'x'

 #       ReportFig('bidirectionality_with_ff_spkCnt_scatter_vs_tau'+dbName, 'Simulation time = %ss'%((simDuration * 1e-3, )), 'Spike count mean and variance with bi-directional connectivity', 'png', 'fano', 'spkCnt_mean_var_alpha%s_tau%s_T%s'%((lAlpha, tau, simDuration)) + fig_name_tag)


    
 #       kb.keyboard()
##        plt.clf()

        # I neurons
        meanSpkCnt = out[0][NE:, 0]
        spkVar = out[0][NE:, 1]
        tmpidx = ~(np.isnan(meanSpkCnt))
        tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
        meanSpkCnt = meanSpkCnt[tmpidx]
        spkVar = spkVar[tmpidx]
        ff = spkVar / meanSpkCnt
 #       np.save('../data/fanofactor_with_FFinput_I_alpha%s_%s_'%((lAlpha, dbName)) + fig_name_tag, ff) 
        ffi = np.mean(ff)
        print "I ff = ", np.mean(ff), "theta - ", lParam
  ##      cnts, bins = np.histogram(ff[~(np.isnan(ff))], 50)
  ##      plt.bar(bins[:-1], cnts, color = 'r', edgecolor = 'r', width = np.mean(np.diff(bins)))

#        np.save('../data/fano_distr_I_fanofactor_with_FFinput_alpha%s'%(lAlpha) + fig_name_tag, (bins[:-1], cnts))

        # E neurons
        meanSpkCnt = out[0][:NE, 0]
        spkVar = out[0][:NE, 1]
        tmpidx = ~(np.isnan(meanSpkCnt))
        tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
        meanSpkCnt = meanSpkCnt[tmpidx]
        spkVar = spkVar[tmpidx]
        ff = spkVar / meanSpkCnt
   #     np.save('../data/fano_distr_E_with_FFinput__alpha%s_%s_'%((lAlpha, dbName)) + fig_name_tag, ff) 
        #np.save('gammafit_params_E_alpha%s'%((lAlpha, )) + fig_name_tag, gammaparams)
        ffe = np.mean(ff)
        print "E ff = ", np.mean(ff), "theta - ", lParam
##        cnts, bins = np.histogram(ff[~(np.isnan(ff))], 50)
  # #        plt.bar(bins[:-1], cnts, color = 'k', edgecolor = 'k', width = np.mean(np.diff(bins)))
  # # #      np.save('../data/fano_distr_E_with_FFinput_alpha%s'%(lAlpha) + fig_name_tag, (bins[:-1], cnts))
  # #       plt.title(r'$\tau = %s \; \alpha = %s$'%(tau, alpha, ))
  # #       plt.xlabel('Fano factor', fontsize = 20)
  # #       plt.ylabel('Counts', fontsize = 20)
  # #       plt.legend(('I', 'E'))

#        ReportFig('bidirectionality_with_ff_fano_distr_vs_tau'+dbName, 'Distribution of fano factor <br> mean fano factor E = %s <br> mean fano factor I = %s <br> Simulation time = %ss'%((ffe, ffi, simDuration * 1e-3, )), 'Fano Factor with bi-directional connectivity', 'png', 'fano', 'fano_distr_alpha%s_tau%s_T%s'%((lAlpha, tau, simDuration)) + fig_name_tag)

  #      kb.keyboard()
        plt.clf()

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

if __name__ == "__main__":
    dbName = sys.argv[1] #"a9t3xi12" #"t3a100xi0" #"t3_vs_a_ff"
    computeType = sys.argv[2]
    alpha = np.array([0.0])
    NE = 10000
    NI = 10000
    simDuration = 1000
    simDT = 0.05
    tau = 3.0
    binSize = 1000.0  # fano factor observation window
    print alpha
    [alpha, simDuration, simDT, NE, NI, tau] = DefaultArgs(sys.argv[1:], [alpha, simDuration, simDT, NE, NI, tau])
    neuronsList = np.arange(NE + NI)
    thetas = np.arange(0., 180., 22.5)
    #thetas = np.arange(0, 360, 45)

    p = Pool(8)
    ffunc = partial(SpkStats.AvgFano, dbName, neuronsList, simDuration, simDT, binSize)
    result = p.map(ffunc, thetas) 

    for ll, lTheta in enumerate(thetas):
        print "alpha = ", lTheta
        out = result[ll]
        # # plot mean spk count vs spk count var
        # plt.ioff()
        # plt.plot(out[0][NE:, 0], out[0][NE:, 1], 'r.', label = 'I')
        # plt.plot(out[0][:NE, 0], out[0][:NE, 1], 'k.', label = 'E')
        # plt.xlabel('Mean spike count')
        # plt.ylabel('Spike count variance')
        # plt.title(r'$\tau = %s \; \alpha = %s$'%(tau, alpha, ))
        # plt.legend()
        # fig_name_tag =  '' #str((ll+1)) + 'x'
        # ReportFig('bidirectionality_with_ff_spkCnt_scatter_vs_tau'+dbName, 'Simulation time = %ss'%((simDuration * 1e-3, )), 'Spike count mean and variance with bi-directional connectivity', 'png', 'fano', 'spkCnt_mean_var__theta%s_alpha%s_tau%s_T%s'%((lTheta, alpha[0], tau, simDuration)) + fig_name_tag)
        # plt.clf()
#        # I neurons
        meanSpkCnt = out[0][NE:, 0]
        spkVar = out[0][NE:, 1]
        tmpidx = ~(np.isnan(meanSpkCnt))
        tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
        meanSpkCnt = meanSpkCnt[tmpidx]
        spkVar = spkVar[tmpidx]
        ff = spkVar / meanSpkCnt
        np.save('../data/fanofactor_with_FFinput_I_theta%s, alpha%s_%s_'%((lTheta, alpha, dbName)) + fig_name_tag, ff) 
        ffi = np.mean(ff)
        print "I ff = ", np.mean(ff)
 #       cnts, bins = np.histogram(ff[~(np.isnan(ff))], 50)
 #       plt.bar(bins[:-1], cnts, color = 'r', edgecolor = 'r', width = np.mean(np.diff(bins)))
#        np.save('../data/fano_distr_I_fanofactor_with_FFinput_theta%s_alpha%s'%(lTheta, alpha) + fig_name_tag, (bins[:-1], cnts))
  #      # E neurons
        meanSpkCnt = out[0][:NE, 0]
        spkVar = out[0][:NE, 1]
        tmpidx = ~(np.isnan(meanSpkCnt))
        tmpidx = np.logical_and(tmpidx, ~(meanSpkCnt == 0))
        meanSpkCnt = meanSpkCnt[tmpidx]
        spkVar = spkVar[tmpidx]
        ff = spkVar / meanSpkCnt
        np.save('../data/fano_distr_E_with_FFinput_theta%s_alpha%s_%s_'%((lThetaalpha, dbName)) + fig_name_tag, ff) 
        ffe = np.mean(ff)
        print "E ff = ", np.mean(ff)
        # cnts, bins = np.histogram(ff[~(np.isnan(ff))], 50)
        # plt.bar(bins[:-1], cnts, color = 'k', edgecolor = 'k', width = np.mean(np.diff(bins)))
        # np.save('../data/fano_distr_E_with_FFinput_alpha%s'%(alpha) + fig_name_tag, (bins[:-1], cnts))
        # plt.title(r'$\tau = %s \; \alpha = %s, \; theta = %s$'%(lTheta, tau, alpha, ))
        # plt.xlabel('Fano factor', fontsize = 20)
        # plt.ylabel('Counts', fontsize = 20)
        # plt.legend(('I', 'E'))
        # ReportFig('bidirectionality_with_ff_fano_distr_vs_tau'+dbName, 'Distribution of fano factor <br> mean fano factor E = %s <br> mean fano factor I = %s <br> Simulation time = %ss'%((ffe, ffi, simDuration * 1e-3, )), 'Fano Factor with bi-directional connectivity', 'png', 'fano', 'fano_distr_theta%s_alpha%s_tau%s_T%s'%((lTheta, alpha, tau, simDuration)) + fig_name_tag)
        # plt.clf()

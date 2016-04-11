# this code plots the mean squared error across two different initial conditions against the simulation time
import numpy as np
import code
import sys
import pylab as plt
from multiprocessing import Pool
from functools import partial
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from Print2Pdf import Print2Pdf
import SetAxisProperties as AxPropSet
from Print2Pdf2Axes import Print2Pdf2Axes
from YYPlot import YYPlot

def MeanSqrdDiff(fr0, fr1):
    # fr1 & fr0 are vector with firing rates
    n0 = fr0.size
    n1 = fr1.size
    out = np.nan
    if n0 == n1:
        frdiff = fr0 - fr1
        out = np.dot(frdiff, frdiff) / n0
    return out

def GetMSFrDiffInSimT(spkArray0, spkArray1, NE, NI, simT):
    # simT in ms
    nNeurons = NE + NI    
    fr0 = np.zeros((nNeurons, ))
    fr1 = np.zeros((nNeurons, ))
    simT = simT - 1000
    for kNeuron in np.arange(nNeurons):
        nSpks0 = spkArray0[spkArray0[:, 1] == kNeuron, 0]
        nSpks0 = np.size(nSpks0 <= simT)
        nSpks1 = spkArray1[spkArray1[:, 1] == kNeuron, 0]
        nSpks1 = np.size(nSpks1 <= simT)
        fr0[kNeuron] = nSpks0 / (simT * 1e-3) # fr in Hz
        fr1[kNeuron] = nSpks1 / (simT * 1e-3)
    mseE = MeanSqrdDiff(fr0[:NE], fr1[:NE])
    mseI = MeanSqrdDiff(fr0[NE:], fr1[NE:])
    return mseE, mseI

if __name__ == "__main__":
    [computeType, alpha, tau_syn, splitInNchunks, bidirType, NE, NI, simDuration, discardT] = DefaultArgs(sys.argv[1:], ['', '', '', '', 'i2i', 20000, 20000, 345000, 1000])
    NE = int(NE)
    NI = int(NI)
    simDurationtmp = 350000    
    simDuration = int(simDuration)
    splitInNchunks = int(splitInNchunks)
    simTEnds = np.linspace(discardT, simDuration, splitInNchunks)
    if computeType == 'compute':
        basefolder = "/homecentral/srao/cuda/data/pub/bidir/%s/"%(bidirType)
        foldername = "tau%s/p%s/"%((tau_syn, alpha))
#        foldername = "tau%s/p%s/fano/"%((tau_syn, alpha))
        filename0 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr2.csv'%(int(alpha), int(tau_syn), simDurationtmp)
        filename1 = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr3.csv'%(int(alpha), int(tau_syn), simDurationtmp)
        print 'loading file: ', filename0, '...',
        sys.stdout.flush()    
        spkArray0 = np.loadtxt(basefolder + foldername + filename0, delimiter = ';')
        spkArray0 = spkArray0[spkArray0[:, 0] > discardT, :]
        print 'done'
        print 'loading file: ', filename1, '...',
        sys.stdout.flush()
        spkArray1 = np.loadtxt(basefolder + foldername + filename1, delimiter = ';')
        spkArray1 = spkArray1[spkArray1[:, 0] > discardT, :]
        print 'done'
        mseVsSimT = np.empty((splitInNchunks, ))
        mseVsSimT[:] = np.nan
#        simTEnds = np.linspace(discardT, simDuration, splitInNchunks)
        print simTEnds[1:]
        pfunc = partial(GetMSFrDiffInSimT, spkArray0, spkArray1, NE, NI)
#        pyWorkers = Pool(min(12, splitInNchunks))
        # results = np.empty((simTEnds.size - 1, )) #np.array(pyWorkers.map(pfunc, simTEnds[1:]))
        # results[:] = np.nan
        results = []
        for i in range(splitInNchunks - 1):
            print simTEnds[i+1]
            results.append(pfunc(simTEnds[i + 1]))
        np.save('./data/MSE_%s_tau%s_p%s_T%s_new'%(bidirType, tau_syn, alpha, simDuration), results)
    else:
        print simTEnds[1:]
        fgE, axE = plt.subplots()
        fgI, axI = plt.subplots()
        results = np.load('./data/MSE_%s_tau%s_p%s_T%s_new.npy'%(bidirType, tau_syn, alpha, simDuration))
        axE.plot(simTEnds[1:] * 1e-3, results[:, 0], 'ko-', linewidth = 0.4, markersize = 1.5, markeredgecolor = 'k')
        axI.plot(simTEnds[1:] * 1e-3, results[:, 1], 'ro-', linewidth = 0.4, markersize = 1.5, markeredgecolor = 'r')
        # fg, ax0, ax1 =  YYPlot(simTEnds[1:] * 1e-3, results[:, 0], results[:, 1], 'o', True)
        AxPropSet.SetProperties(axE, axE.get_xlim(), axE.get_ylim(), '', r'MSE($Hz^2$)')
        AxPropSet.SetProperties(axI, axI.get_xlim(), axI.get_ylim(), 'T(s)', r'MSE($Hz^2$)')
        # axI.set_xlabel('T(s)')        
        # axE.set_ylabel(r'MSE($Hz^2$)')
        # axI.set_ylabel(r'MSE($Hz^2$)')
        # ymax0 = np.max(ax0.get_ylim())
        # ymax1 = np.max(ax1.get_ylim())        
        # if int(alpha) == 0:
        #     ax0.set_ylim([-2, ymax0])
        #     ax1.set_ylim([-2, ymax1])            
        # elif int(alpha) == 9:
        #     ax0.set_ylim([-2, ymax0])
        #     ax1.set_ylim([-200, ymax1])            
        # ax0.set_yticks([0, 0.5 * ymax0, ymax0])
        # ax1.set_yticks([0, 0.5 * ymax1, ymax1])
#        plt.xlim([simTEnds[1], simTEnds[-1] + 2e-3])
        axE.set_xticks(simTEnds[1:] * 1e-3)
        axI.set_xticks(simTEnds[1:] * 1e-3)                
        axE.set_xticklabels(['%.1f'%(tmpstim) for tmpstim in simTEnds[1:] * 1e-3])
        axI.set_xticklabels(['%.1f'%(tmpstim) for tmpstim in simTEnds[1:] * 1e-3])        
        figFolder = './figs/publication_figures/'
        filenameE = 'MSE_%s_tau%s_p%s_T%s_E'%(bidirType, tau_syn, alpha, simDuration)
        filenameI = 'MSE_%s_tau%s_p%s_T%s_I'%(bidirType, tau_syn, alpha, simDuration)        
        axPosition = [0.26, 0.28, .65, 0.65]
        paperSize = [4.0, 3.0]
        figFormat = 'eps'
        Print2Pdf(fgE,  figFolder + filenameE,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
        Print2Pdf(fgI,  figFolder + filenameI,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)        


    


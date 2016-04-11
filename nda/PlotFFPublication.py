basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

def plotff(ff, ax, IF_PRINT_XLABEL, nBins = 50):
    ax.hist(ff[~np.isnan(ff)], nBins, normed = 1, histtype = 'step', linewidth = 0.4)
#      ax.hist(ff[~np.isnan(ff)], nBins, normed = 1, histtype = 'step', label = legendLabel, linewidth = 0.5)
    if IF_PRINT_XLABEL:
        ax.set_xlabel('Fano factor')
    ax.set_ylabel("Probability")

if __name__ == '__main__':
    NE = 20000
    NI = 20000
    plt.ioff()
    alphas = [0, 2, 5, 8]
    tau_syn = 3
#    alphas = [0, 8]
    legendLabels = ['p = 0.%s'%(x[1]) for x in enumerate(alphas)]
    legendLabels[0] = 'control'
    bidirType = sys.argv[1]
    figFormat = sys.argv[2] #'png'
    figFolder = './figs/publication_figures/'
    datafolder = './data/'
    fanoCntrlFilename = 'fanofactor_p0_cntrl'
    if bidirType == 'e2e':
        filenames = ['fanofactor_p%s_E2E'%(x[1]) for x in enumerate(alphas[1:])]
        filetag = 'E2E'
        figFolder = figFolder + 'ac/e2e/'
        ffxlim = 2        
    elif bidirType == 'i2i':
        filenames = ['fanofactor_p%s_I2I_tau%s'%(x[1], tau_syn) for x in enumerate(alphas[1:])]
        filetag = 'I2I'
        figFolder = figFolder + 'ac/i2i/'        
        ffxlim = 5        
    elif bidirType == 'e2i':
        filenames = ['fanofactor_p%s_E2I'%(x[1]) for x in enumerate(alphas[1:])]
        filetag = 'E2I'
        figFolder = figFolder + 'ac/e2i/'        
        ffxlim = 2
    filenames.insert(0, fanoCntrlFilename)
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    print filenames
    for n, nFileName in enumerate(filenames):
        print 'loading file: ', nFileName
        pFanofactor = np.squeeze(np.load(datafolder + nFileName + '.npy'))
        ffe = pFanofactor[:NE]
        ffi = pFanofactor[NE:]
#        plotff(ffe, axE, False)
        plotff(ffe, axE, True)
        plotff(ffi, axI, True)
    axE.set_xlim(0, ffxlim)
    axI.set_xlim(0, ffxlim)
    axE.set_xticks(np.arange(0, ffxlim+1, ffxlim / 2.))
    axI.set_xticks(np.arange(0, ffxlim+1, ffxlim / 2.))
    plt.figure(figE.number)
    plt.figure(figI.number)
    if bidirType == 'e2e':
        axI.set_ylim(0, 3.0)
        axE.set_ylim(0, 3.0)
        axE.set_yticks([0, 1.5, 3.0])
        axI.set_yticks([0, 1.5, 3.0])
        plt.draw()
    elif bidirType == 'i2i':
        axE.set_yticks([0, 1.5, 3.0])
        axI.set_yticks([0, 1.5, 3.0])
        axE.set_xlim(0, 3)
        axI.set_xlim(0, 8)
        axE.set_xticks([0, 1.5, 3.0])
        axI.set_xticks([0, 4, 8])        
    elif bidirType == 'e2i':
       axE.set_ylim(0, 3)
       axI.set_ylim(0, 3)       
       axE.set_yticks([0, 1.5, 3])
       axI.set_yticks([0, 1.5, 3])
    paperSize = [2.0, 1.5]
    axPosition = [0.26, 0.26, .65, 0.65]
    filename = 'FFdistr_E_bidir' + filetag
    Print2Pdf(figE,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    filename = 'FFdistr_I_bidir' + filetag
    Print2Pdf(figI,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.142, 0.15, .77, .74])
    plt.show()

  

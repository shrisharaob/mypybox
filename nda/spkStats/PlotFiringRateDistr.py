import numpy as np
import pylab as plt
import sys, code
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb
from DefaultArgs import DefaultArgs





def plotfr(filename, filetag, filetype = 'tuning', NE = 20000, NI = 20000, nBins = 100):
    if filetype == 'tuning':
        tc = np.load('/homecentral/srao/db/data/tuningCurves_' + filename + '.npy')
        tce = tc[:NE, :].mean(1)
        tci = tc[NE:, :].mean(1)
        cntE, binsE = np.histogram(tce, nBins)
        cntI, binsI = np.histogram(tci, nBins)

    plt.ioff()
    plt.bar(binsE[:-1], cntE, color = 'k', edgecolor = 'k', width = np.mean(np.diff(binsE)))
    plt.xlabel('Firing rate (Hz)')
    plt.ylabel('Count')
    plt.xlim((0, tce.max()))
    plt.title('Firing rate distribution, E population')
    plt.text(0.5*tce.max(), 0.5*plt.ylim()[1], 'Mean: %.4s Hz'%(tce.mean()))
    figFolder = '/homecentral/srao/cuda/data/poster/figs/'
    figname = 'firing_rateDistr_E_' + filename
    Print2Pdf(plt.gcf(),  figFolder + figname,  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])

    plt.clf()
    plt.bar(binsI[:-1], cntI, color = 'k', edgecolor = 'k', width = np.mean(np.diff(binsI)))
    plt.xlabel('Firing rate (Hz)')
    plt.ylabel('Count')
    plt.xlim((0, tci.max()))
    plt.title('Firing rate distribution, I population')
    plt.text(0.5*tci.max(), 0.5*plt.ylim()[1], 'Mean: %.5s Hz'%(tci.mean()))
    figFolder = '/homecentral/srao/cuda/data/poster/figs/'
    figname = 'firing_rateDistr_I_' + filename
    Print2Pdf(plt.gcf(),  figFolder + figname,  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])
#    plt.show()
    kb.keyboard()

if __name__ == '__main__':
    [filename, filetag, filetype, NE, NI, nBins] = DefaultArgs(sys.argv[1:], ['','', 'tuning', 20000, 20000, 100])
    plotfr(filename, filetag, filetype, int(NE), int(NI), int(nBins))

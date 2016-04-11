import numpy as np
import pylab as plt
import sys, code
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
#import Keyboard as kb
from DefaultArgs import DefaultArgs

def plotfr(filename, filetag, filetype = 'tuning', NE = 20000, NI = 20000, nBins = 50):
    papersize = [2.0, 1.5]
    
     
    if filetype == 'tuning':
        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_' + filename + '.npy')
#        tc = np.load('/homecentral/srao/Documents/code/mypybox/db/data
        tce = tc[:NE, :].mean(1)
        tci = tc[NE:, :].mean(1)
        # cntE, binsE = np.histogram(tce, nBins)
        # cntI, binsI = np.histogram(tci, nBins)
#        bins = 10 ** np.linspace(np.log10(np.min(tce[tce>0])), np.log10(np.max(tce[tce>0])), nBins)
        bins = 10 ** np.linspace(np.log10(np.min(tce[tce>0])), np.log10(np.max(tce[tce>0])), nBins)
        plt.hist(tce, bins, normed = 1, histtype = 'step')
        plt.gca().set_xscale("log")
        plt.ion()
    #    plt.bar(binsE[:-1], cntE, color = 'k', edgecolor = 'k', width = np.mean(np.diff(binsE)))
        plt.xlabel('Firing rate (Hz)')
        plt.ylabel('PDF')
#        plt.xlim((0, tce.max()))
        plt.title('Firing rate distribution, E population')
        plt.text(0.5*tce.max(), 0.5*plt.ylim()[1], 'Mean: %.4s Hz'%(tce.mean()))
        figFolder = '/homecentral/srao/db/figs/publications/frdistr/'
        figname = 'firing_rateDistr_E_' + filename
        Print2Pdf(plt.gcf(),  figFolder + figname,  papersize, figFormat='png', labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])

        plt.waitforbuttonpress()
        plt.clf()
#        plt.bar(binsI[:-1], cntI, color = 'k', edgecolor = 'k', width = np.mean(np.diff(binsI)))
        bins = 10 ** np.linspace(np.log10(np.min(tci[tci>0])), np.log10(np.max(tci[tci>0])), nBins)
        plt.hist(tci, bins, normed = 1, histtype = 'step')
        plt.gca().set_xscale("log")            
        plt.xlabel('Firing rate (Hz)')
        plt.ylabel('PDF')
 #       plt.xlim((0, tci.max()))
        plt.title('Firing rate distribution, I population')
        plt.text(0.5*tci.max(), 0.5*plt.ylim()[1], 'Mean: %.5s Hz'%(tci.mean()))
#        figFolder = '/homecentral/srao/cuda/data/poster/figs/'
        figname = 'firing_rateDistr_I_' + filename
        Print2Pdf(plt.gcf(),  figFolder + figname,  papersize, figFormat='png', labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .77, .74])
    #    plt.show()
#        kb.keyboard()
        plt.waitforbuttonpress()

if __name__ == '__main__':
    [filename, filetag, filetype, NE, NI, nBins] = DefaultArgs(sys.argv[1:], ['','', 'tuning', 20000, 20000, 50])
    plotfr(filename, filetag, filetype, int(NE), int(NI), int(nBins))

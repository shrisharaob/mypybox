basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb


if __name__ == '__main__':
    NE = 20000
    NI = 20000
    plt.ioff()
#    tau_syn = np.array([3, 6, 12, 24])
    tau_syn = np.array([3, 6, 12])    
    tau_colors = ['k', 'b', 'g', 'r']
    alphas_default = [0, 2, 5, 8]
    alphas_default = range(9)    
    alphas = range(9)
    bidirType = sys.argv[1]
    figFormat = sys.argv[2] #'png'
    filetag = 'I2I'
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    figFolder = './figs/publication_figures'
    dataFolder = './data'
    print tau_syn
    for mm, mTau in enumerate(tau_syn):
        meanFanofactorsE = []
        meanFanofactorsI = []    
        mAlpha = alphas_default
        if mTau == 3:
            mAlpha = range(9)
#            mAlpha.remove(0)
        for nn, nAlpha in enumerate(mAlpha):
            if mTau == 3:
                if nAlpha == 0:
                    mnFilename = 'fanofactor_cntrl.npy'
                else:
                    mnFilename = 'fanofactor_bidirI2I_p%s.npy'%(nAlpha)
            else:
                if nAlpha == 0:
                    mnFilename = 'fanofactor_cntrl_tau%s.npy'%(mTau)
                else:
                    mnFilename = 'fanofactor__bidir' + filetag + '_p%s_tau%s.npy'%(int(nAlpha), int(mTau))
            print 'p =', nAlpha, 'tau = ', mTau, ' --> loding file: ', mnFilename
            fano = np.load(dataFolder + mnFilename)
            meanFanofactorsE.append(np.nanmean(fano[:NE]))
            meanFanofactorsI.append(np.nanmean(fano[NE:]))
        axE.loglog(1-np.array(mAlpha[0:]) * 0.1, np.array(meanFanofactorsE[0:]), '.-', label = r'$\tau = %s$'%(mTau), linewidth = 0.5, markersize = 2.0)
        axI.loglog(1-np.array(mAlpha[0:]) * 0.1, np.array(meanFanofactorsI[0:]), '.-', label = r'$\tau = %s$'%(mTau), linewidth = 0.5, markersize = 2.0)
#    axE.legend(numpoints = 1, ncol = 1, frameon = False, loc = 2)
 #   axE.set_title('E population')


    # plt.ion()
    # plt.show()
    # kb.keyboard()


 
    axE.set_xlabel(r'$1-p$')
    axE.set_ylabel(r'$\overline{FF}$')
    # axE.set_xlim(0, .85)
    # axE.set_xticks(np.arange(0, 0.85, 0.4))
    # axE.set_yticks(np.arange(0, 17., 8))

#    axI.legend(numpoints = 1, ncol = 2, frameon = False)
#    axI.set_title('I population')
    axI.set_xlabel(r'$1-p$')
    axI.set_ylabel(r'$\overline{FF}$')
    # axI.set_xlim(0, .85)
    # axI.set_xticks(np.arange(0, 0.85, 0.4))
    # axI.set_yticks(np.arange(0, 61., 30.))

#    paperSize = [1.71*1.65, 1.21*1.65]
    paperSize = [2.0, 1.5]
    axPosition = [0.3, 0.3, .6, 0.6]        

    axPositionInset = [0.15, 0.2, .745, .6 * 0.961]
#    axPosition = [0.1745, 0.26, .76, .6 * 0.961]

  #  plt.ion()
#    plt.show()
 #   kb.keyboard()

    filename = figFolder + 'fano_factor_vs_tau_summary_E_' + filetag
    Print2Pdf(figE,  filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    filename = figFolder + 'fano_factor_vs_tau_summary_I_' + filetag
    Print2Pdf(figI,  filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()
            
            
            

        
        
        
        

basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

def plotac(ac, ax, legendlabel, titletext):
    ax.plot(ac, label = legendlabel)
    ax.set_xlabel('Time lag(ms)')
    ax.set_ylabel('Mean activity(Hz)')
    ax.set_title(titletext)

if __name__ == '__main__':
    NE = 20000
    NI = 20000
    plt.ioff()
    alphas = [0, 2, 5, 8]
    legendLabels = ['p = 0.%s'%(x[1]) for x in enumerate(alphas)]
    legendLabels[0] = 'control'
    print legendLabels
    bidirType = sys.argv[1]
    figFormat = sys.argv[2] #'png'
    if bidirType == 'e2e':
        filenames = ['bidir_E2E_p%s'%(x[1]) for x in enumerate(alphas)]
        cvfilenames = ['bidirE2Ep%s'%(x[1]) for x in enumerate(alphas)]
        figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/e2e/'
        filetag = 'E2E'
    else:
        filenames = ['bidir_I2I_p%s'%(x[1]) for x in enumerate(alphas)]
        cvfilenames = ['bidirI2Ip%s'%(x[1]) for x in enumerate(alphas)]
        figFolder = '/homecentral/srao/cuda/data/poster/figs/bidir/i2i/'
        filetag = 'I2I'
    maxLag = 500
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    # plt.figure(figE.number)
    # axinset1 = plt.axes([.29, .45, .18*1.6, .18])
    # axinset2 = plt.axes([.7, .45, .18*1.6, .18]) #[.26, .67, .13*1.6, .13])
    # plt.figure(figI.number)
    # axinset1_I = plt.axes([.29, .45, .18*1.6, .18])
    # axinset2_I= plt.axes([.7, .45, .18*1.6, .18]) #[.26, .67, .13*1.6, .13])    
    figcvE, axinset1 = plt.subplots()
    figcv2E, axinset2 = plt.subplots()
    figcvI, axinset1_I = plt.subplots()
    figcv2I, axinset2_I = plt.subplots()    

    for n, nFileName in enumerate(filenames):
        nAC = np.load('long_tau_vs_ac_mat_' + nFileName + '.npy')
        cv = np.load('coefficientOFVar_' + cvfilenames[n] + '.npy')
#        cv = np.load('coefficientOFVarbidirI2Ip8.npy')
        cv1e = cv[:NE, 0]
        cv2e = cv[:NE, 1]
        cv1i = cv[NE:, 0]
        cv2i = cv[NE:, 1]        
        cv1e = cv1e[~np.logical_or(np.isnan(cv1e), np.isinf(cv1e))]
        cv2e = cv2e[~np.logical_or(np.isnan(cv2e), np.isinf(cv2e))]
        cv1i = cv1i[~np.logical_or(np.isnan(cv1i), np.isinf(cv1i))]
        cv2i = cv2i[~np.logical_or(np.isnan(cv2i), np.isinf(cv2i))]        
        axinset1.hist(cv1e, 200, normed = 1, histtype = 'step')
        axinset2.hist(cv2e, 200, normed = 1, histtype = 'step')
        axinset1_I.hist(cv1i, 200, normed = 1, histtype = 'step')
        axinset2_I.hist(cv2i, 200, normed = 1, histtype = 'step')        
        plotac(nAC[:, 0], axE, legendLabels[n], 'Population averaged AC, E')
        plotac(nAC[:, 1], axI, legendLabels[n], 'Population averaged AC, I')
    axE.set_xlim(0, maxLag)
    axI.set_xlim(0, maxLag)
    plt.figure(figE.number)
#    plt.legend(loc=0)
    plt.figure(figI.number)
 #   axI.legend(loc=0, frameon=False, ncol = 2)
#    axI.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False, numpoints = 1)
    if bidirType == 'e2e':
        print '',
        axI.set_ylim(0, 23.5)
        plt.draw()
    elif bidirType == 'i2i':
        print '',

    axinset1.set_xticks(np.arange(0, axinset1.get_xlim()[1] + 0.1, 1.0))
    axinset1.set_yticks(np.arange(0, axinset1.get_ylim()[1], 2.0))
    axinset1.set_ylabel('')
#    axinset1.set_xlabel('CV')
    axinset1.set_title('CV')    
    axinset1.grid('on')
#    axinset1.set_ylabel('Normalized count')
    axinset2.set_xticks(np.arange(0, axinset2.get_xlim()[1] + 0.1, 1.0))
    axinset2.set_yticks(np.arange(0, axinset2.get_ylim()[1], 2.0))        
    axinset2.set_ylabel('')
    axinset2.set_title('CV2')    
    axinset2.grid('on')
 #   axinset2.set_ylabel('Normalized count')
    
    axinset1_I.set_xticks(np.arange(0, axinset1_I.get_xlim()[1] + 0.1, 1.0))
    axinset1_I.set_yticks(np.arange(0, axinset1_I.get_ylim()[1], 4.0))
    axinset1_I.set_ylabel('')
    axinset1_I.set_title('CV')    
    axinset1_I.grid('on')
  #  axinset1_I.set_ylabel('Normalized count')    
    axinset2_I.set_xticks(np.arange(0, axinset2_I.get_xlim()[1] + 0.1, 1.0))
    axinset2_I.set_yticks(np.arange(0, axinset2_I.get_ylim()[1], 2.0))        
    axinset2_I.set_ylabel('')
    axinset2_I.set_title('CV2')    
    axinset2_I.grid('on')
   # axinset2_I.set_ylabel('Normalized count')


    axPosition = [0.1745, 0.26, .76, .6 * 0.961]
    if bidirType == 'e2e':
        axinset1.set_position([.29, .25, .15*1.6, .15])
        axinset2.set_position([.64, .25, .15*1.6, .15])
        axinset1_I.set_position([.29, .25, .15*1.6, .15]);
        axinset2_I.set_position([.64, .25, .15*1.6, .15]);

    if bidirType =='i2i':
        axPosition = [0.205, 0.26, .7, .6 * 0.961]
        axI.set_yticks(np.arange(0, 121, 40))
        

    paperSize = [1.71*1.65, 1.21*1.65]
    paperSizeInsets = [1.0, 1.0]
    axPositionInset = [0.15, 0.2, .745, .6 * 0.961]        
    filename = 'AC_E_bidir' + filetag
    Print2Pdf(figE,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    filename = 'AC_I_bidir' + filetag
    Print2Pdf(figI,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.142, 0.15, .77, .74])
    filename = 'CV_E_bidir' + filetag
    Print2Pdf(figcvE,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = axPositionInset) #[0.142, 0.15, .77, .74])
    filename = 'CV2_E_bidir' + filetag
    Print2Pdf(figcv2E,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = axPositionInset) #[0.142, 0.15, .77, .74])
    filename = 'CV_I_bidir' + filetag
    Print2Pdf(figcvI,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = axPositionInset) #[0.142, 0.15, .77, .74])
    filename = 'CV2_I_bidir' + filetag
    Print2Pdf(figcv2I,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = axPositionInset) #[0.142, 0.15, .77, .74])



    # plt.ion()
    # plt.show()
    # kb.keyboard()

  

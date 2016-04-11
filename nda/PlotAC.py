basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

def plotac(ac, ax, legendlabel, titletext, IF_PRINT_XLABEL):
    ax.plot(ac, label = legendlabel, linewidth = 0.5)
    if IF_PRINT_XLABEL:
        ax.set_xlabel('Time lag(ms)')
    ax.set_ylabel('Activity(Hz)')
#    ax.set_title(titletext)

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
    figFolder = './figs/publication_figures/'
    datafolder = './data/'
    if bidirType == 'e2e':
        filenames = ['bidir_E2E_p%s'%(x[1]) for x in enumerate(alphas)]
        cvfilenames = ['bidirE2Ep%s'%(x[1]) for x in enumerate(alphas)]
        figFolder = '/homecentral/srao/cuda/data/pub/figs/bidir/e2e/'
        filetag = 'E2E'
        maxLag = 50        
    elif bidirType == 'i2i':
        filenames = ['bidir_I2I_p%s'%(x[1]) for x in enumerate(alphas)]
        cvfilenames = ['bidirI2Ip%s'%(x[1]) for x in enumerate(alphas)]
        figFolder = '/homecentral/srao/cuda/data/pub/figs/bidir/i2i/'
        filetag = 'I2I'
        maxLag = 200        
    elif bidirType == 'e2i':
        filenames = ['bidir_bidirE2I_p%s'%(x[1]) for x in enumerate(alphas)]
        cvfilenames = ['bidirE2Ip%s'%(x[1]) for x in enumerate(alphas)]
        figFolder = '/homecentral/srao/cuda/data/pub/figs/bidir/e2i/'
        filetag = 'E2I'        
        maxLag = 50
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
#    figFolder = './data/figs/publication_figures/'
    
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
        nAC = np.load(datafolder + 'long_tau_vs_ac_mat_' + nFileName + '.npy')
        cv = np.load(datafolder + 'coefficientOFVar_' + cvfilenames[n] + '.npy')
#        cv = np.load('coefficientOFVarbidirI2Ip8.npy')
        cv1e = cv[:NE, 0]
        cv2e = cv[:NE, 1]
        cv1i = cv[NE:, 0]
        cv2i = cv[NE:, 1]        
        cv1e = cv1e[~np.logical_or(np.isnan(cv1e), np.isinf(cv1e))]
        cv2e = cv2e[~np.logical_or(np.isnan(cv2e), np.isinf(cv2e))]
        cv1i = cv1i[~np.logical_or(np.isnan(cv1i), np.isinf(cv1i))]
        cv2i = cv2i[~np.logical_or(np.isnan(cv2i), np.isinf(cv2i))]        
        axinset1.hist(cv1e, 200, normed = 1, histtype = 'step', linewidth = 0.5)
        axinset2.hist(cv2e, 200, normed = 1, histtype = 'step', linewidth = 0.5)
        axinset1_I.hist(cv1i, 200, normed = 1, histtype = 'step', linewidth = 0.5)
        axinset2_I.hist(cv2i, 200, normed = 1, histtype = 'step', linewidth = 0.5)        
        plotac(nAC[:, 0], axE, legendLabels[n], 'Population averaged AC, E', False)
        plotac(nAC[:, 1], axI, legendLabels[n], 'Population averaged AC, I', True)
    axE.set_xlim(0, maxLag)
    axI.set_xlim(0, maxLag)
    axE.set_xticks(np.arange(0, maxLag+1, maxLag / 2.))
    axI.set_xticks(np.arange(0, maxLag+1, maxLag / 2.))        
    plt.figure(figE.number)
#    plt.legend(loc=0)
    plt.figure(figI.number)
 #   axI.legend(loc=0, frameon=False, ncol = 2)
#    axI.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False, numpoints = 1)
    if bidirType == 'e2e':
        print '',
        axI.set_ylim(0, 23.5)
        axE.set_yticks(np.arange(0, 11, 5))
        axI.set_yticks(np.arange(0, 25, 12))
        axinset1.set_xlim(0, 2)
        axinset2.set_xlim(0, 2)
        axinset1.set_yticks([0, 6])
        axinset2.set_yticks([0, 6])        
        axinset1_I.set_xlim(0, 2)
        axinset1_I.set_xlim(0, 2)
        axinset1_I.set_yticks([0, 9])
        axinset2_I.set_yticks([0, 6])                
        plt.draw()
    elif bidirType == 'i2i':
        axE.set_yticks(np.arange(0, 25, 6))
#        axI.set_yticks(np.arange(0, 120, 10))
        axE.set_ylim(0, 24)
#        axI.set_yticks(np.arange(0, 91, 30))
        axI.set_yticks(np.arange(0, 121, 40))         

        print '',
    elif bidirType == 'e2i':
        axE.set_ylim(0, 10)
        axE.set_yticks(np.arange(0, 11, 5))
        axI.set_yticks(np.arange(0, 25, 12))        

    axinset1.set_xticks(np.arange(0, axinset1.get_xlim()[1] + 0.1, 1.0))
    axinset1.set_yticks(np.arange(0, axinset1.get_ylim()[1], 4.0))
    axinset1.set_ylabel('')
#    axinset1.set_xlabel('CV')
  #  axinset1.set_title('CV')    
 #   axinset1.grid('on')
#    axinset1.set_ylabel('Normalized count')
    axinset2.set_xticks(np.arange(0, axinset2.get_xlim()[1] + 0.1, 1.0))
    axinset2.set_yticks(np.arange(0, axinset2.get_ylim()[1], 4.0))        
    axinset2.set_ylabel('')
   # axinset2.set_title('CV2')    
#    axinset2.grid('on')
 #   axinset2.set_ylabel('Normalized count')
    
    axinset1_I.set_xticks(np.arange(0, axinset1_I.get_xlim()[1] + 0.1, 1.0))
    axinset1_I.set_yticks(np.arange(0, axinset1_I.get_ylim()[1], 4.0))
    axinset1_I.set_ylabel('')
   # axinset1_I.set_title('CV')    
  #  axinset1_I.grid('on')
  #  axinset1_I.set_ylabel('Normalized count')    
    axinset2_I.set_xticks(np.arange(0, axinset2_I.get_xlim()[1] + 0.1, 1.0))
    axinset2_I.set_yticks(np.arange(0, axinset2_I.get_ylim()[1], 4.0))        
    axinset2_I.set_ylabel('')
 #   axinset2_I.set_title('CV2')    
#    axinset2_I.grid('on')
   # axinset2_I.set_ylabel('Normalized count')

    axinset1.set_yticks([0, 8])
    axinset2.set_yticks([0, 8])
    axinset1.set_xlim(0, 2)
    axinset2.set_xlim(0, 2)    
    axinset1_I.set_xlim(0, 2)
    axinset2_I.set_xlim(0, 2)
    axinset1_I.set_yticks([0, 10])
    axinset2_I.set_yticks([0, 8])                
    if bidirType == 'e2i':
        axinset2_I.set_yticks([0, 10])                

    axPosition = [0.1745, 0.26, .76, .6 * 0.961]
    if bidirType == 'e2e':
        axinset1.set_position([.29, .25, .15*1.6, .15])
        axinset2.set_position([.64, .25, .15*1.6, .15])
        axinset1_I.set_position([.29, .25, .15*1.6, .15]);
        axinset2_I.set_position([.64, .25, .15*1.6, .15]);

    if bidirType =='i2i':
        axPosition = [0.205, 0.26, .7, .6 * 0.961]
#       axI.set_yticks(np.arange(0, 121, 40))
#       paperSize = [1.71*1.65, 1.21*1.65]
    paperSizeInsets = [.8, 0.6]
    axPositionInset = [0.15, 0.28, .745, .6]

    paperSize = [2.0, 1.5]
    axPosition = [0.26, 0.25, .65, 0.65]

    
    filename = 'AC_E_bidir' + filetag
    Print2Pdf(figE,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    filename = 'AC_I_bidir' + filetag
    Print2Pdf(figI,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition) #[0.142, 0.15, .77, .74])
    filename = 'CV_E_bidir' + filetag
    Print2Pdf(figcvE,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 10, tickFontsize=6, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPositionInset) #[0.142, 0.15, .77, .74])
    filename = 'CV2_E_bidir' + filetag
    Print2Pdf(figcv2E,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 10, tickFontsize=6, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPositionInset) #[0.142, 0.15, .77, .74])
    filename = 'CV_I_bidir' + filetag
    Print2Pdf(figcvI,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 10, tickFontsize=6, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition =  [0.18, 0.28, .745, .6]) #[0.142, 0.15, .77, .74])
    filename = 'CV2_I_bidir' + filetag
    if bidirType == 'e2i':
        Print2Pdf(figcv2I,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 10, tickFontsize=6, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = [0.18, 0.28, .745, .6])
    else:
        Print2Pdf(figcv2I,  figFolder + filename,  paperSizeInsets, figFormat=figFormat, labelFontsize = 10, tickFontsize=6, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPositionInset) #[0.142, 0.15, .77, .74])



    # plt.ion()
    # plt.show()
    # kb.keyboard()

  

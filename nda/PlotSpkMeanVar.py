basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf


def PlotSpkMeanVarScatter(ax, spkCntMean, spkCntVar, NE, IF_YLABEL):
    #    fig, ax = plt.subplots()
    #    figI, axI = plt.subplots()
    ax.plot(spkCntMean[NE:], spkCntVar[NE:], 'r.', markersize = 1.0, label = 'I')
    ax.plot(spkCntMean[:NE], spkCntVar[:NE], 'k.', markersize = 1.0, label = 'E')


    ax.set_xlabel('E[Spike count]')
    xmax = np.max(ax.get_xlim())
    ymax = np.max(ax.get_xlim())
    xymax = np.max([xmax, ymax])
    ax.plot(range(int(xymax - 50)), range(int(xymax-50)), 'g', label = 'x = y')
    if IF_YLABEL:
        ax.set_ylabel('Var[Spike count]')
        ax.legend(loc = 2, frameon = False, numpoints = 1, markerscale = 5.0)
    ax.set_xlim((0, xymax))
    ax.set_ylim((0, xymax))
    ax.set_xticks(np.arange(0, xymax+1, 50))
    ax.set_yticks(np.arange(0, xymax+1, 50))    
#    ax.set_aspect('equal')

if __name__ == '__main__':
    NE = 20000
    p = [0, 2, 5]
    figFormat = 'png'
    ax0 = plt.subplot(1, 3, 1)
    ax1 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    ax = [ax0, ax1, ax3]
    ptitle = ['Control', 'p = 0.2', 'p = 0.5']
    for k, kp in enumerate(p):
 #       fg, ax = plt.subplots()
        if k == 0:
            IF_YLABEL = True
        else:
            IF_YLABEL = False
        sc = np.load('./spkCnt_mean_var_p%s.npy'%(kp))
        PlotSpkMeanVarScatter(ax[k], sc[0, :], sc[1, :], NE, IF_YLABEL)
        ax[k].set_title(ptitle[k])
        axHandle = ax[k]
        axHandle.spines['top'].set_visible(False) 
        axHandle.spines['right'].set_visible(False)
        axHandle.yaxis.set_ticks_position('left')
        axHandle.xaxis.set_ticks_position('bottom')        
        filename = 'spkCnt_var_p%s'%(kp)
#        Print2Pdf(fg, filename,  [4.6*0.9,  4.0*.9], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.15, .77, .74])
#
#
    filename = 'spkCnt_var_vs_p'
    Print2Pdf(plt.gcf(), filename,  [4.6*2.0,  3.75], figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.142, 0.175, .77, .74])
#    plt.show()    
 #   plt.close()
                     



                                                   

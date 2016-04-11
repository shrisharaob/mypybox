#basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
#sys.path.append(basefolder)
#import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#sys.path.append(basefolder + "/nda/spkStats")
#sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
#from reportfig import ReportFig

def FixThisAxis(axHandle, titleSize, labelFontsize, tickFontsize, axPosition, IF_ADJUST_POSITION):
    axHandle.set_title(axHandle.get_title(), fontsize = titleSize); 
    axHandle.set_xlabel(axHandle.get_xlabel(), fontsize = labelFontsize);
    axHandle.set_ylabel(axHandle.get_ylabel(), fontsize = labelFontsize);
    yed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.yaxis.get_major_ticks()]
    xed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.xaxis.get_major_ticks()]
    #-- REMOVE right and top enclosing lines
    axHandle.spines['top'].set_visible(False) 
#    axHandle.spines['right'].set_visible(False)
    axHandle.spines['bottom'].set_linewidth(0.5)
    axHandle.spines['left'].set_linewidth(0.5)
    axHandle.spines['right'].set_linewidth(0.5)    
    axHandle.yaxis.set_ticks_position('left')
    axHandle.xaxis.set_ticks_position('bottom')
    axHandle.xaxis.set_tick_params(width = 0.5, direction = 'out')
    axHandle.yaxis.set_tick_params(width = 0.5, direction = 'out')
    axHandle.xaxis.set_tick_params(which = 'both', direction = 'out')
    axHandle.yaxis.set_tick_params(which = 'both', direction = 'out')    
    if(IF_ADJUST_POSITION):
         axHandle.set_position(axPosition) #[0.15, 0.15, .8, .75]

    

def Print2Pdf2Axes(figHandle, figname, paperSize = [5.26, 4.26], figFormat = 'pdf', labelFontsize = 12.0, tickFontsize = 12.0, titleSize = 12.0, IF_ADJUST_POSITION = False, axPosition = [0.125,  0.1, 0.775  ,  0.8]):
    figHandle.set_figwidth(paperSize[0])
    figHandle.set_figheight(paperSize[1])
    FixThisAxis(figHandle.get_axes()[0], titleSize, labelFontsize, tickFontsize, axPosition, IF_ADJUST_POSITION)
    FixThisAxis(figHandle.get_axes()[1], titleSize, labelFontsize, tickFontsize, axPosition, IF_ADJUST_POSITION)    
    figHandle.canvas.draw()
    figHandle.savefig(figname + '.' + figFormat, format=figFormat)
    
#     axHandle.set_title(axHandle.get_title(), fontsize = titleSize); 
#     axHandle.set_xlabel(axHandle.get_xlabel(), fontsize = labelFontsize);
#     axHandle.set_ylabel(axHandle.get_ylabel(), fontsize = labelFontsize);
#     yed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.yaxis.get_major_ticks()]
#     xed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.xaxis.get_major_ticks()]
#     #-- REMOVE right and top enclosing lines
#     axHandle.spines['top'].set_visible(False) 
#     axHandle.spines['right'].set_visible(False)
#     axHandle.spines['bottom'].set_linewidth(0.5)
#     axHandle.spines['left'].set_linewidth(0.5)
#     axHandle.yaxis.set_ticks_position('left')
#     axHandle.xaxis.set_ticks_position('bottom')
#     axHandle.xaxis.set_tick_params(width = 0.5, direction = 'out')
#     axHandle.yaxis.set_tick_params(width = 0.5, direction = 'out')
#     axHandle.xaxis.set_tick_params(which = 'both', direction = 'out')
#     axHandle.yaxis.set_tick_params(which = 'both', direction = 'out')    
#     # axHandle.xaxis.set_minortick_params(width = 0.5, direction = 'out')
#     # axHandle.yaxis.set_minortick_params(width = 0.5, direction = 'out')    
#     if(IF_ADJUST_POSITION):
# #        plt.gca().set_position(axPosition) #[0.15, 0.15, .8, .75]
#          axHandle.set_position(axPosition) #[0.15, 0.15, .8, .75]
    

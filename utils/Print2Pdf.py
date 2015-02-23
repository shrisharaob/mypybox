basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig


def Print2Pdf(axHandle, figname, paperSize = [4.26, 3.26], figFormat = 'pdf', labelFontsize = 20.0, tickFontsize = 14.0):
#    [axHandle] = DefaultArgs(
    plt.rcParams['figure.figsize'] = paperSize[0], paperSize[1]
    plt.rcParams['axes.labelsize'] = labelFontsize
    yed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.yaxis.get_major_ticks()]
    xed = [tick.label.set_fontsize(tickFontsize) for tick in axHandle.xaxis.get_major_ticks()]
    plt.savefig(figname + '.' + figFormat, format=figFormat)
    

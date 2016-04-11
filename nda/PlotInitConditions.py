basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
#import Keyboard as kb



def genfrplot(file1, file2, titletxt, figname, axPosition = [0.28, 0.28, .6, 0.6], paperSize = [2, 1.5], ne = 20000):
    print file1, file2, titletxt, figname, ne
    fr0 = np.loadtxt(file1)
    fr1 = np.loadtxt(file2)
    plt.plot(fr0[ne:], fr1[ne:], 'r.', markersize = 0.5)
    plt.plot(fr0[:ne], fr1[:ne], 'k.', markersize = 0.5)
    plt.xlabel('Rates(Hz)')
    plt.ylabel('Rates(Hz)')
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.xticks(np.arange(0, 201, 100))
    plt.yticks(np.arange(0, 201, 100))    
    plt.plot(range(200), range(200), 'g', linewidth = 0.5)
    plt.title(titletxt)
    plt.legend(['I', 'E'],frameon = False, numpoints = 1, markerscale = 10, loc = 4, prop = {'size':10})
    figFormat = 'png'
#   plt.gca().set_aspect('equal')
    plt.draw()
    Print2Pdf(plt.gcf(),  figname,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
#    plt.show()



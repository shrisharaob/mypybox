import numpy as np
import pylab as plt
import sys
def SetNUinqueColors(axHdl, n = 7):
    colormap = plt.cm.gist_rainbow
    colors = [colormap(i) for i in np.linspace(0, 0.9, int(n))]
    axHdl.set_color_cycle(colors)

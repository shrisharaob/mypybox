import numpy as np
import pylab as plt
from DefaultArgs import DefaultArgs

def YYPlot(x1, y1, y2, markerType, IF_FIX_YLIM):
    fig, ax1 = plt.subplots()
    ax1.plot(x1, y1, 'k', marker = markerType, linewidth = 0.4, markersize = 1.5, markeredgecolor = 'k')
    if IF_FIX_YLIM:
        ymax1 = np.max(ax1.get_ylim())
        ax1.set_ylim([-2, ymax1])
        ax1.set_yticks([0, 0.5 * ymax1, ymax1])
    # Make the y-axis label and tick labels match the line color.
    # for tl in ax1.get_yticklabels():
    #     tl.set_color('k')
    #     ax2 = ax1.twinx()
    #     ax2.plot(x1, y2, 'r', marker = markerType, linewidth = 0.4, markersize = 1.5, markeredgecolor = 'r')
    #     if IF_FIX_YLIM:
    #         ymax2 = np.max(ax2.get_ylim())
    #         ax2.set_ylim([-200, ymax2])
    #         ax2.set_yticks([0, 0.5 * ymax2, ymax2])            
    #     for tl in ax2.get_yticklabels():
    #         tl.set_color('r')
    ax2 = np.nan
    return fig, ax1, ax2

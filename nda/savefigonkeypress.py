basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
from DefaultArgs import DefaultArgs
from reportfig import ReportFig

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(np.random.rand(10), 'ko-')
img = np.random.rand(10000)
ax.imshow(img.reshape((100, 100)))
txt = None
figbase = '/homecentral/srao/Documents/code/tmp/figs/'
figname = 'tst0'


def onclick(event):
    print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)


# def onclick(event):
#     global txt
#     txt = plt.text(event.xdata, event.ydata, 'TESTTEST', fontsize=8)
#     print "haha hehe"
#     fig.canvas.draw()

#def offclick(event):
#    txt.remove()
 #   fig.canvas.draw()

def keyp(event):
    if event.key == "f":
        print "saving figure as", figname, ' in', figbase
        Print2Pdf(plt.gca(), figbase + figname)

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', keyp)
#6fig.canvas.mpl_connect('button_release_event', offclick) 


plt.show()






# #plt.waitforbuttonpress()
# #for i in range(5):
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# fig.canvas.mpl_disconnect(cid)
# plt.show()

basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
import MySQLdb as mysql
sys.path.append(basefolder)
import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import GetPO



dbNames = []
for i in range(len(sys.argv[1:])):
    dbName = sys.argv[1+i]
    dbNames.append(dbName)
NE = 20000
NI = 20000
simDuration = 3000
fre = np.zeros(len(dbNames))
fri = np.zeros(len(dbNames))
rhos = [0.1, 0.5, 0.6, 0.7]
legendLabels = []
xticklabels = []
for i, irho in enumerate(rhos):
    xticklabels.append(r'$\rho = %s$'%(irho))
#    print xticklabels
#fn = 'firingrates_xi1.2_theta%s_0.00_3.0_cntrst100.0_%s_trwithbug.csv'%(int(0), int(simDuration))
for kk, kdb in enumerate(dbNames):
    fr = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_' + kdb + '.npy')
    fre[kk] = fr[:NE, :].mean()
    fri[kk] = fr[NE:, :].mean()
plt.ion()
figm, axm = plt.subplots()
xaxis = np.arange(1, len(dbNames) + 1, 1)
axm.plot(xaxis, fre, 'ko-', label = 'E')
axm.plot(xaxis, fri, 'ro-', label = 'I')
print fre
print fri
axm.set_xticks(xaxis)
axm.set_xticklabels(xticklabels)
plt.ylabel('Mean activity (Hz)')
#plt.title(r'$\rho = 0.1$, with background')
plt.draw()
plt.grid()
figName = 'rho_vs_meanactivity_a1_withbug_rho1'
Print2Pdf(figm,  figName, [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.15, 0.15, .78, .75])

plt.waitforbuttonpress()

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

#alphaBetas = [[0.5, 1], [1.0, 1.0], [1.5, 1.0],[2.0, 1.0], [4.0, 1.0], [8.0, 1.0]]
#alphaBetas = [[1.0, 1.0], [2.0, 1], [4.0, 1], [8.0, 1.0], [10.0, 1.0]]
alphaBetas = [[1.0, 0.5], [1.0, 1.0], [1.0, 2]]
#alphaBetas = [[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]
#alphaBetas = [[1.0, 1.0], [1.0, 2]] 
legendLabels = []
xticklabels = []
for i in range(len(alphaBetas)):
    legendLabels.append(r'$\alpha = %s, \beta = %s$'%(alphaBetas[i][0], alphaBetas[i][1]))
    xticklabels.append(r'$\alpha = %s$'%(alphaBetas[i][0]) + '\n' + r'$\beta = %s$'%(alphaBetas[i][1]))
#    print xticklabels
if(len(dbNames) != len(alphaBetas)):
    print 'len of dbnames and alphabetas do not match'
    sys.exit(0)
fn = 'firingrates_xi1.2_theta%s_0.00_3.0_cntrst100.0_%s_trwithbug.csv'%(int(0), int(simDuration))

for kk, kdb in enumerate(dbNames):
    fr = np.loadtxt('/homecentral/srao/Documents/code/cuda/cudanw/data/broadI/rho1/' + kdb + '/' + fn)

#    fn = 'firingrates_xi1.2_theta%s_0.00_3.0_cntrst100.0_3000_tr%s.csv'%(int(0.0), kdb)
#    fr = np.loadtxt('/homecentral/srao/Documents/code/cuda/cudanw/' + fn)
    fre[kk] = fr[:NE].mean()
    fri[kk] = fr[NE:].mean()

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
plt.title(r'$\rho = 0.1$, with background')
#plt.title(r'$\beta = 1.0, \; \rho = 0.1$')
#plt.title(r'$\alpha = 1.0, \; \rho = 0.1$')
plt.draw()
plt.grid()
figName = 'alpha_vs_meanactivity_a1_withbug_rho1'
Print2Pdf(figm,  figName, [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.15, 0.15, .78, .75])

plt.waitforbuttonpress()

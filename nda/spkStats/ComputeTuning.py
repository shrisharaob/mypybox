# this is for computing tuning curves directiely from the firing rate file

basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf

rho = '5'
#folderbase = '/homecentral/srao/cuda/data/broadI/xi8em1/rho' + rho + '/'
#folderbase = '/homecentral/srao/cuda/data/poster/bidir/i2i/'
#folderbase = '/homecentral/srao/cuda/data/poster/bidir/e2e/'
folderbase = '/homecentral/srao/cuda/data/poster/'
#folderbase = '/homecentral/srao/cuda/data/broadI/rho' + rho + '/'
[foldername, filetag, NE, NI, xi, nTheta, contrast, alpha, tau_syn, simDuration, nTrials] = DefaultArgs(sys.argv[1:], ['', '', 20000, 20000, '0.8', 8, 100.0, '0', 3.0, 100000, 1])
fldr = foldername
foldername = folderbase + foldername + '/'
NE = int(NE)
NI = int(NI)
nTheta = int(nTheta)
theta = np.arange(0, 180, 22.5)
tuningCurves = np.zeros((NE + NI, nTheta))
for kk, kTheta in enumerate(theta):
    fn = 'firingrates_xi%s_theta%s_0.%s0_%s.0_cntrst100.0_%s_tr0.csv'%(xi, int(kTheta), int(alpha), int(tau_syn), int(simDuration))
    print foldername + fn
    tuningCurves[:, kk] = np.loadtxt(foldername + fn)
plt.ion()
#np.save(basefolder + '/db/data/tuningCurves_bidirI2I_p%s'%(int(alpha)) + fldr + 'rho' + rho + '_xi'+xi, tuningCurves)
#np.save(basefolder + '/db/data/tuningCurves_bidirE2E_' + fldr + 'rho' + rho + '_xi'+xi, tuningCurves)
#np.save(basefolder + '/db/data/tuningCurves_' + fldr + 'rho' + rho + '_xi'+xi + '_tau'+tau_syn, tuningCurves)

#np.save(basefolder + '/db/data/tuningCurves_' + filetag + 'rho' + rho + '_xi'+xi + '_tau'+tau_syn, tuningCurves)

np.save(basefolder + '/db/data/tuningCurves_' + filetag + 'rho' + rho + '_xi'+xi, tuningCurves)



# for kk in np.arange(20):
#     idx = np.random.randint(0, 40000, 1)
#     plt.plot(theta, tuningCurves[idx[0], :], 'ko-')
#     plt.title('%s'%(idx[0]))
#     plt.waitforbuttonpress()
#     plt.clf()
    
                                                                                               

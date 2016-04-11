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
folderbase = '/homecentral/srao/cuda/data/inputTuning/'
#folderbase = '/homecentral/srao/cuda/data/broadI/rho' + rho + '/'
[foldername, alpha, NE, NI, xi, nTheta, contrast,  tau_syn, simDuration, nTrials] = DefaultArgs(sys.argv[1:], ['', 0, 200,  0, '0.8', 8, 100.0, 3.0, 10000, 1])
fldr = foldername
foldername = folderbase + foldername + '/'
NE = int(NE)
NI = int(NI)
nTheta = int(nTheta)
theta = np.arange(0, 180, 22.5)
tuningCurvesE = np.zeros((NE + NI, nTheta))
tuningCurvesI = np.zeros((NE + NI, nTheta))
for kk, kTheta in enumerate(theta):
    fn = 'avgNaBlockedECur_xi%s_theta%s_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(xi, int(kTheta), int(alpha), int(simDuration))
    print foldername + fn
    ctc = np.loadtxt(foldername + fn, delimiter = ';')
    tuningCurvesE[:, kk] = ctc[:, 0]
    tuningCurvesI[:, kk] = ctc[:, 1]
    
plt.ion()
np.save(basefolder + '/db/data/cur_tuningCurves_bidirI2I_p%s'%(int(alpha)) + fldr + 'rho' + rho + '_xi'+xi, np.array([tuningCurvesE, tuningCurvesI]))


#np.save(basefolder + '/db/data/cur_tuningCurves_bidirE2E_' + fldr + 'rho' + rho + '_xi'+xi, tuningCurves)
#np.save(basefolder + '/db/data/cur_tuningCurves_' + fldr + 'rho' + rho + '_xi'+xi, tuningCurves)



for kk in np.arange(20):
    idx = np.random.randint(0, NE, 1)
    plt.plot(theta, tuningCurvesE[idx[0], :], 'ko-')
    plt.plot(theta, tuningCurvesI[idx[0], :], 'ro-')
    plt.plot(theta, tuningCurvesE[idx[0], :] + tuningCurvesI[idx[0], :], 'go-')        
    plt.title('%s'%(idx[0]))
    plt.waitforbuttonpress()
    plt.clf()
    
                                                                                               

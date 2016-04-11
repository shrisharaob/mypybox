import numpy as np
import code
import sys
import pylab as plt
from multiprocessing import Pool
from functools import partial
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf

sys.path.append("/homecentral/srao/Documents/code/mypybox")
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")

def CV(spkTimes):
    out = np.array([np.nan, np.nan])
    if(spkTimes.size > 3):
        isi = np.diff(spkTimes)
        cv = isi.std() / isi.mean()
        cv2 = CV2(isi)
        out=np.array([cv, cv2])
    return out

def CV2(isi):
    denom = isi[1:] + isi[:-1]
    cv2 = 2.0 * np.abs(np.diff(isi)) / denom
    return np.nanmean(cv2)

def CVofList(st, listOfneurons):
    out = np.empty((len(listOfneurons), 2))
    out[:] = np.nan
    for i, iNeuron in enumerate(listOfneurons):
        spkTimes = starray[starray[:, 1] == iNeuron, 0]
        spksTimes = spkTimes[spkTimes > spkTimeStart]
        out[i, :] = CV(spkTimes)
#    st = np.cumsum(np.random.exponential(1/10., size=(200000, ))) # poission spike train
#    out = CV(st)
    return out

if __name__ == '__main__':
    [bidirType, alpha, NE, NI, xi, nTheta, contrast, tau_syn, simDuration, nTrials, dt] = DefaultArgs(sys.argv[1:], ['', '', 20000, 20000, '0.8', 8, 100.0, 3.0, 100000, 1, 0.05])
    print 'loading st from file ...',
    sys.stdout.flush()
    if bidirType == 'i2i':
        foldername = 'i2i/p%s'%((alpha, ))
        filetag = 'I2I'
    elif bidirType == 'e2i':
        foldername = 'e2i/p%s'%((alpha, ))
        filetag = 'E2I'
    elif bidirType == 'e2e':
        foldername = 'e2e/p%s'%((alpha, ))
        filetag = 'E2E'
    datafolder = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/'
    starray = np.loadtxt(datafolder + foldername +'/spkTimes_xi0.8_theta0_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(int(alpha), simDuration), delimiter = ';');
  #  starray = np.loadtxt('spkTimes.csv', delimiter = ';')
    print 'done!'
    maxLag = 1000
    bins = np.arange(-500, 500, 1)
#    filetag =  ''#'bidirI2I_p%s'%(int(alpha))    
    spkTimeStart = 2000.0
    spkTimeEnd = 100000.0
    listOfneurons = np.arange(NE+NI)
    print 'computing cv ...',
    sys.stdout.flush()    
    cv = CVofList(starray, listOfneurons)
    print 'done'
    np.save('./data/coefficientOFVar_bidir_' + filetag + '_p%s'%(alpha, ), cv)
    

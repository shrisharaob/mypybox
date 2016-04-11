basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

#bf = '/homecentral/srao/cuda/data/vmtraces/dt5em2/'
bf = '/homecentral/srao/cuda/data/pub/bidir/'
tau = int(sys.argv[1])
p = int(sys.argv[2])

nTrials = 5
nNeruons = 10
dt = 0.05
paperSize = [10.2, 1.6] #[4.6*2,  3.6]
figFormat = 'eps'
plt.ioff()
tAxis = np.arange(0, 5000, dt)
for kTrial in range(nTrials):
    if tau == 3:
        foldername = bf + 'p%s/'%(p)
        
        frFilename = 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_6000_tr%s.csv'%(p, tau, kTrial)
    else:
        foldername = bf + 'p%s/tau%s/'%(p, tau)        
        frFilename = 'firingrates_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_6000_tr%s.csv'%(p, tau, kTrial)
    fr = np.loadtxt(foldername + frFilename)
#    vmFilename = 'vm_xi0.8_theta0_0.%s0_%s.0_6000_tr%s.csv'%(p, tau, kTrial)
    vmFilename = 'vm_xi0.8_theta0_0.00_%s.0_12000_trvmtr.csv'%(p, tau)
    vm = np.loadtxt(foldername + vmFilename)
    for mNeuron in range(nNeruons):
        plt.plot(tAxis, vm[:, mNeuron + 1], 'k', linewidth = 0.2)
        filename = foldername + 'vm_trace_tr%s_NEURON%s'%(kTrial, mNeuron)        
        plt.ylim(-80, 60)
        plt.yticks([-80, 0, 60])
        plt.xlim(0, 5000)
        plt.xticks([0, 2500, 5000])
        Print2Pdf(plt.gcf(),  filename, paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.12, 0.25, .8, .5])        
#        plt.waitforbuttonpress()
        plt.clf()

    
        




 


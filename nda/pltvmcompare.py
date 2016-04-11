basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb

bf = '/homecentral/srao/cuda/data/poster/'
fr = np.loadtxt(bf + 'cntrl/firingrates_xi0.8_theta0_0.00_3.0_cntrst100.0_6000_tr_vm.csv')
fr1 = np.loadtxt(bf + 'bidir/i2i/p8/firingrates_xi0.8_theta0_0.80_3.0_cntrst100.0_6000_tr_vm.csv')
vm = np.loadtxt(bf + 'cntrl/vm_xi0.8_theta0_0.00_3.0_6000_tr_vm.csv')
vm1 = np.loadtxt(bf + 'bidir/i2i/p8/vm_xi0.8_theta0_0.80_3.0_6000_tr_vm.csv')
fg, ax = plt.subplots()
fg1, ax1 = plt.subplots()

idx=fr[:20].argsort()
idx1=fr1[:20].argsort()
plt.ion()
figFormat = 'eps'
for i in range(20):
    ax.plot(vm[:, 0], vm[:, idx[i] + 1], 'k', linewidth = 0.5, label = 'control(%.4sHz)'%(fr[idx[i]]))
    ax1.plot(vm1[:, 0], vm1[:, idx1[i] + 1], 'k',linewidth = 0.5, label = 'p = 0.8(%.4sHz)'%(fr1[idx1[i]]))
    ax.set_yticks(np.arange(-80, 60, 50.))
    ax.set_yticks(np.arange(-80, 60, 50.))    
    ax.set_xlabel('Time(ms)')
    ax.set_ylabel('mv')

    ax1.set_yticks(np.arange(-80, 60, 50.))
    ax1.set_yticks(np.arange(-80, 60, 50.))    
    ax1.set_xlabel('Time(ms)')
    ax1.set_ylabel('mv')
    
#    ax1.plot(vm1[:, 0], vm1[:, idx1[i] + 1], 'k', linewidth = 0.5)
  #  ax.set_title('Neuron id: %s'%(i))
#    ax.set_xticklabels(['' for x in range(len(ax.get_xticks()))])
#    ax1.set_yticks(np.arange(-80, 60, 50.))    
 #   ax1.set_title('p = 0.8, fr = %.4sHz'%(fr1[idx1[i]]))
#    ax.set_title('control, fr = %.4sHz'%(fr[idx[i]]))
#    ax1.set_ylabel('Voltage(mv)')
  #  axHandle = ax1
    # axHandle.spines['top'].set_visible(False) 
    # axHandle.spines['right'].set_visible(False)
    # axHandle.yaxis.set_ticks_position('left')
    # axHandle.xaxis.set_ticks_position('bottom')
#    ax.legend(bbox_to_anchor=(-0., 1.02, 1.0, .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon=False, numpoints = 1, markerscale = 10.0)

    filename = './vmtrace/vm_trace_cntrl_bidirI2I_%s'%(i)
    filename1 = './vmtrace/vm_trace_p8_bidirI2I_%s'%(i)    
    paperSize = [10.2, 1.6] #[4.6*2,  3.6]

    ax.legend(bbox_to_anchor=(-0., 0.9, 1.0, .102), loc=3, ncol=2, borderaxespad=0., frameon=False, numpoints = 1, markerscale = 100.0)
    ax1.legend(bbox_to_anchor=(-0., 0.9, 1.0, .102), loc=3, ncol=2, borderaxespad=0., frameon=False, numpoints = 1, markerscale = 100.0)        
    Print2Pdf(fg,  filename, paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.12, 0.25, .8, .5])
    Print2Pdf(fg1,  filename1, paperSize, figFormat=figFormat, labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.12, 0.25, .8, .5])
    plt.close('all')
    fg, ax = plt.subplots()
    fg1, ax1 = plt.subplots()

 


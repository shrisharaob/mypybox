from Startup import *
import LoadSimResults as sim

fbGpu = '/home/dhansel/cuda/cudanw/'
fbCpu = '/home/dhansel/Documents/cnrs/simResults/'

def CompareVm(NE, NI):
    nNeurons = NE + NI
    vmGPU = sim.loadvm(nNeurons + 1) # +1if first column is the time vector
    vmCPU = np.fromfile(fbCpu + 'vm.csv', dtype = float, sep = ' ')
    vmCPU = np.reshape(vmCPU, (len(vmCPU) / (nNeurons + 1), nNeurons + 1))
    plt.ion()
    plt.figure()
    plt.plot(vmCPU[:, 1 : nNeurons + 1] - vmGPU[:, 1 : nNeurons + 1])
    plt.waitforbuttonpress()
    print np.sum(abs(vmCPU[:, 1 : nNeurons + 1] - vmGPU[:, 1 : nNeurons + 1]))
                     
def CompareSpks():
    stGPU = sim.loadSpks()
    stCPU = np.loadtxt(fbCpu + 'spkTimes_theta000.csv', delimiter = ';')
    nSpks, tmp = stGPU.shape
    stCPU = np.reshape(stCPU, (nSpks, 2))
    plt.ion()
    plt.figure()
    plt.hist(stGPU[:, 0] - stCPU[:, 0], 100)
    stCPUidx = stCPU[:, 1] - 1
    print np.sum(stGPU[:, 1] - stCPUidx)
    plt.waitforbuttonpress()
if __name__ == "__main__":
    CompareVm(2, 2)

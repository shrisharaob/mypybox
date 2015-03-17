from Startup import *
fb = '/homecentral/srao/Documents/code/cuda/cudanw/'
print fb
def loadSpks():
#    st = np.loadtxt(fb + 'spkTimes.csv', delimiter = ';')
    st = np.loadtxt(fb + 'spkTimes.csv', delimiter = ';')
    st.shape = -1, 2
    return st

def loadCur():
    cur = np.loadtxt(fb + 'currents.csv', delimiter = ';')
    return cur

def loadvm(nNeurons):
#    vm = np.loadtxt(fb + 'vm.csv')
    print "nNeurons = ", nNeurons
    vm = np.fromfile(fb + 'vm.csv', dtype = float, sep = ' ')
    return  np.reshape(vm, (len(vm) / nNeurons, nNeurons))

import numpy as np
import pylab as plt
import sys
#fldr = "~/cuda/cudanw/"
#st = np.loadtxt("/home/dhansel//cuda/cudanw/spkTimes.csv")
def PlotRaster(st):
    plt.ion()
    plt.plot(st[:, 0], st[:, 1], '.k')
    plt.draw()

def RasterPlot(st, spkStart = 0, spkStop = -1, ):
    st = SpikesInInterval(st, spkStart, spkStop)
    print st.shape
    neuronIdx = np.unique(st[:, 1])
    nNeurons = np.size(neuronIdx)
    nSpks, _ = st.shape
    vLenght = 0.5
    plt.ion()
    x = np.array([])
    y = np.array([])
    for idx, iNeuron in enumerate(neuronIdx):
        iSpkTimes = st[st[:, 1] == iNeuron, 0]
        x = np.r_[x, iSpkTimes]
        y = np.r_[y, iNeuron * np.ones((np.size(iSpkTimes),))]
    plt.vlines(x, y, y + vLenght)
    plt.draw()
    plt.show()

def SpikesInInterval(st, spkStart, spkEnd = -1):
    if(spkEnd == -1):
        spkEnd = np.max(st[:, 0])
        idx = st[:, 0] >= spkStart
    else:
        idx = np.logical_and(st[:, 0] >= spkStart, st[:, 0] <= spkEnd)
    return st[idx, :]

# def SpikesOfNeurons(st, neuronsList):
#     rows, clmns = st.shape
#     idx = np.zeros((rows, ))n
#     for k, kneuron in enumerate(neuronList):
#         if(st[
#         idx[k] = 
#     idx = (st[:, 1] == 
    

if __name__ == "__main__":
    st = np.loadtxt(sys.argv[1], delimiter = ';')
    RasterPlot(st)
    plt.waitforbuttonpress()
        
    
                  
        

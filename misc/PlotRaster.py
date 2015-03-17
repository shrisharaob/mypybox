import numpy as np
import pylab as plt
import sys
#fldr = "~/cuda/cudanw/"
#st = np.loadtxt("/home/dhansel//cuda/cudanw/spkTimes.csv")
def PlotRaster(st):
    plt.ion()
    plt.plot(st[:, 0], st[:, 1], 'xk')
    plt.draw()

def RasterPlot(st):
    neuronIdx = np.unique(st[:, 1])
    nNeurons = np.size(neuronIdx)
    nSpks, _ = st.shape
    vLenght = 0.1
    plt.ion()
    x = np.array([])
    y = np.array([])
    for idx, iNeuron in enumerate(neuronIdx):
        iSpkTimes = st[st[:, 1] == iNeuron, 0]
        x = np.r_[x, iSpkTimes]
        y = np.r_[y, iNeuron * np.ones((np.size(iSpkTimes),))]
    plt.vlines(x, y, y + vLenght)
    plt.draw()

if __name__ == "__main__":
    st = np.loadtxt(sys.argv[1], delimiter = ';')
    RasterPlot(st)
    plt.waitforbuttonpress()
        
    
                  
        

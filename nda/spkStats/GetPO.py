import numpy as np
from scipy.optimize import curve_fit

def GetPo(firingRate, thetas):
    a = 0.0 # real part
    b = 0.0 # imaginary part
    thetas = thetas * np.pi / 180.0
    for i in range(thetas.size):
        a += firingRate[i] * np.cos(2.0 * thetas[i])
        b += firingRate[i] * np.sin(2.0 * thetas[i])
    print a, b
    if(a > 0):
        out = 0.5 * np.arctan(b / a) 
    else:
        out = 0.5 * np.arctan(b / a) + np.pi
    print out
    return ((out * 180.) / np.pi) # because arctan principle domain is -90 to 90, returns in radians



def VonMises(theta, r0, r1, PO, D):
    return r0 + r1 * np.exp((np.cos(2.0 * (theta - PO)) - 1.0) / D)

#def VonMisesError

def FitVonMises(firingRate, theta):
    theta = theta * np.pi / 180.0 # to radians
    popt, pcov = curve_fit(VonMises, theta, firingRate) #, p0 = (0.1, 0.1, .1, .01))
    return popt, pcov


def GetF1(firingRate, thetas):
    thetas = thetas * np.pi / 180.0
    N = thetas.size
    h = 0.0 # real part
    k = 0.0 # imaginary part
    for i in range(N):
        h += firingRate[i] * np.cos(2.0 * np.pi * i  / float(N))
        k += firingRate[i] * np.sin(2.0 * np.pi * i  / float(N))

    print h, k
#    po = arctan(k/h)
    # if(h < 0.0):
    #     po = po + np.pi
#    print (po * 180.0 / np.pi)
    return h, k


def ComputePO(firingRate, thetas):
    a = 0.0 # real part
    b = 0.0 # imaginary part
    thetas = thetas * np.pi / 180.0
    for i in range(thetas.size):
        a += firingRate[i] * np.cos(2.0 * thetas[i])
        b += firingRate[i] * np.sin(2.0 * thetas[i])
#    print a, b
    # if(a > 0):
    #     out = 0.5 * np.arctan2(b, a) 
    # else:
    #     out = 0.5 * np.arctan2(b, a) + np.pi
    out = 0.5 * np.arctan2(b, a)
#    out = (out * 180.) / np.pi
    if(out < 0):
        out = out + np.pi
    return out # because arctan principle domain is -90 to 90, returns in radians
    
def POofPopulation(tc):
    # return value in radians
    theta = np.arange(0.0, 180.0, 22.5/2)
    nNeurons, _ = tc.shape
    po = np.zeros((nNeurons, ))
    for kNeuron in np.arange(nNeurons):
        po[kNeuron] = ComputePO(tc[kNeuron, :], theta)
    return po 

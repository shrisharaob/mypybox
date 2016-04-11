import numpy as np
import pylab as plt
import sys, os
sys.path.append("/homecentral/srao/Documents/code/mypybox")
import Keyboard as kb
from multiprocessing import Pool
from functools import partial 
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs


def SetupAndCompile():
    os.system("gcc -g -fPIC -lm -o correl.so -shared nrutil.c correl.c")
    #os.system("gcc -g -ggdb -o gensparsevec.so -shared GenSparseVec.c")
    mycfunc = np.ctypeslib.load_library('correl.so', '.') # use gcc -o gensparsevec.so -shared GenSparseVec.c
    mycfunc.correl.restype = None
    mycfunc.correl.argtypes = [np.ctypeslib.ndpointer(np.float32, flags = 'aligned, contiguous, writeable'),
                                     np.ctypeslib.ndpointer(np.float32, flags = 'aligned, contiguous, writeable'),
                                     np.ctypeslib.c_intp,
                                     np.ctypeslib.ndpointer(np.float32, flags = 'aligned, contiguous, writeable')]
    return mycfunc

def Call_C_CorrFunc(x):
    mycfunc = SetupAndCompile()
    requires = ['CONTIGUOUS', 'ALIGNED']
    x = np.require(x, np.float32, requires)
    nPoints = np.array(len(x) - 1, dtype = np.uint32)    
    acX = np.zeros((2 * nPoints + 2, ), dtype = np.float32)
    acX = np.require(acX, np.float32, requires)
    mycfunc.correl(x, x, nPoints, acX)
    return acX
    
if __name__ == "__main__":
    N = 2**12 + 1# = 4096
    print "N = ", N, "2*N = ", 2 * N
    x = np.sin(2.* 3.142 * 0.001 * np.arange(N))
    z = np.zeros((N, ))
    x = np.concatenate((x, z))
    ac = Call_C_CorrFunc(x)
    plt.plot(ac[1:])
    plt.show()

    


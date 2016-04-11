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
    os.system("gcc -g -fPIC -o ctypesTest0.so -shared ctypesTest0.c")
    #os.system("gcc -g -ggdb -o gensparsevec.so -shared GenSparseVec.c")
    mycfunc = np.ctypeslib.load_library('ctypesTest0.so', '.') # use gcc -o gensparsevec.so -shared GenSparseVec.c
    mycfunc.PrintReceivedInput.restype = None
    mycfunc.PrintReceivedInput.argtypes = [np.ctypeslib.ndpointer(np.int32, flags = 'aligned, contiguous, writeable'),
                                     np.ctypeslib.ndpointer(np.float32, flags = 'aligned, contiguous, writeable'),
                                     np.ctypeslib.c_intp]

    return mycfunc

def Call_C_CorrFunc():
    mycfunc = SetupAndCompile()
    requires = ['CONTIGUOUS', 'ALIGNED']

    a = np.array(101, dtype = np.int32)
    a = np.require(a, np.int32, requires)
    mycfunc.PrintReceivedInput(a, 0, 0)
    
    
if __name__ == "__main__":
    Call_C_CorrFunc()

    


    # b = np.array(np.pi, dtype = np.float32)
    # b = np.require(b, np.float32, requires)
    
    # c = np.array(101987, dtype = np.uint32)
    # c = np.require(c, np.uint32, requires)
    

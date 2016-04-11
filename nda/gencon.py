''' USAGE: [dbName, NetworkType, K, NE, NI, thetaSig] = DefaultArgs(sys.argv[1:], ['', 'ori', 1000, 10000, 10000, 0.5]) '''

import numpy as np
import code, sys, os
import pylab as plt

probFunc = lambda thetaDiff, sig: 1.0 / (sig * np.sqrt(2.0*np.pi)) * np.exp(-(np.sin((thetaDiff)))**2 / (2.0*sig**2))
def ConProbFunc(thetaDiff, sig):
    #thetaDiff in radians
    return probFunc(thetaDiff, sig)

os.system("gcc -fPIC -o gensparsevec.so -shared GenSparseVec.c") 
#os.system("gcc -g -ggdb -o gensparsevec.so -shared GenSparseVec.c")
mycfunc = np.ctypeslib.load_library('gensparsevec.so', '.') # use gcc -o gensparsevec.so -shared GenSparseVec.c
mycfunc.GenSparseMat.restype = None
mycfunc.GenSparseMat.argtypes = [np.ctypeslib.ndpointer(np.int32, flags = 'aligned, contiguous'),
                                 np.ctypeslib.c_intp,
                                 np.ctypeslib.c_intp,
                                 np.ctypeslib.ndpointer(np.int32, flags = 'aligned, contiguous, writeable'),
                                 np.ctypeslib.ndpointer(np.int32, flags = 'aligned, contiguous, writeable'),
                                 np.ctypeslib.ndpointer(np.int32, flags = 'aligned, contiguous, writeable')]

def GenSparseMat(convec, rows, clmns, sparseVec, idxvec, nPostNeurons):
    requires = ['CONTIGUOUS', 'ALIGNED']
    convec = np.require(convec, np.int32, requires)
    sparseVec = np.require(sparseVec, np.int32, requires)
    idxvec = np.require(idxvec, np.int32, requires)
    nPostNeurons = np.require(nPostNeurons, np.int32, requires);
    mycfunc.GenSparseMat(convec, rows, clmns, sparseVec, idxvec, nPostNeurons)
    fpsparsevec = open('sparseConVec.dat', 'wb')
    sparseVec.tofile(fpsparsevec)
    fpsparsevec.close()
    fpIdxVec = open('idxVec.dat', 'wb')
    idxvec.tofile(fpIdxVec)
    fpIdxVec.close()
    fpNpostNeurons = open('nPostNeurons.dat', 'wb')
    nPostNeurons.tofile(fpNpostNeurons)
    fpNpostNeurons.close()

if __name__ == '__main__':
    if(len(sys.argv) < 4):
        print "USAGE: python gencon.py K NE NI"
        sys.exit(0)
    K = int(sys.argv[1])
    NE = int(sys.argv[2])
    NI = int(sys.argv[3])
    cprob = np.zeros((NE + NI, NE + NI))
    cprob[:NE, :NE] = float(K) / NE # E --> E
    cprob[:NE, NE:] = float(K) / NE # E --> I
    cprob[NE:, :NE] = float(K) / NI # I --> E
    cprob[NE:, NE:] = float(K) / NI # I --> I
    print 'generating  connectivity matrix...',
    sys.stdout.flush()
    cprob = cprob > np.random.uniform(size = (cprob.shape))
    print 'done!'
    idxvec = np.zeros((NE + NI, ), dtype = np.int32)
    nPostNeurons = np.zeros((NE + NI, ), dtype = np.int32)
    sparseVec = np.zeros(shape = (cprob.sum(), ), dtype = np.int32)
    rows = NE + NI
    clmns = NE + NI
    print 'Generating sparse representation ...',
    sys.stdout.flush()
    cprob = cprob.transpose()
    GenSparseMat(cprob.astype(np.int32), rows, clmns, sparseVec, idxvec, nPostNeurons)
    print 'done!'
    print 'npost sum =', nPostNeurons.sum()
    print 'convec sum = ', cprob.sum()

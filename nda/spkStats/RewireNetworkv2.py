''' USAGE: [dbName, NetworkType, K, NE, NI, thetaSig] = DefaultArgs(sys.argv[1:], ['', 'ori', 1000, 10000, 10000, 0.5]) '''

basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from multiprocessing import Pool
from functools import partial 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import GetPO

# probFunc = lambda thetaDiff, sig: 1.0 / (sig * np.sqrt(2.0*np.pi)) * np.exp(-(np.sin((np.pi/180.0)*(thetaDiff)))**2 / (2.0*sig**2))
# out = probFunc(thetaDiff, sig)
# out[thetaDiff >= np.pi / 2.0] = (1.0/(sig * np.sqrt(2.0 * np.pi))) - out[thetaDiff >= np.pi/2.0] 
# return 1.0 + np.cos(2.0 * thetaDiff)
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
    # NetworkType : {'uni', 'ori'}, 'uni' is for standard random network, 'ori' is to rewire depending on the distance in ori space
    [dbName, NetworkType, K, NE, NI, thetaSig, thetaSigI] = DefaultArgs(sys.argv[1:], ['', 'oriE', 1000, 10000, 10000, .75, 0.75, ])
    NE = int(NE)
    NI = int(NI)
    K = int(K)
    thetaSig = float(thetaSig)
    thetaSigI = float(thetaSigI)
    cprob = np.zeros((NE + NI, NE + NI))
    print 'Network type: ', NetworkType
    print 'generating connection probabilities, NE, NI, K, thetaSig', NE, NI, K, thetaSig
    tuningCurves = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, )));
    po = GetPO.POofPopulation(tuningCurves)
    if NetworkType == 'uni':
        cprob[:NE, :NE] = float(K) / NE # E --> E
        cprob[:NE, NE:] = float(K) / NE # E --> I
        cprob[NE:, :NE] = float(K) / NI # I --> E
        cprob[NE:, NE:] = float(K) / NI # I --> I
    elif NetworkType == 'oriEEEI' or NetworkType == 'oriEEIE':
        print 'computing E --> E'
        for i in np.arange(NE):
            cprob[i, :NE] = ConProbFunc(np.abs(po[i] - po[:NE]), thetaSig)
        zb = float(K) / cprob[:NE, :NE].sum(0)
        cprob[:NE, :NE] *= zb # E --> E
        cprob[NE:, NE:] = float(K) / NI # I --> I
        if NetworkType == 'oriEEIE':
            print 'computing E --> I'
            for i in np.arange(NE):
                cprob[i, NE:] = ConProbFunc(np.abs(po[i] - po[NE:]), thetaSig)
            zb = float(K) / cprob[:NE, NE:]
            cprob[:NE, NE:] *= zb # E --> I
            cprob[NE:, :NE] = float(K) / NI # I --> E
        elif NetworkType == 'oriEEEI':
            print 'computing I --> E'
            for i in np.arange(NE, NE + NI):
                cprob[i, :NE] = ConProbFunc(np.abs(po[i] - po[:NE]), thetaSigI)
            zb = float(K) / cprob[NE:, :NE].sum(0)
            cprob[NE:, :NE] *= zb
            cprob[:NE, NE:] = float(K) / NE # E --> I
    elif NetworkType == 'oriII':
        print 'computing I --> I'
        for i in np.arange(NE, NE + NI):
            cprob[i, NE:] = ConProbFunc(np.abs(po[i] - po[NE:]), thetaSigI)
        zb = float(K) / cprob[NE:, NE:].sum(0)
        cprob[NE:, NE:] *= zb
        cprob[:NE, :NE] = float(K) / NE # E --> E
        cprob[:NE, NE:] = float(K) / NE # E --> I
        cprob[NE:, :NE] = float(K) / NI # I --> E
    elif NetworkType == 'oriEE':
        print 'computing E --> E'
        for i in np.arange(NE):
            cprob[i, :NE] = ConProbFunc(np.abs(po[i] - po[:NE]), thetaSig)
        zb = float(K) / cprob[:NE, :NE].sum(0)
        cprob[:NE, :NE] *= zb
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
#     plt.ion()
#     xxx = cprob[:NE, :NE]
#     plt.hist(xxx.sum(0), 100)
#     plt.waitforbuttonpress()
#     plt.clf()
# #    np.save('convec', cprob.astype(np.int32))
# #    convec[:NE, :NE] = cprob[:NE, :NE] > np.random.uniform(size = (cprob[:NE, :NE].shape))
    figFolder = figFolder = basefolder + '/nda/spkStats/figs/rewiring/'
    plotCon = 'I'
    for i in range(2):
        if(plotCon == 'E'):
            poe = po[:NE]
            idx = np.random.randint(0, NE, 1)[0]
            plt.hist(poe[cprob[idx, :NE]] * 180.0 / np.pi)
#            plt.hist(poe[cprob[idx, :NE]] * 180.0 / np.pi)
            plt.title("PO = %s"%(poe[idx] * 180.0 / np.pi))
            plt.waitforbuttonpress()
            Print2Pdf(plt.gcf(),  figFolder + 'rewired_input_ori_distr_fig%s'%(i), [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])
            plt.clf()
        elif(plotCon == 'I'):
            poi = po[NE:]
            idx = np.random.randint(NE, NE+NI, 1)[0]
            plt.hist(poi[cprob[idx, NE:]] * 180.0 / np.pi)
            plt.title("PO = %s"%(po[idx] * 180.0 / np.pi))
            plt.waitforbuttonpress()
            Print2Pdf(plt.gcf(),  figFolder + 'rewired_input_ori_distr_fig%s'%(i), [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])
            plt.clf()
    kb.keyboard()
    









#     if NetworkType == 'oriE':
#         tuningCurves = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, )));
#         po = GetPO.POofPopulation(tuningCurves)
#         print "po done "
#         poe = po[:NE]
#         # sigarray = np.array([0.0125, 0.0125/2, 0.0125/4, 0.0125/8])
#         print "computing prob"
# #        for i in np.arange(NE):
#         for i in np.arange(NE, NE+NI):
# #            cprob[i, :NE] = ConProbFunc(np.abs(poe[i] - poe[:]), thetaSig)
#             cprob[i, NE:] = ConProbFunc(np.abs(po[i] - po[NE:]), thetaSig)
#         print "computing prefactor"
#         #zb = K / cprob[:NE, :NE].sum(0)
#         #cprob[:NE, :NE] = cprob[:NE, :NE] * zb

#         zb = K / cprob[NE:, NE:].sum(0)
#         cprob[NE:, NE:] = cprob[NE:, NE:] * zb
        
#         cprob[:NE, NE:] = float(K) / NE     # E --> I
# #        cprob[NE:, NE:] = float(K) / NI     # I --> I
#         cprob[:NE, :NE] = float(K) / NE     # E --> E
#         cprob[:NE, NE:] = float(K) / NI # I --> E
#     elif NetworkType == 'oriEI': # EE & E-to-I
#         tuningCurves = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, )));
#         po = GetPO.POofPopulation(tuningCurves)
#         print "po done "
#         poe = po[:NE]
#         # sigarray = np.array([0.0125, 0.0125/2, 0.0125/4, 0.0125/8])
#         print "computing prob E --> E"
#         for i in np.arange(NE):
#             cprob[i, :NE] = ConProbFunc(np.abs(poe[i] - poe[:]), thetaSig) # E --> E
#         print "computing prefactor"
#         zb = K / cprob[:NE, :NE].sum(0)
#         cprob[:NE, :NE] = cprob[:NE, :NE] * zb
#         print "computing prob I --> E"
#         for i in np.arange(NE, NE+NI, 1):
#             cprob[i, :NE] = ConProbFunc(np.abs(po[i] - poe[:]), thetaSigI) # I --> E
#         print "computing prefactor"        
#         zb = K / cprob[NE:, :NE].sum(0)
#         cprob[NE:, :NE] = cprob[NE:, :NE] * zb
#         cprob[:NE, NE:] = float(K) / NE     # E --> I
#         cprob[NE:, NE:] = float(K) / NI     # I --> I
#     elif NetworkType == 'oriEEIE': # EE & I-to-E
#         tuningCurves = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, )));
#         po = GetPO.POofPopulation(tuningCurves)
#         print "po done "
#         poe = po[:NE]
#         # sigarray = np.array([0.0125, 0.0125/2, 0.0125/4, 0.0125/8])
#         print "computing prob E --> E"
#         for i in np.arange(NE):
#             cprob[i, :NE] = ConProbFunc(np.abs(poe[i] - poe[:]), thetaSig) # E --> E
#         print "computing prefactor"
#         zb = K / cprob[:NE, :NE].sum(0)
#         cprob[:NE, :NE] = cprob[:NE, :NE] * zb
#         print "computing prob I --> E"
#         for i in np.arange(NE):
#             cprob[i, NE:] = ConProbFunc(np.abs(po[i] - po[NE:]), thetaSigI) # I --> E
#         print "computing prefactor"        
#         zb = K / cprob[:NE, NE:].sum(0)
#         cprob[:NE, NE:] = cprob[:NE, NE:] * zb
#         cprob[NE:, :NE] = float(K) / NE     # E --> I
#         cprob[NE:, NE:] = float(K) / NI     # I --> I        
# #        cprob[:NE, NE:] = float(K) / NE     # E --> I
#     elif NetworkType == 'oriII': # II
#         tuningCurves = np.load('/homecentral/srao/Documents/code/mypybox/db/data/tuningCurves_%s.npy'%((dbName, )));
#         po = GetPO.POofPopulation(tuningCurves)
#         print "po done "
#         poe = po[:NE]
#         # sigarray = np.array([0.0125, 0.0125/2, 0.0125/4, 0.0125/8])
#         print "computing prob I --> I"
#         for i in np.arange(NE, NE+NI):
#             cprob[i, NE:] = ConProbFunc(np.abs(po[i] - po[NE:]), thetaSigI) # I --> I
#         print "computing prefactor"
#         zb = K / cprob[NE:, NE:].sum(0)
#         cprob[NE:, NE:] = cprob[NE:, NE:] * zb
#         cprob[NE:, :NE] = float(K) / NE # E --> I
#         cprob[:NE, :NE] = float(K) / NE # E --> E
#         cprob[:NE, NE:] = float(K) / NI # I --> E
#     elif NetworkType == 'check':
#         NE = 2
#         NI = 2
#         idxvec = np.zeros((NE + NI, ), dtype = np.int32)
#         nPostNeurons = np.zeros((NE + NI, ), dtype = np.int32)
#         sparseVec = np.zeros(shape = (cprob.sum(), ), dtype = np.int32)
#         rows = 2
#         clmns = 2
#         cprob = np.array([[0, 0], [1, 0]])
#         print cprob
#         GenSparseMat(cprob.astype(np.int32), rows, clmns, sparseVec, idxvec, nPostNeurons)
#         sys.exit()
#     elif NetworkType == 'uni':
#         cprob[:NE, :NE] = float(K) / NE # E --> E
#         cprob[NE:, :NE] = float(K) / NE # E --> I
#         cprob[:NE, NE:] = float(K) / NI # I --> E
#         cprob[NE:, NE:] = float(K) / NI # I --> I
#     print 'generating  connectivity matrix...',
#     sys.stdout.flush()
#     cprob = cprob > np.random.uniform(size = (cprob.shape))
#     print 'done!'
#     idxvec = np.zeros((NE + NI, ), dtype = np.int32)
#     nPostNeurons = np.zeros((NE + NI, ), dtype = np.int32)
#     sparseVec = np.zeros(shape = (cprob.sum(), ), dtype = np.int32)
#     rows = NE + NI
#     clmns = NE + NI
#     print 'Generating sparse representation ...',
#     sys.stdout.flush()
#     GenSparseMat(cprob.astype(np.int32), rows, clmns, sparseVec, idxvec, nPostNeurons)
#     print 'done!'
#     print 'npost sum =', nPostNeurons.sum()
#     print 'convec sum = ', cprob.sum()
# #     plt.ion()
# #     xxx = cprob[:NE, :NE]
# #     plt.hist(xxx.sum(0), 100)
# #     plt.waitforbuttonpress()
# #     plt.clf()
# # #    np.save('convec', cprob.astype(np.int32))
# # #    convec[:NE, :NE] = cprob[:NE, :NE] > np.random.uniform(size = (cprob[:NE, :NE].shape))
#     figFolder = figFolder = basefolder + '/nda/spkStats/figs/rewiring/'
#     plotCon = 'I'
#     for i in range(25):
#         if(plotCon == 'E'):
#             idx = np.random.randint(0, NE, 1)[0]
#             plt.hist(poe[cprob[:NE, idx]] * 180.0 / np.pi)
# #            plt.hist(poe[cprob[idx, :NE]] * 180.0 / np.pi)
#             plt.title("PO = %s"%(poe[idx] * 180.0 / np.pi))
#             plt.waitforbuttonpress()
#             Print2Pdf(plt.gcf(),  figFolder + 'rewired_input_ori_distr_fig%s'%(i), [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])
#             plt.clf()
#         elif(plotCon == 'I'):
#             idx = np.random.randint(NE, NE+NI, 1)[0]
#             plt.hist(po[cprob[idx, NE:]] * 180.0 / np.pi)
#             plt.title("PO = %s"%(po[idx] * 180.0 / np.pi))
#             plt.waitforbuttonpress()
#             Print2Pdf(plt.gcf(),  figFolder + 'rewired_input_ori_distr_fig%s'%(i), [4.6,  3.39], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.175, 0.15, .78, .75])
#             plt.clf()
#     kb.keyboard()



# #-------------------
#     for i in np.arange(NE):
#         if(i < NE):
#             for j in np.array([101]): #np.arange(NE):
#                 cprob[i, j] = 1000.0 / NE
# #            kb.keyboard()
#     for j in np.array([101]): #np.arange(NE):
#         probSum = 0.0;
#         for i in np.arange(NE):
#             probSum += cprob[i, j]
#         zb = K / probSum
#         cprob[:, j] = cprob[:, j] * zb
#     c = cprob[:, 101] > np.random.uniform(size = cprob[:, 101].shape)
#     poe = po[:NE]
#     print c.sum()
#     plt.hist(poe[c])
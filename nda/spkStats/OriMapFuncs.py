basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
sys.path.append(basefolder)
import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
from scipy import interpolate
import GetPO

def CircVarNeig(atTheta):
    zk = 0.0
    firingRate = np.nan
    if(atTheta.size > 1):
        firingRate = np.ones((atTheta.size, ))
        zk = np.dot(firingRate, np.exp(2j * atTheta * np.pi / 180))
    return 1 - np.absolute(zk) / np.sum(firingRate)

#def movingaverage(xx, yy): #(interval, window_size):
#    window = numpy.ones(int(window_size))/float(window_size)
 #   return numpy.convolve(interval, window, 'same')
     
 
def AzimuthVsPO():
    patchSize = int(np.sqrt(nNeurons))
    azimuthOfNeuron = np.zeros((nNeurons, ))
    IF_PERI_OR_CENTER = np.empty((nNeurons, ))
    IF_PERI_OR_CENTER[:] = False
    L = 1.0
    for i in range(nNeurons):
        tmp = np.unravel_index(i, (patchSize, patchSize))
        x = np.fmod(float(i), patchSize) * (L / (patchSize - 1));
        y = np.floor(float(i) / patchSize) * (L / (patchSize - 1.0))
        x = x - (L * 0.5)
        y = y - (L * 0.5)
        #azimuthOfNeuron[i] = 0.5 * (np.arctan2(float(y), float(x))) + np.pi * 0.5 # in radians
        azimuthOfNeuron[i] = (np.arctan2(float(y), float(x))) + np.pi # in radians
        IF_PERI_OR_CENTER[i] = np.sqrt(x **2 + y **2 ) > 0.25 # true if in peri
    validIdx = np.logical_not(np.reshape(invalidNeurons, (nNeurons, )))
    azimuthOfNeuron = azimuthOfNeuron * 180.0 / np.pi
    centerIdx = np.logical_and(validIdx, np.logical_not(IF_PERI_OR_CENTER))
    periIdx = np.logical_and(validIdx, IF_PERI_OR_CENTER)
    tmpPoe = poe * 180 / np.pi
    tmpPoeIdx0 = np.logical_and(azimuthOfNeuron < 90, tmpPoe > 90.0)
    tmpPoeIdx = np.logical_and(azimuthOfNeuron > 300, tmpPoe < 90.0)
    tmpPoe[tmpPoeIdx] = np.nan #tmpPoe[tmpPoeIdx] + 180.0
    tmpPoe[tmpPoeIdx0] = np.nan #tmpPoe[tmpPoeIdx] - 180.0
    nDiscaredNeurons = 0
    nDiscaredNeurons = tmpPoeIdx.sum() + tmpPoeIdx0.sum()
    #aziAxis = np.linspace(0.0, 360.0, 360)
    #aziOriFunc = interpolate.UnivariateSpline(azimuthOfNeuron[periIdx], tmpPoe[periIdx], s= 0.1)(aziAxis)   #interp1d(azimuthOfNeuron[periIdx], tmpPoe[periIdx], kind='cubic')
    #kb.keyboard()
#    aziOriFunc = interp1d(azimuthOfNeuron[periIdx], tmpPoe[periIdx], kind='cubic')
    plt.plot(azimuthOfNeuron[periIdx], tmpPoe[periIdx], 'r.', label = r'periphery, N=%s'%(np.sum(~np.isnan(tmpPoe[periIdx]))))
    #plt.plot(aziAxis, aziOriFunc, 'k')
    plt.plot(azimuthOfNeuron[centerIdx], tmpPoe[centerIdx], 'k.', label = 'center, N=%s'%(np.sum(~np.isnan(tmpPoe[centerIdx]))))
    plt.legend()
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('PO')    
    plt.title(r'%s neurons, $fr_{thresh} >= %s, cv_{thresh} < %s$'%(neuronType, firingThresh, cvThresh))
    plt.ion()
    filename = 'azimuth_vs_po_%s_%s'%(neuronType, dbName)
    ftsize = 12
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png',  paperSize = [8.26, 5.8], labelFontsize = ftsize, tickFontsize = ftsize, titleSize = ftsize)
    print np.min(tmpPoe), np.max(tmpPoe)
#    kb.keyboard()
    np.savetxt('delthis', np.transpose([azimuthOfNeuron[periIdx], tmpPoe[periIdx]]), delimiter = ';')
    plt.waitforbuttonpress()
        
def test_func(values):
#    print values
    out = np.nan
    poAtCenter = values[4] # 8 neighbours and center element 
    idx = [0, 1, 2, 3, 5, 6, 7, 8]
    values = values[idx]
    neighPOs = values[~np.isnan(values)]
    if(neighPOs.size > 0):
        out = np.cos(2.0 * (neighPOs - poAtCenter)).mean()
        #out = CircVarNeig(neighPOs - poAtCenter)
    return out

def PlotMaskedMap(omMap, n=1000):
    rndmask = np.random.rand(*omMap.shape) > float(omMap.size - n) / float(omMap.size)
    rndmask = np.logical_or(rndmask, invalidNeurons)
    maskedOm = np.ma.array(omMap, mask = rndmask)
    cmap = plt.get_cmap('hsv')
    cmap.set_bad(color = 'w', alpha = 1.)
    print 'printing masked ori map, discarding %s pixels'%(n)
    print 'min max angles:', np.min(omMap), np.max(omMap)
    plt.imshow(maskedOm, cmap = cmap, vmin = 0.0, vmax = 180.0)
    #plt.pcolor(maskedOm, cmap = cmap, vmin = 0.0, vmax = 180.0)
    plt.colorbar()
    filename = 'masked_om_discarded%spix_%s_'%(n, neuronType) + dbName
    ftsize = 12.0
    plt.title('PO %s, %s neurons removed'%(neuronType, n))
    plt.axis('equal')
    plt.xlabel('x cordinate')
    plt.ylabel('y cordinate')
    plt.ylim((0, yDim))
    plt.xlim((0, xDim))
    plt.gca().set_xticks(np.arange(0, xDim+1, xDim/5))
    plt.gca().set_yticks(np.arange(0, yDim+1, xDim/5))
    plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
    plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
    prps = [5.26/ 1.2, 4.26/1.2]  #[3.38, 2.74]
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png',  paperSize = prps, labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)
    plt.clf()


figFolder = '/homecentral/srao/Documents/code/mypybox/nda/spkStats/figs/'
dbName = sys.argv[1] #"omR020a0T3Tr15" #
neuronType = 'I'
NE = 40000
NI = 10000
tuningCurves = np.load(basefolder + '/db/data/tuningCurves_' + dbName + '.npy')
#tuningCurves = np.load(basefolder + '/db/data/tuningCurves_omoldgff_old.npy')
circVar = np.load(basefolder + '/db/data/Selectivity_' + dbName + '.npy')
firingThresh = 5.0 # cells with value above are selected
cvThresh = 0.5 # cells with value below are selected
try:
    po = np.load('./data/po_' + dbName + '.npy')
except IOError:
    print 'computing po ...'
    po = GetPO.POofPopulation(tuningCurves)
    np.save('./data/po_' + dbName, po)
if(neuronType == 'E'):
    poe = po[:NE]
    xDim = int(np.sqrt(NE))
    yDim = xDim
    invalidNeurons = np.logical_or(np.max(tuningCurves[:NE, :], 1) < firingThresh, circVar[:NE] > cvThresh)
    nNeurons = NE
else:
    poe = po[NE:]
    xDim = int(np.sqrt(NI))
    yDim = xDim
    invalidNeurons = np.logical_or(np.max(tuningCurves[NE:, :], 1) < firingThresh, circVar[NE:] > cvThresh)
    nNeurons = NI
invalidNeurons = np.reshape(invalidNeurons, (xDim, yDim))
print '# valid neurons', np.sum(np.logical_not(invalidNeurons))

def main():
    #x = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #x = np.random.rand(9)
    IF_MASKED = 0
    if(len(sys.argv) > 2):
        IF_MASKED = sys.argv[2]
    x = poe.reshape((xDim, xDim))
    #x = x * (np.pi / 8.0)
    footprint = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]])
    #results = ndimage.generic_filter(x, test_func, footprint=footprint, mode = 'constant', cval=np.nan)
    plt.ioff()
    #plt.ion()
    plt.figure()
    #plt.imshow(x, cmap = 'hsv')
    print np.min(x), np.max(x)
    plt.pcolor(x * 180.0 / np.pi, cmap = 'hsv')
    plt.axis('equal')
    plt.xlabel('x cordinate')
    plt.ylabel('y cordinate')
    plt.ylim((0, yDim))
    plt.xlim((0, xDim))
    plt.gca().set_xticks(np.arange(0, xDim+1, xDim/5))
    plt.gca().set_yticks(np.arange(0, yDim+1, xDim/5))
    plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
    plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
    plt.colorbar()
    #kb.keyboard()
    #
    #plt.pcolor(x, vmin = 0, vmax = np.max(x[:]), cmap =  'hsv')
    #plt.colorbar()
    plt.title('PO %s neurons'%(neuronType))
    #figFolder = '/homecentral/srao/Documents/cnrs/figures/feb28/'

    filename = 'OriMapFunc_' + dbName + '_POmap_%s'%(neuronType)  
    #plt.show()
    L = float(xDim)
    nRings = 3.0
    radii = np.arange(0, L * 0.5 + 0.001, L * 0.5 / nRings) # left limits of the set
    axHandle = plt.gca()
    for kk, kRadius in enumerate(radii[1:]):
        circObj = plt.Circle((L*0.5, L*0.5), kRadius, color = 'w', fill = False, linewidth = 2)
        axHandle.add_artist(circObj)
        if(kk > 0 and kk < 5):
            plt.text(L*0.5, L*0.5 + (radii[kk] + radii[kk+1])*0.5 - 0.025, '%s'%(kk), color='w', weight = 'bold')
        if(kk == 0):
            plt.text(L*0.5, L*0.5, '0', color='w', weight = 'bold')
    plt.draw()
    #plt.waitforbuttonpress()

    #Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png') #, tickFontsize=14, paperSize = [4.0, 3.0])
    ftsize = 12.0
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png',  paperSize = [5.26/ 1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)
    plt.figure()
    #plt.imshow(results, cmap = 'rainbow', interpolation='gaussian')
    #plt.pcolor(results, vmin = 0.0, vmax = 1.0, cmap = 'rainbow')
    
#    plt.pcolor(results, vmin = -1.0, vmax = 1.0, cmap = 'rainbow')
#    plt.colorbar()
    plt.title('orimap local correlation')
    plt.xlabel('x cordinate')
    plt.ylabel('y cordinate')
    plt.gca().set_xticks(np.arange(0, xDim+1, xDim/5))
    plt.gca().set_yticks(np.arange(0, yDim+1, xDim/5))
    plt.gca().set_xticklabels(np.arange(0, 1.1, .2))
    plt.gca().set_yticklabels(np.arange(0, 1.1, .2))
    filename = 'OriMapFunc_' + dbName + '_POmapLocalCorr_%s'%(neuronType)
 #   Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png',  paperSize = [5.26/ 1.2, 4.26/1.2], labelFontsize = 10, tickFontsize = ftsize, titleSize = ftsize)
    plt.clf()
    poCnt, poBins = np.histogram(poe * 180.0 / np.pi, 10)
    print poCnt.sum()
    #kb.keyboard()
    plt.bar(poBins[:-1], poCnt, color = 'k', edgecolor='w', width = np.mean(np.diff(poBins)))
    plt.xlabel('Preffered orientation')
    plt.ylabel('Counts')
    plt.title('PO distribution in I neurons')
    plt.draw()
    filename = 'OriMapFunc_' + dbName + '_POhist_%s'%(neuronType)
    Print2Pdf(plt.gcf(), figFolder + filename, figFormat='png', paperSize = [6.0, 4.56])
#plt.waitforbuttonpress()
    plt.figure()
    if(IF_MASKED):
        nPixels = x.size
        maxPixels2discard = int(nPixels * 0.99)
        nMaps = 10
        mapDiscardStep = int(nPixels / 40.0)  # # of pixels to discard
        nPixels2discard = np.arange(mapDiscardStep, maxPixels2discard, mapDiscardStep)
        ommap = x * 180.0 / np.pi
        ommap = PlotSubsampledOM(ommap)
        for ii, iin in enumerate(nPixels2discard):
            PlotMaskedMap(ommap, iin)

def PlotSubsampledOM(omMap):
    # omMap in degrees
    poe = omMap
    rows, columns = poe.shape
    # poe[np.logical_and(poe >= 0.0, poe < 45.0)] = 0.0
    # poe[np.logical_and(poe >= 45.0, poe < 90.0)] = 45.0
    # poe[np.logical_and(poe >= 90.0, poe < 90.0 + 45.0)] = 90.0
    # poe[np.logical_and(poe >= 90.0 + 45.0, poe < 180.0)] = 90.0 + 45.0
    return poe
    # plt.ion()
    # plt.imshow(np.reshape(poe, (200, 200)), cmap = 'hsv')
    # plt.colorbar()
    # kb.keyboard()
if __name__ == '__main__':
    main()
#    AzimuthVsPO()

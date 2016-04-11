basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf
import Keyboard as kb
#sys.path.append("/homecentral/srao/Documents/code/tmp")
import PltCircvar
import SetAxisProperties as AxPropSet

pltLineWidth = 0.5

if __name__ == "__main__":
#    filename = sys.argv[1]
    filenames = ['bidirK500CFFei2em1rho5_xi0.8_kff800_p0', 'bidirK500CFFei8em1rho5_xi0.8_kff800_p0', 'bidirK500CFFi4em1rho5_xi0.8_kff800_p0', 'bidirK500CFFi16em1rho5_xi0.8_kff800_p0']
#    filenames = ['bidirK500CFFei2em1rho5_xi0.8_kff800_p0', 'bidirK500CFFi4em1rho5_xi0.8_kff800_p0', 'bidirK500CFFi16em1rho5_xi0.8_kff800_p0']    
#    filenames = ['bidirI2Irho5_xi0.8_kff800_p0', 'bidirK500CFFi16em1rho5_xi0.8_kff800_p0', 'bidirK500CFFei2em1rho5_xi0.8_kff800_p0']    
    nFiles = len(filenames)
    NE = 20000
    NI = 20000
    fgE, axE = plt.subplots()
    fgI, axI = plt.subplots()
    datafolder = "/homecentral/srao/db/data/"
    for i, iFile in enumerate(filenames):
        PltCircvar.plotlogfr(iFile, axE, axI, '')
    AxPropSet.SetProperties(axE, [-3, 3], [0, 1], 'Log firing rates', 'Probability')
    AxPropSet.SetProperties(axI, [-3, 3], [0, 1.2], 'Log firing rates', 'Probability')
    figFormat = 'eps'
    paperSize = [2.0, 1.5]
    axPosition = [0.27, .27, .65, .65]
    figFolder = '/homecentral/srao/Documents/code/tmp/figs/publication_figures/'
    fignameE = 'brdI_fr_E'
    fignameI = 'brdI_fr_I'
    filetag = ''
    Print2Pdf(fgE,  figFolder + fignameE + filetag,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    Print2Pdf(fgI,  figFolder + fignameI + filetag,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)    
    plt.close('all')
    fgE, axE = plt.subplots()
    fgI, axI = plt.subplots()
    for i, iFile in enumerate(filenames):
        circVar = np.load(datafolder + 'Selectivity_' + iFile + '.npy')
        PltCircvar.PlotCircVars(circVar, NE, NI, axE, axI, '')
    AxPropSet.SetProperties(axE, [0, 1], [0, 3], 'Circular Variance', 'Probability')
    AxPropSet.SetProperties(axI, [0, 1], [0, 5], 'Circular Variance', 'Probability')
    fignameE = 'brdI_circvar_E'
    fignameI = 'brdI_circvar_I'
    filetag = ''
    Print2Pdf(fgE,  figFolder + fignameE + filetag,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    Print2Pdf(fgI,  figFolder + fignameI + filetag,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=8, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
    plt.show()




    


    
    


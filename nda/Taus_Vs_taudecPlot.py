basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
import sys, os
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
import Print2Pdf as pdf
import Keyboard as kb

tau_s = [3, 6, 12, 24, 48]

fgLog, axLog = plt.subplots()
tdE = np.load('est_tau_dec_E.npy')
tdI = np.load('est_tau_dec_I.npy')
pcolor = ('b', 'g', 'r', 'k', 'c')
figFolder = '/homecentral/srao/Documents/code/tmp/figs/publication_figures/powerlaw/'
axPosition = [0.26, 0.28, .65, 0.65]
paperSize = [2.0, 1.5]
axLog.loglog(1 - np.array(tdE[0][0][:]), tdE[1][0][:] / 3.0, '.k-', linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1, label = 'E')
axLog.loglog(1 - np.array(tdI[0][0][:]), tdI[1][0][:] / 3.0, '.r-', linewidth = 0.2, markersize = 2.0 , markeredgewidth = 0.1, label = 'I')
axLog.legend(frameon = False, loc = 3, numpoints = 1, prop = {'size': 8})
plt.xlabel('1-p')
plt.ylabel(r'$\tau_{dec} / \tau_s$')
# #plt.yticks([0, 45, 90])
# plt.xticks([0, 0.5, 1.0])

filename = 'est_tau_dec_ov_taus_3ms_loglog'
figFormat = 'eps'
axPosition = [0.33, 0.33, .6, 0.6]
pdf.Print2Pdf(fgLog,  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
plt.ion()
plt.show()
plt.waitforbuttonpress()
plt.close('all')
fgE, axE = plt.subplots()
fgI, axI = plt.subplots()
for kk, kTau in enumerate(tau_s):
    axE.errorbar(tdE[0][kk][:], tdE[1][kk][:] / kTau, yerr = tdE[2][kk][:] /  kTau, color = pcolor[kk], fmt = '.-', ecolor = pcolor[kk], elinewidth = 0.2, linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)
#    axE.errorbar(tdE[0][kk][:-1], tdE[1][kk][:-1] / kTau, yerr = tdE[2][kk][:-1] /  kTau, color = pcolor[kk], fmt = '.-', ecolor = pcolor[kk], elinewidth = 0.2, linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)    
    axI.errorbar(tdI[0][kk][:], tdI[1][kk][:] / kTau, yerr = tdI[2][kk][:] / kTau, color = pcolor[kk], fmt = '.-', ecolor = pcolor[kk], elinewidth = 0.2, linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)
axE.set_xlabel('p')
axE.set_ylabel(r'$\tau_{dec} / \tau_s$')
axI.set_xlabel('p')
axI.set_ylabel(r'$\tau_{dec} / \tau_s$')
axE.set_yticks([0, 35, 70])
axI.set_yticks([0, 45, 90])
axE.set_xticks([0, 0.5, 1.0])
axI.set_xticks([0, 0.5, 1.0])
filenameE = 'tdec_vs_tsyn_E'
filenameI = 'tdec_vs_tsyn_I'
figFormat = 'eps'
plt.show()
plt.waitforbuttonpress()
pdf.Print2Pdf(fgE,  figFolder + filenameE,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
pdf.Print2Pdf(fgI,  figFolder + filenameI,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)    


plt.close('all')

td = np.load('est_tau_dec_E.npy')
plt.errorbar(td[0][0][:], td[1][0][:]/3.0, yerr = td[2][0][:], fmt = '.k-', ecolor = 'k', elinewidth = 0.2, linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)
td = np.load('est_tau_dec_I.npy')
plt.waitforbuttonpress()
plt.errorbar(td[0][0][:], td[1][0][:]/3.0, yerr = td[2][0][:], fmt = '.r-', ecolor = 'r', elinewidth = 0.2, linewidth = 0.2, markersize = 2.0 , markeredgewidth = 0.1)
plt.legend({'I', 'E'}, frameon = False, loc = 2, numpoints = 1, prop = {'size': 8})
plt.show()
plt.waitforbuttonpress()

plt.xlabel('p')
plt.ylabel(r'$\tau_{dec} / \tau_s$')
plt.yticks([0, 45, 90])
plt.xticks([0, 0.5, 1.0])
filename = 'est_tau_dec_ov_taus_0'
figFormat = 'eps'
axPosition = [0.26, 0.28, .65, 0.65]
pdf.Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
plt.waitforbuttonpress()

from Startup import *
import LoadSimResults as sim
from spkStats import *
import traceback, code, sys
import pylab as plt
import numpy as np
def cvdistr(NE, NI, simDuration, discardDuration = 0.0):
    # simDuration = 1.0 # seconds
    # NE = 10000
    # NI = 10000
    print "NE = %s, NI = %s, simDuration = %s" %(NE, NI, simDuration)
#try:
#addpath('/home/dhansel/code/mypybox')
    st = sim.loadSpks()
    st = st[st[:, 0] > discardDuration*1000.0, :]
    se = st[st[:, 1] < NE, :]
    si = st[st[:, 1] >= NE, :]
    
    print len(se), len(si)
    print "mean firing rate E : %s, I : %s " %(float(len(se))/ (NE * (simDuration - discardDuration)), float(len(si))/ (NI * (simDuration - discardDuration)))
    cvE = ComputeCV(se, 5)
    frE = FiringRate(se, simDuration, 5)
    cvI = ComputeCV(si, 5)
    frI = FiringRate(si, simDuration, 5)
    plt.ion()
    plt.figure()
    plt.plot(frE, cvE, 'go')
    plt.plot(frI, cvI, 'ro')
    plt.ylabel('CV')
    plt.xlabel('firing rate (Hz)')
    cvE = cvE[~np.isnan(cvE)]
    cvI = cvI[~np.isnan(cvI)]
    plt.figure()
    n, bins, patches = plt.hist(cvE[cvE < 2.5], 100)
    plt.setp(patches, 'facecolor', 'g')
    plt.figure()
    n, bins, patches = plt.hist(cvI[cvI < 2.5], 100)
    plt.setp(patches, 'facecolor', 'r')
    # plt.figure()
    #  frE = frE[~np.isnan(frE)];
    # plt.hist(frE[frE < 200], 100)
    # plt.figure()
    # plt.hist(frI[~np.isnan(frI)], 100)
    plt.draw()
# except:
#     type, value, tb = sys.exc_info()
#     traceback.print_exc()
#     last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
#     frame = last_frame().tb_frame
#     ns = dict(frame.f_globals)
#     ns.update(frame.f_locals)
#     code.interact(local=ns)

if __name__ == "__main__":
    NE = sys.argv[1]
    NI = sys.argv[2]
    simDuration = sys.argv[3]
    discardDuration = sys.argv[4]
    cvdistr(int(NE), int(NI), float(simDuration), float(discardDuration))
    plt.waitforbuttonpress()

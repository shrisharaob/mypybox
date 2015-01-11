import numpy as np
import sys
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")
from DefaultArgs import DefaultArgs


tau = [24.0]
alpha = np.array([0.6])
simT = 100000
condition = [0, 1]
tauStart = 24.0
tauStep = 1.0
tauEnd = 24.1
tauStart, tauStep, tauEnd, alpha, simT, condition = DefaultArgs(sys.argv[1:], [tauStart, tauStep, tauEnd, alpha, simT, condition])

tau = np.arange(float(tauStart), float(tauEnd), float(tauStep))
np.asarray(condition).astype('int')
np.asarray(simT).astype('int')
print alpha
np.asarray(alpha).astype('float')
print condition
fp = open("list.txt", 'w')
for cc, cCond in enumerate(condition):
    print "condition : ", cc
    simDuration = int(simT) + int(cCond)
    for mm, mAlpha in enumerate(alpha):
        print "alpha : ", mAlpha
        theta = ['%s%s%s'%((int(10*x), int(10 * mAlpha), cCond)) for x in tau]
        fn = ['spkTimes_%.1f_%.1f_%d.csv'%((float(mAlpha), x, simDuration)) for x in tau]
        print theta, fn
        for kk in range(len(fn)):
            fp.write("%s %s\n" % ((fn[kk], theta[kk])))    

fp.close()

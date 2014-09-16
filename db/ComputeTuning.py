import MySQLdb as mysql
import numpy as np
import code
import sys
import pylab as plt

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return 

dbName = "tstDb"
N_NEURONS = np.arange(0, 10001, 1)
thetaStart = 0
thetaStep = 45
thetaEnd = 360
theta = np.arange(thetaStart, thetaEnd, thetaStep)
trialLength = 25.0 # in seconds
db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
dbCursor = db.cursor()
db.autocommit(True)
tuningCurve = np.zeros((len(N_NEURONS), len(theta)))
for idx, kNeuron in enumerate(N_NEURONS):
    print 'NEURON - ', kNeuron, 'theta :', 
    sys.stdout.flush()
    for thetaIdx, kTheta in enumerate(theta):
        print kTheta,
        sys.stdout.flush()
        nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s and theta = %s;", (kNeuron, kTheta))
        tuningCurve[idx, thetaIdx] = nSpks / trialLength
        
    print ' '
dbCursor.close()
db.close()
np.save('tuningCurves', tuningCurve)
keyboard()

import MySQLdb as mysql
import numpy as np
import code
import sys
import pylab as plt
from multiprocessing import Pool
from functools import partial

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


def NspksForTheta(dbName, neuronId, discardTime, theta):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s and theta = %s and spkTimes > %s;", (neuronId, theta, discardTime))
#    print nSpks
    dbCursor.close()
    db.close()
    return float(nSpks)

def NspksForThetaForAllTrials(dbName, neuronId, discardTime, nTrials, theta):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    nSpks = 0
    #print theta
    for kTrial in np.arange(nTrials):
        kTheta = int((kTrial+1) * 1000) + int(theta)
#kTheta = theta
        nSpks += dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s and theta = %s and spkTimes > %s;", (neuronId, kTheta, discardTime))
#    print nSpks
    dbCursor.close()
    db.close()
    return float(nSpks)




dbName = sys.argv[1] #"bidirEE" #"anatomic" #"tstDb"
N_NEURONS = np.arange(0, 20000, 1)
#N_NEURONS = np.arange(0, 50000, 1)
thetaStart = 0.0
thetaStep = 22.5/2.0
thetaEnd = 180.0
theta = np.arange(thetaStart, thetaEnd, thetaStep)
#theta = np.array([0., 22.5, 45., 56.25, 67.5, 90. , 112.5, 123.75, 135., 157.5])
theta = theta.astype('int')
trialLength = 3.0 # in seconds
discardTime = 1000.0 #ms
if(len(sys.argv)> 1):
    try:
        nTrials = int(sys.argv[2])
    except ValueError:
        print 'ntrials not an interetr !'
        raise
print "nTrials = ", nTrials

#db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
#dbCursor = db.cursor()
#db.autocommit(True)
tuningCurve = np.zeros((len(N_NEURONS), len(theta)))
trialLength = trialLength - discardTime / 1000.0
z = trialLength * nTrials
pool = Pool(16)
print "Computing ...",
sys.stdout.flush()
for idx, kNeuron in enumerate(N_NEURONS):
#    print 'NEURON - ', kNeuron, 'theta :',
#    print kNeuron,
    sys.stdout.flush()
    
#    nSpksVec = pool.map(partial(NspksForTheta, dbName, kNeuron, discardTime), theta)
    nSpksVec = pool.map(partial(NspksForThetaForAllTrials, dbName, kNeuron, discardTime, nTrials), theta)

    # for thetaIdx, kTheta in enumerate(theta):
    #     print kTheta,
    #     sys.stdout.flush()
        
    #       nSpks = dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s and theta = %s and spkTimes > %s;", (kNeuron, kTheta, discardTime))
    tuningCurve[idx, :] = np.array(nSpksVec, dtype = 'float') / z
#    print ' '
#dbCursor.close()
#db.close()

pool.close()
print "saving as: ", './data/tuningCurves_' + dbName
np.save('./data/tuningCurves_'+dbName, tuningCurve)
#keyboard()

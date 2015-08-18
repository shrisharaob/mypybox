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

def NspksForThetaForAllTrials(dbName, neuronId, nTrials, stimIdx, stimTimes, theta):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    dbCursor = db.cursor()
    db.autocommit(True)
    nSpks = 0
    thetaArray = np.arange(0, 360.0, 30)
    allTrialLength = 0.0
    for kTrial in np.arange(nTrials):
        query = np.where(np.all(stimIdx == np.array([neuronId, kTrial, np.where(thetaArray == theta)[0][0]]), axis = 1))[0]
        if(query.size):
            spkStartTime, spkStopTime = stimTimes[query[0], :]
            allTrialLength += (spkStopTime - spkStartTime)
            kTheta = int((kTrial+1) * 1000) + int(theta)
            nSpks += dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId = %s and theta = %s and spkTimes > %s and spkTimes < %s;", (neuronId, kTheta, spkStartTime, spkStopTime))
    dbCursor.close()
    db.close()
    out = 0.0
    if(allTrialLength > 0):
        out = float(nSpks) / (allTrialLength)
    return float(nSpks)

dbName = sys.argv[1]
N_NEURONS = np.arange(35)
thetaStart = 0.0
thetaStep = 30.0
thetaEnd = 360.0
theta = np.arange(thetaStart, thetaEnd, thetaStep)
theta = theta.astype('int')
if(len(sys.argv)> 1):
    try:
        nTrials = int(sys.argv[2])
    except ValueError:
        print 'ntrials not an interetr !'
        raise
print "nTrials = ", nTrials
tuningCurve = np.zeros((len(N_NEURONS), len(theta)))
pool = Pool(theta.size)
print "Computing ...",
sys.stdout.flush()
stimTimesArray = np.loadtxt('stimtimes.csv', delimiter = ';')
stimIdx = stimTimesArray[:, [0, 1, 2]]
stimTimes = stimTimesArray[:, [3, 4]]
for idx, kNeuron in enumerate(N_NEURONS):
    sys.stdout.flush()
    nSpksVec = pool.map(partial(NspksForThetaForAllTrials, dbName, kNeuron, nTrials, stimIdx, stimTimes), theta)
    tuningCurve[idx, :] = np.array(nSpksVec, dtype = 'float')
pool.close()
print "saving as: ", './data/tuningCurves_' + dbName
np.save('./data/tuningCurves_'+dbName, tuningCurve)
#keyboard()

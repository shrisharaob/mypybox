basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import code, sys, os
import pylab as plt
import MySQLdb as mysql
sys.path.append(basefolder)
import Keyboard as kb
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import GetPO

workstationIPlist = [72, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]

def RetrunDbCursor(dbName):
    IF_SUCCESS = False
    tryCount = 0
    while(not(IF_SUCCESS or (tryCount >= len(workstationIPlist)))):
        try:
            print 'searching for ', dbName, ' on pc%s'%(workstationIPlist[tryCount]),
            db = mysql.connect(host = "172.20.62.%s"%(workstationIPlist[tryCount]), user = "root", passwd = "toto123", db = dbName)
            dbCursor = db.cursor()
            db.autocommit(True)
            IF_SUCCESS = True
            print ' ==> found db ', dbName
        except:
            print " ==> db not found"
            tryCount = tryCount + 1
            # if tryCount >= len(workstationIPlist):
            #     break;
    out = []
    if(IF_SUCCESS):
        out = [dbCursor, db]
    return out

def MeanFiringRate(dbName, simDuration, NE = 10000, NI = 10000):
    # db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
    # dbCursor = db.cursor()
    # db.autocommit(True)
    tmpDBCursor = RetrunDbCursor(dbName)
    if(len(tmpDBCursor)):
        dbCursor = tmpDBCursor[0]
        db = tmpDBCursor[1]
        discardTime = 2000
        duration = simDuration - discardTime
        theta = 1000
        frE = float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId < %s AND theta = %s AND spkTimes > %s", (NE, theta, discardTime))) / (duration * 1e-3 * NE)
        frI = float(dbCursor.execute("SELECT spkTimes FROM spikes WHERE neuronId > %s AND theta = %s AND spkTimes > %s", (NE, theta, discardTime))) / (duration * 1e-3 * NI)
        print '-------------------------------------------------'
        print "MEAN RATE E = ", frE
        print "MEAN RATE I = ", frI
        dbCursor.close()
        db.close()
        out = [frE, frI]
    else:
        print 'db not found on any workstation!'
        out = []
    return out

def InstantaneousPopRate(st, simDuration, NE, NI):
    discardTime = 2000
    neuronIdx = np.unique(st[:, 1])
    print np.max(st[:, 0])
    idx = st[:, 0] > discardTime
    st = st[idx, :]
    idx = st[:, 1] < NE
    ste = st[idx, :]
    sti = st[~idx, :]
    spkStart = discardTime
    intervalLength = (simDuration - spkStart) * 1e-3 # in sec
    binSize = 1 #ms
    nBins = int(intervalLength + 1)
    cntsE, bins = np.histogram(ste, nBins)
    cntsI, bins = np.histogram(sti, nBins)
    cntsE = cntsE.astype(float)
    cntsI = cntsI.astype(float)
    instFrE = cntsE / (NE * binSize)
    instFrI = cntsI / (NI * binSize)
    print cntsE.sum() / (NE * binSize * 1e-3), cntsI.sum() / (NI * binSize * 1e-3)
    print np.sum(float(len(ste[:, 0]))) / (NE * intervalLength), np.sum(float(len(sti[:, 0]))) / (NE * intervalLength)
    return instFrE, instFrI 

if __name__ == '__main__':
    [dbName, simDuration, NE, NI] = DefaultArgs(sys.argv[1:], ['', 0, 10000, 10000])
    MeanFiringRate(dbName, float(simDuration), int(NE), int(NI))


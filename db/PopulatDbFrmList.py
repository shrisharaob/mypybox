#  Populate db from supplied list 
import MySQLdb as mysql
import numpy as np
import os, sys

def Populate(dbName, fileList, tblType, filebase=''):
    fp = open(fileList)
    pairs = (line.split() for line in fp)
    if(tblType == 'defalut'):
        filepath_theta_pairs = [(filepath, float(theta)) for filepath, theta in pairs]
        filenames = [tmp[0] for tmp in filepath_theta_pairs]
        theta = np.array([tmp[1] for tmp in filepath_theta_pairs])
    else:
        filepath_theta_pairs = [(filepath, float(theta), int(trialId)) for filepath, theta, trialId in pairs]
        filenames = [tmp[0] for tmp in filepath_theta_pairs]
        theta = np.array([tmp[1] for tmp in filepath_theta_pairs])
        trialId = np.array([tmp[2] for tmp in filepath_theta_pairs])

    fp.close()
    
    tableName = "spikes"
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName, local_infile = 1)
    db.autocommit(True)
    dbCursor = db.cursor()
    dbCursor.execute("SET unique_checks=0;")
    for mm, mTheta in enumerate(theta):
        mFileName = filebase + filenames[mm]
        print mFileName
        if(os.path.isfile(mFileName)):
            print "populating db: theta = %s" % (mTheta,)
            if(tblType == 'defalut'):
                dbCursor.execute("LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ';' LINES TERMINATED BY '\\n' (spkTimes, neuronId) SET theta = %s;" % (mFileName, tableName, mTheta))        
            else:
                for kk, kTr in enumerate(trialId):
                    dbCursor.execute("LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ';' LINES TERMINATED BY '\\n' (spkTimes, neuronId) SET theta = %s, trial = %s;" % (mFileName, tableName, mTheta, kTr))
        else:
            print "file does not exist (%s) :"  %(mFileName, )
    print "done"
    dbCursor.close()
    db.close()

if __name__ == "__main__":
    argc = len(sys.argv)
    filebase = "/homecentral/srao/Documents/code/cuda/cudanw/"
    thetaStart = 0
    thetaStep = 45
    thetaEnd = 1
    dbName = "biII_8"
    tblType = 'defalut'
    if(argc > 1):
        dbName = sys.argv[1]
        print "dbname = ", dbName
    if(argc > 2):
        if(sys.argv[2] != "[]"):
            filebase = sys.argv[2]
            print "filebase = ", filebase
    if(argc > 3):
        if(sys.argv[3] != "[]"):
            fileList = sys.argv[3]
    if(argc > 4):
        if(sys.argv[4] != "[]"):
            tblType = sys.argv[4]
            
    print "table type:", tblType
    Populate(dbName, fileList, tblType, filebase)



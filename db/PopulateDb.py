# Populate existing mysql database with spike times and stimulus features
# 


import MySQLdb as mysql
import numpy as np
import os, sys

def Populate(dbName, filebase, thetaStart = 0, thetaEnd = 360, thetaStep = 45):
    theta = np.arange(thetaStart, thetaEnd, thetaStep)
    theta = np.array([10, 12, 24, 6, 8])
    
    theta = np.array([12111])
    tableName = "spikes"
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName, local_infile = 1)
    db.autocommit(True)
    dbCursor = db.cursor()
    IF_FILE_LIST = True
    # if(IF_FILE_LIST):
    #     filenames = 
    for mTheta in theta:
#        filename = filebase + "spkTimes_theta%s.csv" % (mTheta,) 
#        filename = filebase + "spkTimes_trial%s.csv" % (int(mTheta),) 
#        filename = filebase + "spkTimes_1.0_%s.0_50.csv" %(int(mTheta))
        filename = filebase + "spkTimes_1.0_12_250001.csv"
#        filename = filebase + "spkTimes_1.0_3.0_10.csv"
        print filename
        if(os.path.isfile(filename)):
            print "populating db: theta = %s" % (mTheta,)
            dbCursor.execute("LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ';' LINES TERMINATED BY '\\n' (spkTimes, neuronId) SET theta = %s;" % (filename, tableName, mTheta))        
        else:
            print "file does not exist (%s) :"  %(filename,)

#    print "Creating MSQL indices",
#    sys.stdout.flush()
 #  dbCursor.execute("ALTER TABLE %s ADD KEY (neuronId, theta);" % (tableName, ))
    print "done"
    dbCursor.close()
    db.close()



if __name__ == "__main__":
    argc = len(sys.argv)
    filebase = "/homecentral/srao/Documents/code/cuda/cudanw/"
    #filebase = "/home/shrisha/Documents/exp/mysql/data/"
    #filebase = "/homecentral/srao/Documents/code/cuda/tmp/pc83/"
    thetaStart = 0
    thetaStep = 45
    thetaEnd = 1
    # theta = np.arange(thetaStart, thetaEnd, thetaStep)
    # tableName = "spikes"
    dbName = "biII_8" #"anatomic" #"tstDb"

    if(argc > 1):
        dbName = sys.argv[1]
        print "dbname = ", dbName
    if(argc > 2):
        if(sys.argv[2] != "[]"):
            filebase = sys.argv[2]
            print "filebase = ", filebase
    if(argc > 3):
        thetaStart = float(sys.argv[3])
    if(argc > 4):
        thetaEnd = float(sys.argv[4])
    if(argc > 5):
        thetaStep = float(sys.argv[5])

    Populate(dbName, filebase, thetaStart, thetaEnd, thetaStep)


# db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName, local_infile = 1)
# db.autocommit(True)
# dbCursor = db.cursor()
# for mTheta in theta:
#     filename = filebase + "spkTimes_theta%s.csv" % (mTheta,) 
#     filename = "/homecentral/srao/Documents/code/cuda/tmp/pc83/spkTimes_dt05.csv"
#     print filename
#     if(os.path.isfile(filename)):
#         print "populating db: theta = %s" % (mTheta,)
#         dbCursor.execute("LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ';' LINES TERMINATED BY '\\n' (spkTimes, neuronId) SET theta = %s;" % (filename, tableName, mTheta))        
#     else:
#         print "file does not exist (%s) :"  %(filename,)

# print "Creating MSQL indices",
# sys.stdout.flush()
# dbCursor.execute("ALTER TABLE %s ADD KEY (neuronId, theta);" % (tableName, ))
# print "done"
# dbCursor.close()
# db.close()

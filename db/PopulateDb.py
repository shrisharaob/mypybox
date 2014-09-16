# Populate existing mysql database with spike times and stimulus features
# 


import MySQLdb as mysql
import numpy as np
import os, sys
filebase = "/homecentral/srao/Documents/code/cuda/cudanw/"
#filebase = "/home/shrisha/Documents/exp/mysql/data/"
thetaStart = 0
thetaStep = 45
thetaEnd = 360
theta = np.arange(thetaStart, thetaEnd, thetaStep)
tableName = "spikes"
dbName = "tstDb"
db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName, local_infile = 1)
db.autocommit(True)
dbCursor = db.cursor()
for mTheta in theta:
    filename = filebase + "spkTimes_theta%s.csv" % (mTheta,) 
    print filename
    if(os.path.isfile(filename)):
        print "populating db: theta = %s" % (mTheta,)
        dbCursor.execute("LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ';' LINES TERMINATED BY '\\n' (spkTimes, neuronId) SET theta = %s;" % (filename, tableName, mTheta))        
    else:
        print "file does not exist (%s) :"  %(filename,)

print "Creating MSQL indices",
sys.stdout.flush()
dbCursor.execute("ALTER TABLE '%s' ADD KEY (neuronId, theta);" % (tableName, ))
print "done"
dbCursor.close()
db.close()

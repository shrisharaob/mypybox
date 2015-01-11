import MySQLdb as mysql
import numpy as np
import os, sys

def GenerateIndex(dbName, tblType = 'default'):
    print "generating index...", 
    sys.stdout.flush()
    tableName = "spikes"
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName, local_infile = 1)
    db.autocommit(True)
    dbCursor = db.cursor()

    if(tblType == 'default'):
        dbCursor.execute("ALTER TABLE %s ADD KEY (neuronId, theta);" % (tableName, ))
    else:
        dbCursor.execute("ALTER TABLE %s ADD KEY (neuronId, theta, trial);" % (tableName, ))
    dbCursor.close()
    db.close()
    print "done"

if __name__ == "__main__":
    argc = len(sys.argv)
    tblType = 'default'
    if(argc > 1):
        dbName = sys.argv[1]
    if(argc > 2):
        tblType = sys.argv[2]

    print "table type:", tblType
    GenerateIndex(dbName, tblType)

    

    

import MySQLdb as mysql
import numpy as np
import sys

def CreateDb(dbName, tblType = 'defalut'):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123");
    db.autocommit(True)
    dbCursor = db.cursor()
    dbCursor.execute("CREATE DATABASE " +  dbName + ";")
    dbCursor.execute("USE " + dbName + ";")
    if(tblType == 'defalut'):
        dbCursor.execute("CREATE TABLE spikes (spkId INT UNSIGNED AUTO_INCREMENT, spkTimes DOUBLE, neuronId INT, theta DOUBLE, PRIMARY KEY (spkId));")
    else:
        dbCursor.execute("CREATE TABLE spikes (spkId INT UNSIGNED AUTO_INCREMENT, spkTimes DOUBLE, neuronId INT, theta DOUBLE, trial INT, PRIMARY KEY (spkId));")
    dbCursor.close()
    db.close()


if __name__ == "__main__":
    argc = len(sys.argv)
    dbName = "biII_8" #"anatomic" #"tstDb"
    tblType = 'defalut'
    if(argc > 1):
        dbName = sys.argv[1]
        print "dbname = ", dbName
    if(argc > 2):
        tblType = sys.argv[2]

    CreateDb(dbName, tblType)







# dbName = "biII_8" #"anatomic" #"tstDb"
# db = mysql.connect(host = "localhost", user = "root", passwd = "toto123");
# db.autocommit(True)
# dbCursor = db.cursor()
# dbCursor.execute("CREATE DATABASE " +  dbName + ";")
# dbCursor.execute("USE " + dbName + ";")

# #dbCursor.execute("CREATE TABLE theta (thetaId INT UNSIGNED AUTO_INCREMENT, thetaVal DOUBLE, PRIMARY KEY (thetaId));")

# dbCursor.execute("CREATE TABLE spikes (spkId INT UNSIGNED AUTO_INCREMENT, spkTimes DOUBLE, neuronId INT, theta DOUBLE, PRIMARY KEY (spkId));")

# #dbCursor.execute("CREATE TABLE spikes ( spkTimes DOUBLE, neuronId INT, spkId INT UNSIGNED AUTO_INCREMENT, PRIMARY KEY (spkId), thetaId INT UNSIGNED, FOREIGN KEY (thetaId) REFERENCES theta(thetaId));")


# dbCursor.close()
# db.close()

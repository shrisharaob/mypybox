import MySQLdb as mysql
import numpy as np
import sys

def CreateDb(dbName):
    db = mysql.connect(host = "localhost", user = "root", passwd = "toto123");
    db.autocommit(True)
    dbCursor = db.cursor()
    dbCursor.execute("CREATE DATABASE " +  dbName + ";")
    dbCursor.execute("USE " + dbName + ";")
    dbCursor.execute("CREATE TABLE spikes (spkId INT UNSIGNED AUTO_INCREMENT, spkTimes DOUBLE, neuronId INT, theta DOUBLE, PRIMARY KEY (spkId));")
    dbCursor.close()
    db.close()




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


if __name__ == "__main__":
    argc = len(sys.argv)
    dbName = "biII_8" #"anatomic" #"tstDb"
    if(argc > 1):
        dbName = sys.argv[1]
        print "dbname = ", dbName
    
    CreateDb(dbName)

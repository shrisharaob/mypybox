import MySQLdb as mysql
import numpy as np
dbName = "bidir" #"anatomic" #"tstDb"
db = mysql.connect(host = "localhost", user = "root", passwd = "toto123");
db.autocommit(True)
dbCursor = db.cursor()
dbCursor.execute("CREATE DATABASE " +  dbName + ";")
dbCursor.execute("USE " + dbName + ";")

#dbCursor.execute("CREATE TABLE theta (thetaId INT UNSIGNED AUTO_INCREMENT, thetaVal DOUBLE, PRIMARY KEY (thetaId));")

dbCursor.execute("CREATE TABLE spikes (spkId INT UNSIGNED AUTO_INCREMENT, spkTimes DOUBLE, neuronId INT, theta DOUBLE, PRIMARY KEY (spkId));")

#dbCursor.execute("CREATE TABLE spikes ( spkTimes DOUBLE, neuronId INT, spkId INT UNSIGNED AUTO_INCREMENT, PRIMARY KEY (spkId), thetaId INT UNSIGNED, FOREIGN KEY (thetaId) REFERENCES theta(thetaId));")


dbCursor.close()
db.close()

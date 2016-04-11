import MySQLdb as mysql
import numpy as np
dbName = "tstDb"
db = mysql.connect(host = "localhost", user = "root", passwd = "q-1234");
dbCursor = db.cursor()
dbCursor.execute("CREATE DATABASE " +  dbName + ";")
dbCursor.execute("USE " + dbName + ";")
dbCursor.execute("CREATE TABLE spikes (neuronId INT, spkTimes DOUBLE, theta DOUBLE);")

dbCursor.close()
db.close()

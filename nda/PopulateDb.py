import MySQLdb as mysql
import numpy as np
import os
filebase = "/homecentral/srao/Documents/code/tmp/"
filename = "spk01.csv"
tableName = "spikes"
dbName = "tstDb04"
os.system("ln -s " + filebase + filename + " /tmp/" + filename)
db = mysql.connect(host = "localhost", user = "root", passwd = "toto123", db = dbName)
dbCursor = db.cursor()
dbCursor.execute("LOAD DATA LOCAL INFILE '/tmp/%s' INTO TABLE %s FIELDS TERMINATED BY ';' LINES TERMINATED BY '\\n';" % (filename, tableName))
tableName
dbCursor.close()
db.close()
 


from  peewee import *
from playhouse.postgres_ext import *
from numpy import *
spkDB = PostgresqlExtDatabase('simData', user='postgres')

class SimData(Model):
    neuronId = IntegerField();
    theta = DoubleField();
    spkTimes = ArrayField(DoubleField);
    
    class Meta:
        database = spkDB

st = loadtxt('spkTimes.csv')
curTheta = 0;
SimData.create_table()
for k in unique(st[:, 1]):
    print k
    tmp = SimData.create(neuronId = k, theta = curTheta, spkTimes = st[st[:, 1] == k], 0)
    tmp.save()

print 'done'

    
    

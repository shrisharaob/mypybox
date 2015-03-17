from  peewee import *
import numpy as np
spkDB = SqliteDatabase('simData.db')

class SimData(Model):
    neuronId = IntegerField();
    class Meta:
        database = spkDB
class SpikeTimes(Model):
    spkNeuronId = ForeignKeyField(SimData, related_name = 'neuron')
    theta = DoubleField();
    spkTimes = DoubleField();
    class Meta:
        database = spkDB

st = loadtxt('spkTimes.csv')
curTheta = 0;
SimData.create_table()
SpikeTimes.create_table()

with spkDB.transaction():
    for k in unique(st[:, 1]):
        tmp = SimData.create(neuronId = k)
        tmp.save()
        for m in st[st[:, 1] == k, 0]:
            tmpSt = SpikeTimes.create(spkNeuronId = tmp, theta = curTheta, spkTimes = m)
            tmpSt.save()

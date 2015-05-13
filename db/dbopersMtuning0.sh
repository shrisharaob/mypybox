#!/bin/bash

DATA_DIRECTORY=/homecentral/srao/Documents/code/cuda/tmp/alpha0/$2/
NTRIALS=$3
set -x
echo \# Trials = $NTRIALS
if [ -d "$DATA_DIRECTORY" ]; then
    python CreateMySqlDB.py $1
    python PopulatDbFrmList.py $1 /homecentral/srao/Documents/code/cuda/tmp/alpha0/$2/ list1.txt;
    python GenIndex.py $1; 
    python ComputeTuning.py $1 $NTRIALS; 
    python Selectivity.py $1; 
    python ../nda/spkStats/FFvsOri.py $1 compute $NTRIALS; 
# ##    mv /homecentral/srao/Documents/code/mypybox/nda/spkStats/FFvsOri_$2.npy /homecentral/srao/Documents/code/mypybox/nda/spkStats/data
    python ../nda/spkStats/FFvsOri.py $1
else
    echo $DATA_DIRECTORY does not exist
fi

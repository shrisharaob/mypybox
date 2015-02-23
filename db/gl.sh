#!/bin/bash

#DATA_DIRECTORY=/homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/$2/
DATA_DIRECTORY=/homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/

if [ -d "$DATA_DIRECTORY" ]; then
    python CreateMySqlDB.py $1
    python PopulatDbFrmList.py $1 /homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/ list.txt;
    python GenIndex.py $1; 
    python ComputeTuning.py $1; 
    python Selectivity.py $1; 
    python ../nda/spkStats/FFvsOri.py $1 compute; 
  #  mv FFvsOri_$2.npy /homecentral/srao/Documents/code/mypybox/nda/spkStats/data
    python ../nda/spkStats/FFvsOri.py $1
else
    echo $DATA_DIRECTORY does not exist
fi
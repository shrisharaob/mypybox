#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "USAGE: ./dbopers.sh <dbname> <datadirectory> <ntrials> <listfile>"
    exit 1
fi

#DATA_DIRECTORY=/homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/$2/
DATA_DIRECTORY=$2 # /homecentral/srao/Documents/code/cuda/
NTRIALS=$3
set -x
echo \# Trials = $NTRIALS
if [ -d "$DATA_DIRECTORY" ]; then
    python CreateMySqlDB.py $1

###    python PopulatDbFrmList.py $1 /homecentral/srao/Documents/code/cuda/tmp/alpha0/oriMap/$2/ $4;
###    python PopulatDbFrmList.py $1 /homecentral/srao/Documents/code/cuda/$2/ $4;

    python PopulatDbFrmList.py $1 $2 $4;
    python GenIndex.py $1; 
    python ComputeTuning.py $1 $NTRIALS; 
    python Selectivity.py $1; 
   
 #  python ../nda/spkStats/FFvsOri.py $1 compute $NTRIALS;
    
# ##    mv /homecentral/srao/Documents/code/mypybox/nda/spkStats/FFvsOri_$2.npy /homecentral/srao/Documents/code/mypybox/nda/spkStats/data
    
#    python ../nda/spkStats/FFvsOri.py $1
else
    echo $DATA_DIRECTORY does not exist
fi

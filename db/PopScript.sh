for i in  ; do 
    echo spkTimes_0.${i}_3.0_100000.csv 3${i} > list.txt; 
    python CreateMySqlDB.py "a5t3th${i}"; 
    python PopulatDbFrmList.py "a5t3th${i}" /homecentral/srao/Documents/code/cuda/tmp/alpha${i}/ list.txt; 
    python GenIndex.py "a5t3th${i}"; 
done

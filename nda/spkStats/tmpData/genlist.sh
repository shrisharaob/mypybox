counter=0
for fl in `ls *.csv`; 
do
    counter=$(($counter+1))
    echo $fl $counter >> list.txt
done
#!/bin/bash

title=$2
outputFolder="images"

for f in $(ls "$1"*all.csv)
do
	file=$(echo $f | sed 's|all.csv$||g')
	date=$(echo $f | sed -r 's|.*_([0-9]+-[0-9]+-[0-9]{4})_.*|\1|')
	echo plot.py $file "\"$title\"" $outputFolder $date
	./plot.py $file "$title" $outputFolder $date 
done


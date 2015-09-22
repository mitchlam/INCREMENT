#!/bin/bash

dt=`date '+%m-%d-%Y_%H-%M-%S'`
echo "$dt"

dataset=$(echo "$1" | sed -r 's|.*/||g; s|\.csv$||g')

outfile="stdout/INCREMENT_${dataset}_${dt}.txt"

echo "./incDriver.py $@ -v 1" > $outfile

echo "" >> $outfile
echo "" >> $outfile

unbuffer ./incDriver.py "$@" -v 1 2>&1 | tee -a $outfile


echo "" >> $outfile
echo "" >> $outfile
echo "" >> $outfile
echo "" >> $outfile

echo "Solver:" >> $outfile
cat solver.prototxt >> $outfile

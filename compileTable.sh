#!/bin/bash


FOLDER="$1/"

echo "Dataset, Algorithm, Initial, Final, Accuracy, Initial, Final, Homogeneity, Initial, Final, Completeness, Initial, Final, V-measure, Initial, Final, JCC"

for f in $(ls "$FOLDER"*.csv)
do
	cat $f
done

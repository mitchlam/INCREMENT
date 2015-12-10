#!/bin/bash

for file in $(ls "$1")
do
	filename="${1}/$file"
	FILE_NAME=$(echo "$file" | sed 's|.*/||g;s|INCREMENT_||g;s|....$||g')
	QUERY_FILE="results/${FILE_NAME}_queries.csv"
	MISC_FILE="results/${FILE_NAME}_misc.csv"
	FINAL_FILE="results/${FILE_NAME}_final.csv"
	ALL_FILE="results/${FILE_NAME}_all.csv"

	echo $FILE_NAME

	./extract_queries.sh "$filename" > $QUERY_FILE
	./extract_initial.sh "$filename" > $MISC_FILE
	./extract_final.sh "$filename" > $FINAL_FILE
	paste $QUERY_FILE $MISC_FILE > $ALL_FILE

done

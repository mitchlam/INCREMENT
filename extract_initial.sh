#!/bin/bash

echo "Alg, Accuracy, Homogeneity, Completeness, V-measure, JCC"

grep "Initial Clustering:" -A 3 "$1" | sed -E ':a;N;$!ba;s|Initial Clustering: (.*)\nInitial: .*\nAccuracy: [0-9]* of [0-9]*: ([0-9]*\.[0-9]*) %\nH: ([0-9]*\.[0-9]*) C: ([0-9]*\.[0-9]*) V: ([0-9]*\.[0-9]*) JCC: ([0-9]*\.[0-9]*)(\n--)?|\1, \2, \3, \4, \5, \6|g'

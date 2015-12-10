#!/bin/bash
 
#args=('data/washpass_final.csv -i pre -S 2'
#      'data/padeaths_final.csv -i pre -S 2'
##      'data/england_final.csv -i pre -S 10'
#
#      'data/faces-names.csv -I -k 10 -S 2'
#      'data/faces-names.csv -I -k 10 -i spectral -S 2'
#      'data/faces-names.csv -I -i active -S 5'
#      'data/faces-names.csv -I -i none -S 2'
#      'data/faces-names.csv -I -k 10 -i mean-shift -S 2'
#      'data/faces-names.csv -I -k 10 -i complete -S 2'
#      
#      'data/faces-poses.csv -I -k 4 -S 2'
#      'data/faces-poses.csv -I -k 4 -i spectral -S 2'
#      'data/faces-poses.csv -I -i active -S 5'
#      'data/faces-poses.csv -I -i none -S 2'
#      'data/faces-poses.csv -I -k 4 -i mean-shift -S 2'
#      'data/faces-poses.csv -I -k 4 -i complete -S 2'
#      
#      'data/faces-eyes.csv -I -k 2 -S 2'
#      'data/faces-eyes.csv -I -k 2 -i spectral -S 2'
#      'data/faces-eyes.csv -I -i active -S 5'
#      'data/faces-eyes.csv -I -i none -S 2'
#      'data/faces-eyes.csv -I -k 2 -i mean-shift -S 2'
#      'data/faces-eyes.csv -I -k 2 -i complete -S 2'
#     
#      'data/faces-expressions.csv -I -k 4 -S 2'
#      'data/faces-expressions.csv -I -k 4 -i spectral -S 2'
#      'data/faces-expressions.csv -I -i active -S 5'
#      'data/faces-expressions.csv -I -i none -S 2'
#      'data/faces-expressions.csv -I -k 4 -i mean-shift -S 2'
#      'data/faces-expressions.csv -I -k 4 -i complete -S 2'
#
#      'data/leaf_1000-images.csv -I -k 60 -S 2'
#      'data/leaf_1000-images.csv -I -k 60 -i spectral -S 2'
#      'data/leaf_1000-images.csv -I -k 60 -i active -S 10'
#      'data/leaf_1000-images.csv -I -k 60 -i none -S 2'
#      'data/leaf_1000-images.csv -I -k 60 -i mean-shift -S 2'
#      'data/leaf_1000-images.csv -I -k 60 -i complete -S 2'
#      
#      'data/leedsbutterfly.csv -I -k 10 -S 2'
#      'data/leedsbutterfly.csv -I -k 10 -i spectral -S 2'
#      'data/leedsbutterfly.csv -I -k 10 -i active -S 10'
#      'data/leedsbutterfly.csv -I -k 10 -i none -S 2'
#      'data/leedsbutterfly.csv -I -k 10 -i mean-shift -S 2'
#      'data/leedsbutterfly.csv -I -k 10 -i complete -S 2'
#      )

#args=('data/faces-expressions.csv -I -k 4 -S 2'
#      'data/faces-expressions.csv -I -k 4 -i spectral -S 2'
#      'data/faces-expressions.csv -I -i active -S 5'
#      'data/faces-expressions.csv -I -i none -S 2'
#      'data/faces-expressions.csv -I -k 4 -i mean-shift -S 2'
#      'data/faces-expressions.csv -I -k 4 -i complete -S 2')


args=('data/leedsbutterfly.csv -I -k 10 -S 4 -C -F'
      'data/leedsbutterfly.csv -I -k 10 -i active -S 10 -C -F'
      'data/leedsbutterfly.csv -I -k 10 -i none -S 4 -C -F'
      )
     

for a in "${args[@]}"
do
	echo ./incDriver.py $a -T
	./run.sh $a -T

done

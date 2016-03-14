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


args=('data/leaf_1000-images.csv -I -k 60 -C -F'
      'data/leaf_1000-images.csv -i spectral -I -k 60 -C -F'
      'data/leaf_1000-images.csv -i complete -I -k 60 -C -F'

      'data/leedsbutterfly.csv -I -k 10 -C -F'
      'data/leedsbutterfly.csv -i spectral -I -k 10 -C -F'
      'data/leedsbutterfly.csv -i complete -I -k 10 -C -F'

      'data/faces-expressions.csv -I -k 4 -C -F'
      'data/faces-expressions.csv -i spectral -I -k 4 -C -F'
      'data/faces-expressions.csv -i complete -I -k 4 -C -F'
      
      'data/faces-names.csv -I -k 10 -C -F'
      'data/faces-names.csv -i spectral -I -k 10 -C -F'
      'data/faces-names.csv -i complete -I -k 10 -C -F'

      'data/faces-poses.csv -I -k 4 -C -F'
      'data/faces-poses.csv -i spectral -I -k 4 -C -F'
      'data/faces-poses.csv -i complete -I -k 4 -C -F'

      'data/faces-eyes.csv -I -k 2 -C -F'
      'data/faces-eyes.csv -i spectral -I -k 2 -C -F'
      'data/faces-eyes.csv -i complete -I -k 2 -C -F'
      )
     

for a in "${args[@]}"
do
	echo ./incDriver.py $a
	./run.sh $a 

done

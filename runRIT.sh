#!/bin/bash
#Code written by Jonathon Romero

#Usage: runRIT.sh feature name

#name of feature
feature=$1
#echo $feature

#pre-name
prename=$2
#echo $name

threaded=$3

module load python/3.7-anaconda3

# cd into iRF yvec directory

cat ${prename}*.pathfile > ${prename}_${feature}.paths
python trimpath.py ${prename}_${feature}.paths
mv ${prename}_${feature}.paths ${prename}_${feature}.paths.old
mv ${prename}_${feature}.paths.fixed ${prename}_${feature}.paths
python preprocessPathsForRIT.py ${prename}_${feature}.paths
ritw ${prename}_${feature}.paths.int.out > ${prename}_${feature}.full
_rit
sort -k1rg ${prename}_${feature}.full_rit > ${prename}_${feature}.full_rit_sort
head -n 20 ${prename}_${feature}.full_rit_sort > ${prename}_${feature}.rit 
python ritEval.py ${prename}_${feature}.importance4 ${feature} $threa
ded 
sort -k3rg ${prename}_${feature}.importance4.effect > ${prename}.importance4.effect_sorted 

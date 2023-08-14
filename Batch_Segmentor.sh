#!/bin/bash
segmentor_script=$1
image=$2

# 1 2 3 8 5 12 14 17 

source ~/anaconda3/etc/profile.d/conda.sh
conda activate manskelab

# conda info


for bone in  2 
do
   echo $bone
#    ((i=i%N)); ((i++==0)) && wait
   python $1 $2 $bone > $bone.txt &
done

wait

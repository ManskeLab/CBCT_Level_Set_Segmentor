#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Automatic Segmentor of Hands in Cone Beam Computed Tomography (CBCT) Images"
   echo
   echo "Requirements:"
   echo "Nvidia graphics card with updated drivers"
   echo "Cuda and Numba libraries installed"
   echo "Manskelab conda environment"
   echo "Linux OS (Preferred)"
   echo
   echo "options:"
   echo "h     OPTIONAL: Print this Help."
   echo "i     Path to input CBCT image."
   echo "o     OPTIONAL: Path to output directory. If not provided, an output directory will be created within the input image's directory."
   echo "b     OPTIONAL: Number of bones to segment in barallel in one batch. Default is 2."
   echo "s     OPTIONAL: Path to python segmentor script. If not provided, the default script will be used."
   echo "p     OPTIONAL: Path to python preprocessing script. If not provided, the default script will be used."
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# Set variables
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR=$SCRIPT_DIR/Segmentation_results
BONE_PER_BATCH=2
SEGMENTOR_SCRIPT=$SCRIPT_DIR
PREPROCESSING_SCRIPT=$SCRIPT_DIR

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts ":hiosp:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      i) # Enter a image path
         IMAGE=$OPTARG;;
      o) # Enter an output directory
         OUTPUT_DIR=$OPTARG/Segmentation_results;;
      b) # Number of bones to batch
         $BONE_PER_BATCH=$OPTARG
      s) # Enter segmentor script location
         SEGMENTOR_SCRIPT=$OPTARG;;
      p) # Enter preproccessing script location
         PREPROCESSING_SCRIPT=$OPTARG;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

mkdi -p $OUTPUT_DIR
# echo "hello $Name!"
# segmentor_script=$1
# preprocessing_script=
# image=$2

# # 1 2 3 8 5 12 14 17 

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate manskelab

# # conda info

# # Get Pre-processed (contrast enhanced image)
# python 


# for bone in  2 
# do
#    echo $bone
# #    ((i=i%N)); ((i++==0)) && wait
#    python $1 $2 $bone > $bone.txt &
# done

# wait

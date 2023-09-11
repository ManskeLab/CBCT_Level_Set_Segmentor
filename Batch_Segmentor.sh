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
   echo "b     OPTIONAL: Number of bones to segment in parallel in one batch. Default is 2."
   echo "s     OPTIONAL: Path to python segmentor script. If not provided, the default script will be used."
   echo "p     OPTIONAL: Path to python preprocessing script. If not provided, the default script will be used."
   echo "c     OPTIONAL: Path to python mask combiner script. If not provided, the default script will be used."
   echo "v     OPTIONAL: Debug flag."
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
SEGMENTOR_SCRIPT=$SCRIPT_DIR/Hand_Segmentor.py
PREPROCESSING_SCRIPT=$SCRIPT_DIR/Bone_Contrast_Enhancer.py
MASK_COMBINER_SCRIPT=$SCRIPT_DIR/Mask_Combiner.py
DEBUG_FLAG=0

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts ":h:i:o:b:s:p:c:v:" option; do
   case $option in
      h) # display Help
         Help
         exit
         ;;
      i) # Enter a image path
         IMAGE=$OPTARG
         ;;
      o) # Enter an output directory
         OUTPUT_DIR=$OPTARG\Segmentation_Results_
         ;;
      b) # Number of bones to batch
         BONE_PER_BATCH=$OPTARG
         ;;
      s) # Enter segmentor script location
         SEGMENTOR_SCRIPT=$OPTARG
         ;;
      p) # Enter preproccessing script location
         PREPROCESSING_SCRIPT=$OPTARG
         ;;
      p) # Enter mask combiner script location
         MASK_COMBINER_SCRIPT=$OPTARG
         ;;
      v) # Debug flag
         DEBUG_FLAG=$OPTARG
         ;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit
         ;;
   esac
done

IMAGE_BASENAME="$(basename "$IMAGE" | sed 's/\(.*\)\..*/\1/')"
OUTPUT_DIR=$OUTPUT_DIR$IMAGE_BASENAME
mkdir -p $OUTPUT_DIR

echo "Outputs will be stored in $OUTPUT_DIR"

# # 1 2 3 8 5 12 14 17 

# source ~/anaconda3/etc/profile.d/conda.sh
conda activate manskelab

# # conda info

# Get Pre-processed (contrast enhanced image)
echo "Preprocessing image..."
echo "python $PREPROCESSING_SCRIPT $IMAGE $OUTPUT_DIR"
python $PREPROCESSING_SCRIPT $IMAGE $OUTPUT_DIR
wait

CONTRAST_ENHANCED_IMAGE=$OUTPUT_DIR/CONTRAST_ENHANCED_$IMAGE_BASENAME.nii
echo "Preprocessed image stored in $CONTRAST_ENHANCED_IMAGE"

echo "Starting batch processes..."
N=$BONE_PER_BATCH 
(
for BONE_NUM in {1..20}; do 
   ((i=i%N)); ((i++==0)) && wait
   python $SEGMENTOR_SCRIPT $CONTRAST_ENHANCED_IMAGE $OUTPUT_DIR $BONE_NUM $DEBUG_FLAG &
done
)
wait

echo "Combining individual bone masks together"
FINAL_OUTPUT_NAME=$OUTPUT_DIR/COMBINED_MASK_$IMAGE_BASENAME.nii

# Combine all masks
python $MASK_COMBINER_SCRIPT $OUTPUT_DIR $FINAL_OUTPUT_NAME

echo "Done!"
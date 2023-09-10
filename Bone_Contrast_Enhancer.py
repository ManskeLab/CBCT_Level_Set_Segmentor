import SimpleITK as sitk
import numpy as np
import os
import argparse
import shutil

import numba as nb
from numba.typed import List
from timeit import default_timer as timer   

@nb.jit((nb.float64[:,:,:])(nb.float64[:,:,:], nb.float64, nb.float64), parallel=True)
def generate_tissue(void_masked_array, tissue_mean, tissue_standard_deviation):
    """
    Generate voxels along a gaussian distribution of soft tissue and add it to the back ground.

    input: 
    - Rough mask of skin-air boundary
    - mean intensity of soft tissue
    - standard deviation of soft tissue

    output:
    - numpy array of image without skin-air boundary 
    """

    random_gauss = []
    for idx in nb.prange(100):

        X = tissue_standard_deviation*np.random.normal() + tissue_mean
        Y = tissue_standard_deviation*np.random.normal() + tissue_mean

        random_gauss.append(X)
        random_gauss.append(Y)

    dimensions = np.shape(void_masked_array)
    
    random_gauss = np.array(random_gauss)
    for slice_idx in nb.prange(dimensions[0]):
        for row in nb.prange(dimensions[1]):
            for col in nb.prange(dimensions[2]):
                if void_masked_array[slice_idx, row, col] == 0:
                    idx = np.random.randint(200)
                    void_masked_array[slice_idx, row, col] = random_gauss[idx]

    return void_masked_array

def contrast_enhancer(hand_image):
    """
    Enhance contrast of bone through iterative smoothing and sharpening.

    input: 
    - sitk image of CBCT scan

    output: 
    - contrast enhanced sitk image
    """
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.LaplacianSharpening(hand_image)
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.LaplacianSharpening(hand_image)
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.LaplacianSharpening(hand_image)
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.LaplacianSharpening(hand_image)
    hand_image = sitk.LaplacianSharpening(hand_image)
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.Median(hand_image)
    hand_image = sitk.Normalize(hand_image)

    return hand_image

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("hand_image_path", type=str, help="Image (path + filename)")
    parser.add_argument("output_dir", type=str, help="Path to ouput directory")
    args = parser.parse_args()

    hand_image_path = args.hand_image_path
    output_dir = args.output_dir

    hand_image = sitk.ReadImage(hand_image_path)
    hand_image = sitk.Normalize(hand_image)

    # mask out entire hand
    hand_image_temp = hand_image*sitk.Cast(hand_image>1, sitk.sitkFloat64)

    # mask out soft tissue in hand
    tissue_mask = sitk.BinaryThreshold(hand_image, 1, 1.4)
    stat_filter = sitk.LabelStatisticsImageFilter()

    stat_filter.Execute(hand_image, tissue_mask)

    # Find mean and SD of soft tissue
    tissue_mean = stat_filter.GetMean(1)
    tissue_standard_deviation = np.sqrt(stat_filter.GetVariance(1))

    hand_image_arr = sitk.GetArrayFromImage(hand_image_temp)

    print("Generating artificial soft tissue...")
    new_tissue_gauss = generate_tissue(hand_image_arr, tissue_mean, tissue_standard_deviation)
    spacing = hand_image.GetSpacing()[0]
    
    hand_image = sitk.GetImageFromArray(new_tissue_gauss)

    print("Smoothing and sharpening...")
    hand_image = contrast_enhancer(hand_image)

    input_filename = os.path.basename(hand_image_path).split('/')[-1]
    sitk.WriteImage(hand_image, os.path.join(output_dir, 'CONTRAST_ENHANCED_'+input_filename))

    return

if __name__ == '__main__':
    main()

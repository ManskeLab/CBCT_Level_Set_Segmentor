import os
import SimpleITK as sitk
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("masks_dir", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    masks_dir = args.masks_dir
    output_path = args.output_path

    combined_mask = False

    for file in os.listdir(masks_dir):
        if not('mask_bone' in file):
            continue
        i = file.split('mask_bone_')[1]
        i = i.split('.nii')[0]
        i = int(i)

        mask = sitk.ReadImage(os.path.join(masks_dir, file))
        mask = mask != 0
        mask = mask*i

        if not combined_mask:
            combined_mask = mask
        else:
            combined_mask = combined_mask | mask

    sitk.WriteImage(combined_mask, output_path)
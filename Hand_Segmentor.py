import SimpleITK as sitk
import numpy as np
import os
import argparse
import shutil
import copy
import time
import math
import random

import numba as nb
from numba.typed import List
from timeit import default_timer as timer   

@nb.extending.overload(np.gradient)
def np_gradient(f):
    def np_gradient_impl(f):
        out_y = np.empty_like(f, np.float64)
        out_y[1:-1] = (f[2:] - f[:-2]) / 2.0
        out_y[0] = f[1] - f[0]
        out_y[-1] = f[-1] - f[-2]
        f = np.transpose(f)
        out_x = np.empty_like(f, np.float64)
        out_x[1:-1] = (f[2:] - f[:-2]) / 2.0
        out_x[0] = f[1] - f[0]
        out_x[-1] = f[-1] - f[-2]
        out_x = np.transpose(out_x)
        return [out_x, out_y]

    return np_gradient_impl

@nb.jit((nb.float64[:])(nb.int8, nb.float64), nopython=True)
def gaussian_kernel(l, sig):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / np.sum(gauss)

@nb.jit(nopython = True)
def divergence_operator(A):
    """
    
    """
    return np.gradient(A[0])[0] + np.gradient(A[1])[1]

@nb.jit((nb.float64[:,:])(nb.float64[:,:], nb.float64[:]), nopython = True, parallel=True)
def conv2d(image, kernel):
    input_rows = image.shape[0]
    input_cols = image.shape[1]
    kernel_size = len(kernel)
    out_row = input_rows + kernel_size - 1
    out_col = input_cols + kernel_size - 1

    out = np.zeros((out_row,out_col))

    for i in nb.prange(input_cols):
        out[:,i] = np.convolve(image[:,i], kernel)
    for i in nb.prange(out_row):
        out[i,:]=np.convolve(out[i,0:input_cols], kernel)

    startx = out_row//2 - input_rows//2
    starty = out_col//2 - input_cols//2
    out = out[startx:startx+input_rows, starty:starty+input_cols]

    return out

@nb.jit((nb.float64[:,:])(nb.float64[:,:]), nopython=True)
def del2(M):
    dx = 1
    dy = 1
    rows, cols = M.shape
    dx = dx * np.ones ((1, cols - 1))
    dy = dy * np.ones ((rows-1, 1))
    mr, mc = M.shape
    D = np.zeros ((mr, mc))
    if (mc >= 3):
        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:,0] * dx[:,1])
        D[:, mc-1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc-1])/ (dx[:,mc - 3] * dx[:,mc - 2])
        tmp1 = D[:, 1:mc - 1] 
        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
        tmp3 = np.kron (dx[:,0:mc -2] * dx[:,1:mc - 1], np.ones ((mr, 1)))
        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3
    if (mr >= 3):
        D[0, :] = D[0,:]  + (M[0, :] - 2 * M[1, :] + M[2, :] ) / (dy[0,:] * dy[1,:])
        D[mr-1, :] = D[mr-1, :] + (M[mr-3,:] - 2 * M[mr-2, :] + M[mr-1, :])/(dy[mr-3,:] * dx[:,mr-2])
        tmp1 = D[1:mr-1, :] 
        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr-2, :])
        tmp3 = np.kron (dy[0:mr-2,:] * dy[1:mr-1,:], np.ones ((1, mc)))
        D[1:mr-1, :] = tmp1 + tmp2 / tmp3
    return D / 4

def main():
    global temp_dir
    global debug_flag
    
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("hand_image_path", type=str, help="Image (path + filename)")
    parser.add_argument("output_dir", type=str, nargs='?', default=os.path.realpath(os.path.dirname(__file__)), help="Path to outpt directory")
    parser.add_argument("bone_num", type=int, help="Bone to segment")
    parser.add_argument("debug_flag", type=int, nargs='?', default=0, help="Enable debug")
    args = parser.parse_args()

    hand_image_path = args.hand_image_path
    output_dir = args.output_dir

    bone_num = args.bone_num
    debug_flag = args.debug_flag
    print("Processing bone #{}".format(bone_num))
    if debug_flag:
        print("Debugging is enabled.")
    
    hand_image = sitk.ReadImage(hand_image_path)
    print("Image read!")

    if debug_flag:
        temp_dir = os.path.join(output_dir, 'intermediate_files_bone_{}'.format(bone_num))
        # if(os.path.isdir(temp_dir)):
        #     shutil.rmtree(temp_dir)
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        print("Intermediary images will be stored in {}".format(temp_dir))

    # Run main levelset process function
    mask = region_based_levelset(sitk.Cast(hand_image, sitk.sitkFloat64), bone_num)
    sitk.WriteImage(mask, os.path.join(output_dir, 'mask_bone_{}.nii'.format(bone_num)))

    return

@nb.jit((nb.float64[:,:])(nb.float64[:,:], nb.float64[:,:], nb.int8), nopython=True)
def calc_levelset(levelset, image, iter):

    image = image+1
    image = image*100/np.max(image)
    iteration_time = 8
    if(iter > 100):
        iteration_time = 2
    alpha = 6*iteration_time
    beta = 3
    k = 0.1
    mu = 1
    epsilon = 1.6
    lam = 10*iteration_time

    levelset[0,0] = levelset[2,2]
    levelset[0,-1] = levelset[2,-3]
    levelset[-1,0] = levelset[-3,2]
    levelset[-1,-1] = levelset[-3,-3]

    levelset[0,1:-2] = levelset[2,1:-2]
    levelset[-1,1:-2] = levelset[-3,1:-2]

    levelset[1:-2, 0] = levelset[1:-2, 2]
    levelset[1:-2,-1] = levelset[1:-2,-3]

    gkern = gaussian_kernel(30, epsilon)

    in_mask = np.where(levelset<=0.0, 1.0, 0.0)
    out_mask = np.where(levelset>0.0, 1.0, 0.0)

    img_smooth = conv2d(image, gkern)

    # find local means
    in_phi = image*in_mask
    out_phi = img_smooth*out_mask

    in_phi_gauss = conv2d(img_smooth*in_mask, gkern)
    # x = conv2d(in_mask, gkern)

    # print(in_phi_gauss[100,179:189])
    # print(x[100,179:189])
    in_mean = in_phi_gauss/(conv2d(in_mask, gkern)+1e-10)
    
    out_phi_gauss = conv2d(img_smooth*out_mask, gkern)
    out_mean = out_phi_gauss/(conv2d(out_mask, gkern)+1e-10)


    gradient_arr = np.gradient(img_smooth)
    gradient_info = np.square(gradient_arr[0]) + np.square(gradient_arr[1])

    gradient_phi = np.gradient(levelset)

    s = np.sqrt(np.square(gradient_phi[0]) + np.square(gradient_phi[1]))

    # speed coefficients
    v = alpha*np.exp(-beta*np.absolute(in_mean-out_mean))+k
    # v = alpha
    # print(max(v))
    # print(np.mean(v))

    # in_phi_variance = conv2d(np.square((in_phi-in_mean)*in_mask), gkern)
    # out_phi_variance = conv2d(np.square((out_phi-out_mean)*out_mask), gkern)
    g = 1/(1+(gradient_info))
    # g = 1/(1+(gradient_info/(in_phi_variance+out_phi_variance+1)))

    # x = 1/(1+(gradient_info))
    # y = in_mean
    # print(gradient_info[100,179:189])
    # print(y[100,179:189])
    # print(in_phi_gauss[100,179:189])
    # print(in_phi_variance[100,179:189])
    # print(out_phi_variance[100,179:189])
    # print(x[100,179:189])
    # print(g[100,179:189])
    # g = 1/(1+(gradient_info))


    mask_ps = np.logical_and(s>=0, s<=1)
    ps =  mask_ps*np.sin(2*np.pi*s)/(2*np.pi) + (1-mask_ps)*(s-1)

    dps = ((s!=0)*ps)/((s!=0)*s + (s==0)) + (s==0)
    # max_idx = np.where(dps == np.max(dps))
    # print(max_idx)
    # print(dps[max_idx[0]])
    # print(ps[max_idx[0]])
    # print(s[max_idx[0]])

    distance_regularization_energy = divergence_operator([dps*gradient_phi[0], dps*gradient_phi[1]])
    # x = np.all((np.absolute(dps)<=1))
    # if(x == False):
    #     print(np.all((np.absolute(s)<=1)))
    # print(np.mean(dps))
    # print(np.mean(gradient_phi[0]))
    # print(np.mean(4*del2(levelset)))

    area_energy_term = (1/(2*epsilon))*(1+np.cos((np.pi * levelset/epsilon)))*(np.absolute(levelset)<=epsilon)

    gradient_g = np.gradient(g)
    edge_energy_term = area_energy_term*(((gradient_g[0]*gradient_phi[0]) + (gradient_g[1]*gradient_phi[1])) + g*s*divergence_operator([gradient_phi[0]/(s+1e-10), gradient_phi[1]/(s+1e-10)]))/(s+1e-10)
    # edge_energy_term = area_energy_term*(divergence_operator((g*gradient_phi[0]/(s+1e-10)), (g*gradient_phi[1]/(s+1e-10))))
    # print(levelset[120,179:189])
    # print(distance_regularization_energy[120,179:189])
    # print(area_energy_term[120,179:189])
    # print(edge_energy_term[120,179:189])


    if(np.isnan(distance_regularization_energy).any()):
        print('dist')
    # if(np.isnan(v).any()):
    #     print('v')
    if(np.isnan(g).any()):
        print('g')
    if(np.isnan(edge_energy_term).any()):
        print('edge')
    if(np.isnan(area_energy_term).any()):
        print('area')
    a = mu*distance_regularization_energy
    b = lam*edge_energy_term
    c = area_energy_term*g*v
    print(np.mean(a))
    print(np.mean(b))
    print(np.mean(c))
    print(np.mean(g))
    change = a+b+c
    # print(change[112,213])
    if(np.isnan(change).any()):
        raise OverflowError
    # levelset_updated = levelset+change

    # levelset_mask = levelset_updated<=0
    # image_masked = levelset_mask*image

    return change

def set_mask_value(image, mask, value):
    mask = sitk.Cast(mask, sitk.sitkInt8)
    return sitk.Cast(sitk.Cast(image, sitk.sitkInt8) *
                     sitk.InvertIntensity(mask, maximum=1.0) + 
                     mask*float(value), sitk.sitkInt8)
        
def region_based_levelset(hand_image, bone_num):

    # Parse through coronal 
    img_slices = hand_image.GetHeight()

    # normalize image
    hand_image = sitk.Normalize(hand_image)
    # sitk.WriteImage(hand_image, os.path.join(temp_dir, 'hand.nii'))

    erode_filter = sitk.BinaryErodeImageFilter()
    dilate_filter = sitk.BinaryDilateImageFilter()
    
    # add canny and gradient magnitude filters together
    canny_edge = sitk.CannyEdgeDetection(hand_image, lowerThreshold=0.7, upperThreshold=0.99, variance = 3*[0.5*hand_image.GetSpacing()[0]])
    gradient_edge = sitk.Cast(sitk.GradientMagnitude(hand_image-(2*canny_edge)), sitk.sitkFloat64)

    gradient_edge_thresh = sitk.BinaryThreshold(gradient_edge, 1.3, 9999, 1, 0)
    # erode_filter.SetKernelRadius(1)
    # gradient_edge_thresh = erode_filter.Execute(gradient_edge_thresh)
    hand_thresh = sitk.Cast(canny_edge,sitk.sitkUInt8) | gradient_edge_thresh


    img_conn = sitk.ConnectedComponent(hand_thresh, hand_thresh)
    img_conn = sitk.RelabelComponent(img_conn, sortByObjectSize=True)
    if debug_flag:
        sitk.WriteImage(gradient_edge, os.path.join(temp_dir, 'gradient_edge.nii'))
        sitk.WriteImage(gradient_edge_thresh, os.path.join(temp_dir, 'gradient_edge_thresh.nii'))
        sitk.WriteImage(canny_edge, os.path.join(temp_dir, 'canny_edge.nii'))
        sitk.WriteImage(hand_thresh, os.path.join(temp_dir, 'thresh_init.nii'))
        sitk.WriteImage(img_conn, os.path.join(temp_dir, 'conn.nii'))

    # if bone_num == 6:
    #     dilate_filter.SetKernelRadius(70)
    #     erode_filter.SetKernelRadius(68)
    # elif bone_num == 7:
    #     dilate_filter.SetKernelRadius(35)
    #     erode_filter.SetKernelRadius(34)
    # elif bone_num == 13:
    #     dilate_filter.SetKernelRadius(17)
    #     erode_filter.SetKernelRadius(17)
    # elif bone_num == 16:
    #     dilate_filter.SetKernelRadius(25)
    #     erode_filter.SetKernelRadius(25)
    # else:
    #     dilate_filter.SetKernelRadius(70)
    #     erode_filter.SetKernelRadius(68)

    dilate_filter.SetKernelRadius(70)
    erode_filter.SetKernelRadius(68)

    thresh_bone = 1 * (img_conn == bone_num)

    thresh_bone_dilated = dilate_filter.Execute(thresh_bone)
    thresh_bone_filled = sitk.BinaryFillhole(thresh_bone_dilated)
    thresh_bone = erode_filter.Execute(thresh_bone_filled)

    if debug_flag:
        sitk.WriteImage(thresh_bone_dilated, os.path.join(temp_dir, 'dilated_{}.nii').format(bone_num))
        sitk.WriteImage(thresh_bone_filled, os.path.join(temp_dir, 'filled_{}.nii').format(bone_num))
        sitk.WriteImage(thresh_bone, os.path.join(temp_dir, 'eroded_{}.nii').format(bone_num))

    # return thresh_bone
    full_hand_mask = hand_image*0
    full_hand_mask = sitk.Cast(full_hand_mask, sitk.sitkUInt8)
    for idx in range(0, img_slices):
        hand_slice = hand_image[:,idx,:]
        hand_slice= sitk.GetArrayFromImage(hand_slice)
        max_iter = 500

        bone_mask = np.zeros_like(hand_slice)
        # for thresh_bone_idx in range(bone_nums):
        # thresh_bone = individual_thesh_bones[thresh_bone_idx]
        thresh_bone_slice = thresh_bone[:,idx,:]
        levelset = sitk.GetArrayFromImage(sitk.Cast(thresh_bone_slice, sitk.sitkFloat64))

        one_count = np.count_nonzero(levelset)

        if(one_count<5):
            continue
        elif(one_count<50):
            max_iter = 100
        elif(one_count<100):
            max_iter = 200
        elif(one_count<200):
            max_iter = 400

        # levelset = np.ones(levelset.shape)*2
        levelset[levelset==0] = 2
        levelset[levelset==1] = -2

        i = 0
        counter = 0
        while(i<max_iter):
            start_time = time.time()
            # image_copy = hand_image.copy()

            # print('Mean Inside Bounding box: {}'.format(mean_in_new))
            # print('Mean Outside Bounding box: {}'.format(mean_out_new))
            # print(type(hand_image[0][0]))
            levelset_change = calc_levelset(levelset, hand_slice, i)
            levelset = levelset + levelset_change

            x = np.mean(levelset_change)

            print(x)

            if x < 0:
                counter += 1
                if counter == 3:
                    break

            # if not levelset_change.any():
            #     counter += 1
            #     if counter == 15:
            #         break
            # else:
            #     counter = 0
            #     levelset = levelset + levelset_change
            # image_cropped = sitk.GetImageFromArray(image_cropped)

            # levelset_new = levelset.copy()

            # if i%10 == 0:
                # if levelset_old
            #     sitk.WriteImage(image_cropped, os.path.join(temp_dir, 'region_inv'+str(i)+'.nii'))
            i += 1

            # print('Mean Inside Bounding box: {}'.format(mean_in_new))
            # print('Mean Outside Bounding box: {}'.format(mean_out_new))

            # if mean_in_new-mean_out_new > 2.2:
            #     break

            time_elapsed = time.time() - start_time
            print('Bone #{}, slice #{}. Time for iteration #{}: {}s...'.format(bone_num, idx, i, time_elapsed))
            print('************************************')

        bone_mask = bone_mask + 1.0*(levelset<=0)
        bone_mask = sitk.GetImageFromArray(bone_mask)
        bone_mask = sitk.Cast(bone_mask, sitk.sitkUInt8)
        full_hand_mask[:, idx, :] = sitk.BinaryFillhole(bone_mask)
        # sitk.WriteImage(full_hand_mask, os.path.join(temp_dir, 'region_inv'+str(idx)+'.nii'))

    return full_hand_mask

temp_dir = ''
debug_flag = 0
if __name__ == '__main__':
    main()
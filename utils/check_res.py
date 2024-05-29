# import nibabel as nib
# import os
# import statistics


# data_folder = "/data/crossmoda2023/data/crossmoda23_validation"
# # data_folder = "/data/crossmoda2023/data/crossmoda23_training/TrainingTarget"

# x_resolutions = []
# y_resolutions = []
# z_resolutions = []

# for file in os.listdir(data_folder):
#     if file.endswith(".nii.gz"):
#         img = nib.load(os.path.join(data_folder, file))
#         hdr = img.header
#         x_resolutions.append(hdr.get_zooms()[0])
#         y_resolutions.append(hdr.get_zooms()[1])
#         z_resolutions.append(hdr.get_zooms()[2])

# x_mean = sum(x_resolutions) / len(x_resolutions)
# y_mean = sum(y_resolutions) / len(y_resolutions)
# z_mean = sum(z_resolutions) / len(z_resolutions)

# x_median = statistics.median(x_resolutions)
# y_median = statistics.median(y_resolutions)
# z_median = statistics.median(z_resolutions)

# print(x_mean, y_mean, z_mean)
# print(x_median, y_median, z_median)


# # training
# # 0.4275480818950524 0.4459703705068362 0.9819192396382154
# # 0.41015625 0.41015625 1.0

# # validation
# # 0.4262716230005026 0.44461470376700163 1.00717766272525
# # 0.41015625 0.41015625 1.0


import SimpleITK as sitk
from monai.transforms import *
import os
import os.path as osp
from tqdm import tqdm
from glob import glob
import nibabel as nib

# Set the input and output folders

input_folder = "/data/crossmoda2023/data/crossmoda23_training/TrainingSourceImage"
output_folder = "/data/crossmoda2023/data/crossmoda23_training/TrainingSource_resample"


# def resample1(data_dict, out_spacing):    
#     return Compose([
#         LoadImaged(keys=['image', 'label']),
        
#         AddChanneld(keys=["image", "label"]),
        
#         Orientationd(keys=["image", "label"], axcodes="LPS"), 
#         CropForegroundd(keys=["image", "label"], source_key='image'),

#         Spacingd(
#             keys=['image', 'label'], 
#             pixdim=out_spacing, 
#             mode=("bilinear", "nearest")),
        
#         SaveImaged(
#             keys=['image'], 
#             output_dir="/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceImage", 
#             output_postfix='', 
#             output_ext='.nii.gz', 
#             resample=False,
#             separate_folder=False,
#             print_log=False),
        
#         SaveImaged(
#             keys=['label'], 
#             output_dir="/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceLabel", 
#             output_postfix='', 
#             output_ext='.nii.gz', 
#             resample=False,
#             separate_folder=False,
#             print_log=False),
        
#     ])(data_dict)


# img_paths = glob('/data/crossmoda2023/data/crossmoda23_training/TrainingSourceImage' + '/*.*')
# with tqdm(total=len(img_paths)) as pbar:
#     for path in img_paths:
#         # if 'crossmoda2023_ukm_12_ceT1' not in path:
#         #     continue
#         msk_path = path.replace('ceT1', 'Label').replace('TrainingSourceImage', 'TrainingSourceLabel')
#         data_dict = {'image': path, 'label': msk_path}

#         img = nib.load(path)
#         header = img.header
#         z_res = header["pixdim"][3]
        
#         resample1(data_dict, (0.41015625, 0.41015625, z_res))  
#         pbar.update(1)



# def resample2(data_dict, out_spacing):    
#     return Compose([
#         LoadImaged(keys=['image']),
        
#         AddChanneld(keys=["image"]),
#         Orientationd(keys=["image"], axcodes="LPS"), 
#         CropForegroundd(keys=["image"], source_key='image'),
#         Spacingd(
#             keys=['image'], 
#             pixdim=out_spacing, 
#             mode=("bilinear")),
        
#         SaveImaged(
#             keys=['image'], 
#             output_dir="/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingTarget", 
#             output_postfix='', 
#             output_ext='.nii.gz', 
#             resample=False,
#             separate_folder=False,
#             print_log=False),
        
#     ])(data_dict)


# img_paths = glob('/data/crossmoda2023/data/crossmoda23_training/TrainingTarget' + '/*.*')
# with tqdm(total=len(img_paths)) as pbar:
#     for path in img_paths:
#         data_dict = {'image': path}
#         img = nib.load(path)
#         header = img.header
#         z_res = header["pixdim"][3]
        
#         resample2(data_dict, (0.41015625, 0.41015625, z_res))  
#         pbar.update(1)

import numpy as np
from monai.transforms import *
from tqdm import tqdm
import SimpleITK as sitk

import numpy as np
import matplotlib.pyplot as plt

# # Load the npz file
# path = '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceImage_2D/crossmoda2023_ukm_5_ceT1_slice_7.npz'
# data = np.load(path)

# # Extract the 2D image array
# image = data['data']

# # Plot the image using Matplotlib
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.show()


# 3D -> 2D saving

# def save_slices_to_npz(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     with tqdm(total=len(os.listdir(input_folder))) as pbar:
#         for filename in os.listdir(input_folder):
#             if filename.endswith('.nii.gz'):
#                 # Load the NIfTI file
#                 nii_file_path = os.path.join(input_folder, filename)
#                 nii_image = nib.load(nii_file_path)
#                 data = nii_image.get_fdata()

#                 # cropping  
#                 # x_min, x_max = int(data.shape[0]*3/16), int(data.shape[0]*13/16)
#                 # y_min, y_max = int(data.shape[1]*3/16), int(data.shape[1]*13/16)
#                 # data = data[x_min:x_max, y_min:y_max, :]
#                 # cur_xx = x_max - x_min
#                 # cur_yy = y_max - y_min
                
#                 # xx.append(cur_xx)
#                 # yy.append(cur_yy)

#                 # transform = Compose([
#                 #     NormalizeIntensity(nonzero=False),
#                 #     ScaleIntensityRangePercentiles(lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#                 #     ])

#                 # data = transform(data)
                
#                 for z_index in range(data.shape[2]):
#                     slice_data = np.transpose(data[:, :, z_index])
#                     output_filename = f'{os.path.splitext(filename)[0]}_slice_{z_index}.npz'.replace('.nii', '')
#                     output_file_path = os.path.join(output_folder, output_filename)
#                     np.savez(output_file_path, data=slice_data, slice_id=z_index)

#                 pbar.update(1)


# save_slices_to_npz(
#     '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceLabel', 
#     '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceLabel_2D')


# import ants

# mi = ants.image_read('/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceImage/crossmoda2023_ukm_42_ceT1.nii.gz')
# fi = ants.image_read('/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingTarget/crossmoda2023_ukm_72_T2.nii.gz')
# df = ants.registration(fixed=fi, moving=mi, type_of_transform = 'Rigid' )
# df_img = ants.apply_transforms(fixed=fi, moving=mi, transformlist=df['fwdtransforms'])
# ants.image_write(df_img, 'reg.nii.gz')

# import pdb

# def save_slices_to_npz(input_folder, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     ldn, etz, ukm = [], [], []

#     with tqdm(total=len(os.listdir(input_folder))) as pbar:
#         for filename in os.listdir(input_folder):
#             if filename.endswith('.nii.gz'):
#                 # Load the NIfTI file
#                 nii_file_path = os.path.join(input_folder, filename)
#                 nii_image = nib.load(nii_file_path)
#                 data = nii_image.get_fdata()

#                 if 'ldn' in filename:
#                     ldn.append(data.shape[2])
#                 if 'etz' in filename:
#                     etz.append(data.shape[2])
#                 if 'ukm' in filename:
#                     ukm.append(data.shape[2])

#                 pbar.update(1)

#     pdb.set_trace()



# 3D -> 2D saving

def save_slices_to_npz(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with tqdm(total=len(os.listdir(input_folder))) as pbar:
        for filename in os.listdir(input_folder):
            if filename.endswith('.nii.gz'):
                # Load the NIfTI file
                nii_file_path = os.path.join(input_folder, filename)
                nii_image = nib.load(nii_file_path)
                data = nii_image.get_fdata()

                # cropping  
                # x_min, x_max = int(data.shape[0]*3/16), int(data.shape[0]*13/16)
                # y_min, y_max = int(data.shape[1]*3/16), int(data.shape[1]*13/16)
                # data = data[x_min:x_max, y_min:y_max, :]
                # cur_xx = x_max - x_min
                # cur_yy = y_max - y_min
                
                # xx.append(cur_xx)
                # yy.append(cur_yy)

                # transform = Compose([
                #     NormalizeIntensity(nonzero=False),
                #     ScaleIntensityRangePercentiles(lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
                #     ])

                # data = transform(data)
                
                for z_index in range(data.shape[2]):
                    slice_data = np.transpose(data[:, :, z_index])
                    output_filename = f'{os.path.splitext(filename)[0]}_slice_{z_index}.npz'.replace('.nii', '')
                    output_file_path = os.path.join(output_folder, output_filename)
                    np.savez(output_file_path, data=slice_data, slice_id=z_index)

                pbar.update(1)

save_slices_to_npz(
    '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingTarget', 
    '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingTarget_2D')
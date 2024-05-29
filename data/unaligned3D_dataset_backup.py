import os.path
from glob import glob
from data.base_dataset import BaseDataset
# from base_dataset import BaseDataset
import random
import numpy as np
import torch
from monai.transforms import *


class Unaligned3DDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, 'srcImageROI') 
            self.dir_A_msk = os.path.join(opt.dataroot, 'srcLabelROI') 
            self.dir_B = os.path.join(opt.dataroot, 'tgtImageROI')  
            self.A_img_paths = sorted(glob(self.dir_A + '/*.nii.gz'))
            self.A_msk_paths = [path.replace('Image', 'Label').replace('ceT1', 'Label') for path in self.A_img_paths]   
            self.A_edge_paths = [path.replace('srcLabelROI', 'srcEdgeV4ROI') for path in self.A_msk_paths]   # check version!
            self.B_img_paths = sorted(glob(self.dir_B + '/*.nii.gz'))

            # self.dir_A = os.path.join(opt.dataroot, 'srcImageROI_thin') 
            # self.dir_A_msk = os.path.join(opt.dataroot, 'srcLabelROI_thin') 
            # self.dir_B = os.path.join(opt.dataroot, 'tgtImageROI_thin')  
            # self.A_img_paths = sorted(glob(self.dir_A + '/*.nii.gz'))
            # self.A_msk_paths = [path.replace('Image', 'Label').replace('ceT1', 'Label') for path in self.A_img_paths]   
            # self.A_edge_paths = [path.replace('srcLabelROI_thin', 'srcEdgeV4ROI') for path in self.A_msk_paths]   # check version!
            # self.B_img_paths = sorted(glob(self.dir_B + '/*.nii.gz'))

        if opt.phase == "test":
            pass

        self.A_size = len(self.A_img_paths)  
        self.B_size = len(self.B_img_paths)
        self.transform = modaTransform()
        self.phase = opt.phase

    def __getitem__(self, index):
        A_img_path = self.A_img_paths[index % self.A_size] 
        A_msk_path = self.A_msk_paths[index % self.A_size] 
        A_edge_path = self.A_edge_paths[index % self.A_size] 

        if self.opt.serial_batches:   
            index_B = index % self.B_size
        else:   
            index_B = random.randint(0, self.B_size - 1)
        B_img_path = self.B_img_paths[index_B]
        data_dict = {'A': A_img_path, 'B': B_img_path, 'A_msk': A_msk_path, 'A_edge': A_edge_path, 'A_paths': A_img_path, 'B_paths': B_img_path}
        if self.phase == 'train':
            data_dict = self.transform.train(data_dict)
        elif self.phase == 'test':
            pass  # inference code is not implemented here
        return data_dict

    def __len__(self):
        return max(self.A_size, self.B_size)


class GetCoded(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        if 'ukm' in data['B_paths']:
            data['code'] = torch.tensor([0,0,1])
        elif 'etz' in data['B_paths']:
            data['code'] = torch.tensor([0,1,0])
        elif 'ldn' in data['B_paths']:
            data['code'] = torch.tensor([1,0,0])
        return data


class NormalizeForegroundd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            img = data[k]
            _mean = img[img!=0].mean()
            _std = img[img!=0].mean()
            data[k] = (data[k] - _mean)/_std
        return data


class AdjustEdged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['A_edge'][data['A_edge']==2] = 10
        data['A_edge'][data['A_edge']==1] = 2 
        return data


class InvertT1d(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['A'] = -data['A']
        data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
        # data['Inv_A'][(data['A_msk']==1) | (data['A_msk']==2)] = -1 
        return data


class modaTransform(object):
    def __init__(self):
        self.train = Compose([        
            GetCoded(keys='B_paths'),
            LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
            AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
            # RandAdjustContrastd(keys=['A', 'B'], prob=0.2, gamma=(0.8, 1.2)),
            RandAdjustContrastd(keys='A', prob=0.4, gamma=(0.8, 1.2)),
            RandAdjustContrastd(keys='B', prob=0.1, gamma=(0.8, 1.2)),
            NormalizeForegroundd(keys=['A', 'B']),
            ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-0.2, b_max=1, clip=True, relative=False),
            ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
            InvertT1d(keys=['A']),
            RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
            RandFlipd(keys='B', prob=0.5, spatial_axis=0),
            RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
            AdjustEdged(keys=['A_edge']),
            CastToTyped(keys=['A', 'B', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.uint8, np.uint8)),
            ToTensord(keys=['A', 'B', 'A_msk', 'A_edge']),])


# class InvertT1d(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         # data['A'] = -data['A']
#         data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
#         data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
#         return data



# class modaTransform(object):
#     def __init__(self):
#         self.train = Compose([        
#             GetCoded(keys='B_paths'),
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             # RandAdjustContrastd(keys=['A', 'B'], prob=0.2, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys='A', prob=0.4, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys='B', prob=0.1, gamma=(0.8, 1.2)),
#             NormalizeForegroundd(keys=['A', 'B']),
#             ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
#             AdjustEdged(keys=['A_edge']),
#             CastToTyped(keys=['A', 'B', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'A_msk', 'A_edge']),])



# class tran(object):
#     def __init__(self):
#         self.train = Compose([        
#             GetCoded(keys='B_paths'),
#             LoadImaged(keys=['A', 'B', 'A_msk']),
#             AddChanneld(keys=['A', 'B', 'A_msk']),
#             RandAdjustContrastd(keys=['A', 'B'], prob=1, gamma=(0.8, 1.2)),
#             NormalizeForegroundd(keys=['A', 'B']),
#             ScaleIntensityRangePercentilesd(keys=['A', 'B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandFlipd(keys=['A', 'B'], prob=0, spatial_axis=0),
#             # RandSpatialCropd(keys=['A', 'B', 'A_msk'], roi_size=(256, 144, 12), random_center=True, random_size=False),
#             CastToTyped(keys=['A', 'B', 'A_msk'], dtype=(np.float32, np.float32, np.uint8)),
#             SaveImaged(
#                     keys=['A'], 
#                     output_dir=f'/data/crossmoda2023/data/crossmoda23_training', 
#                     output_postfix='', 
#                     output_ext='.nii.gz', 
#                     resample=False,
#                     separate_folder=False,
#                     print_log=False)])


# if __name__ == "__main__":

#     data_dict = {
#         'A': '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_6_ceT1.nii.gz', 
#         'B': '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_106_T2.nii.gz', 
#         'A_msk': '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_etz_1_Label.nii.gz',
#         'A_paths': '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_1_ceT1.nii.gz',
#         'B_paths': '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_106_T2.nii.gz'}

#     transform = tran().train
#     output = transform(data_dict)

    # Load the npz file
    # path = '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingTarget_2D/crossmoda2023_etz_106_T2_slice_4.npz'
    # image = output['B'][0, ...]

    # # Plot the image using Matplotlib
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()
import os.path
from glob import glob
# from data.base_dataset import BaseDataset
from base_dataset import BaseDataset  # debugging
import random
import numpy as np
import torch
from monai.transforms import *


class SIDDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        num_samples = 4

        if opt.phase == 'train':
            self.dir_B = os.path.join(opt.dataroot, 'tgtImageROI')  
            self.B_img_paths = sorted(glob(self.dir_B + '/*.nii.gz'))

        self.B_size = len(self.B_img_paths)
        self.transform = SIDTransform(num_sampels=num_samples)
        self.phase = opt.phase

    def __getitem__(self, index):
        if self.opt.serial_batches:   
            index_B = index % self.B_size
        else:   
            index_B = random.randint(0, self.B_size - 1)
        B_img_path = self.B_img_paths[index_B]
        if self.phase == 'train':
            content_dict = self.transform.content({'B': B_img_path})
            style_dict = self.transform.style({'B': B_img_path})
        return content_dict + style_dict

    def __len__(self):
        return self.B_size


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



class SIDTransform(object):
    def __init__(self, num_samples):
        self.content = Compose([        
            LoadImaged(keys='B'),
            AddChanneld(keys='B'),
            NormalizeForegroundd(keys=['B']),
            ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
            RandFlipd(keys='B', prob=0.5, spatial_axis=0),
            RandSpatialCropSamplesd(keys='B', roi_size=(256, 144, 8), num_samples=num_samples, random_center=True, random_size=False),
            CastToTyped(keys='B', dtype=np.float32),
            ToTensord(keys='B'),])

        self.style = Compose([        
            LoadImaged(keys='B'),
            AddChanneld(keys='B'),
            NormalizeForegroundd(keys='B'),
            ScaleIntensityRangePercentilesd(keys='B', lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
            RandFlipd(keys='B', prob=0.5, spatial_axis=0),
            RandSpatialCropd(keys='B', roi_size=(256, 144, 8), random_center=True, random_size=False),
            RandSpatialCropSamplesd(keys='B', roi_size=(256, 144, 8), num_samples=num_samples, random_center=True, random_size=False),
            RandAdjustContrastd(keys='B', prob=1, gamma=(0.5, 1.5)),
            CastToTyped(keys='B', dtype=np.float32),
            ToTensord(keys='B'),])


if __name__ == "__main__":

    num_samples = 4
    data_dict = {'B': '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_106_T2.nii.gz'}
    transform1 = SIDTransform(num_samples=num_samples).content
    output1 = transform1(data_dict)

    data_dict = {'B': '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_106_T2.nii.gz'}
    transform2 = SIDTransform(num_samples=num_samples).style
    output2 = transform2(data_dict)

    breakpoint()
    
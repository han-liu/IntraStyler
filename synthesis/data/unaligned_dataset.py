import os.path
from glob import glob
from data.base_dataset import BaseDataset
# from base_dataset import BaseDataset
import random
import numpy as np
import torch
from monai.transforms import *


class UnalignedDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, 'resampled_TrainingSourceImage_2D') 
            self.dir_A_msk = os.path.join(opt.dataroot, 'resampled_TrainingSourceLabel_2D') 
            self.dir_B = os.path.join(opt.dataroot, 'resampled_TrainingTarget_2D')  
            self.A_img_paths = sorted(glob(self.dir_A + '/*.npz'))
            self.A_msk_paths = [path.replace('Image', 'Label').replace('ceT1', 'Label') for path in self.A_img_paths]   
            self.B_img_paths = sorted(glob(self.dir_B + '/*.npz'))

        if opt.phase == "test":
            self.dir_A = os.path.join(opt.dataroot, 'resampled_TrainingSourceImage') 
            self.dir_B = os.path.join(opt.dataroot, 'resampled_TrainingTarget_2D')  
            self.A_img_paths = sorted(glob(self.dir_A + '/*.nii.gz'))
            self.A_msk_paths = [path.replace('Image', 'Label').replace('ceT1', 'Label') for path in self.A_img_paths]   
            self.B_img_paths = sorted(glob(self.dir_B + '/*.npz'))

        self.A_size = len(self.A_img_paths)  
        self.B_size = len(self.B_img_paths)
        self.transform = modaTransform()
        self.phase = opt.phase

    def __getitem__(self, index):
        A_img_path = self.A_img_paths[index % self.A_size] 
        A_msk_path = self.A_msk_paths[index % self.A_size] 
        if self.opt.serial_batches:   
            index_B = index % self.B_size
        else:   
            index_B = random.randint(0, self.B_size - 1)
        B_img_path = self.B_img_paths[index_B]
        data_dict = {'A': A_img_path, 'B': B_img_path, 'A_msk': A_msk_path, 'A_paths': A_img_path, 'B_paths': B_img_path}
        if self.phase == 'train':
            data_dict = self.transform.train(data_dict)
        elif self.phase == 'test':
            data_dict = self.transform.infer(data_dict)
        return data_dict

    def __len__(self):
        return max(self.A_size, self.B_size)


class LoadNumpyd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            data[k] = dict(np.load(data[k], allow_pickle=True))['data']
        return data


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


class BinaryMaskd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            data[k][data[k] != 0] = 1
        return data


class CleanMaskd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            mask = data[k]
            mask[mask < 0] = 0
            data[k] = mask
        return data


class modaTransform(object):
    def __init__(self, crop_size=(320, 320)):
        self.crop_size = crop_size
        self.train = Compose([        
            GetCoded(keys='B_paths'),
            LoadNumpyd(keys=['A', 'B', 'A_msk']),
            AddChanneld(keys=['A', 'B', 'A_msk']),
            SpatialPadd(keys=['A', 'B'], spatial_size=self.crop_size, mode='constant', constant_values=-1),
            SpatialPadd(keys='A_msk', spatial_size=self.crop_size, mode='constant', constant_values=0),
            CenterSpatialCropd(keys=['A', 'B', 'A_msk'], roi_size=self.crop_size),
            # RandShiftIntensityd(keys='A', offsets=0.1, prob=0.2),  
            # RandShiftIntensityd(keys='B', offsets=0.1, prob=0.2),  
            # RandScaleIntensityd(keys='A', factors=0.1, prob=0.2),
            # RandScaleIntensityd(keys='B', factors=0.1, prob=0.2),
            # BinaryMaskd(keys=['A_msk']),
            CastToTyped(keys=['A', 'B', 'A_msk'], dtype=(np.float32, np.float32, np.uint8)),
            ToTensord(keys=['A', 'B', 'A_msk'])])

        self.infer = Compose([
            GetCoded(keys='B_paths'),
            LoadNumpyd(keys=['A', 'B', 'A_msk']),
            AddChanneld(keys=['A', 'B', 'A_msk']),
            SpatialPadd(keys=['A', 'B'], spatial_size=self.crop_size, mode='constant', constant_values=-1),
            SpatialPadd(keys='A_msk', spatial_size=self.crop_size, mode='constant', constant_values=0),
            CenterSpatialCropd(keys=['A', 'B', 'A_msk'], roi_size=self.crop_size),
            # BinaryMaskd(keys=['A_msk']),
            CastToTyped(keys=['A', 'B', 'A_msk'], dtype=(np.float32, np.float32, np.uint8)),
            ToTensord(keys=['A', 'B', 'A_msk'])])



# if __name__ == "__main__":

#     data_dict = {
#     'A': '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceImage_2D/crossmoda2023_etz_1_ceT1_slice_0.npz', 
#     'B': '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingTarget_2D/crossmoda2023_etz_106_T2_slice_4.npz', 
#     'A_msk': '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingSourceLabel_2D/crossmoda2023_etz_1_Label_slice_0.npz', 
#     'A_paths': 'a', 
#     'B_paths': 'b'}

#     transform = modaTransform().train

#     output = transform(data_dict)
    
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Load the npz file
#     path = '/data/crossmoda2023/data/crossmoda23_training/resampled_TrainingTarget_2D/crossmoda2023_etz_106_T2_slice_4.npz'
#     image = output['B'][0, ...]

#     # Plot the image using Matplotlib
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.show()
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from glob import glob
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from monai.transforms import *
import util.util as util
from tqdm import tqdm 
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball, disk
from skimage import filters



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


# class PostprocessFakeBd(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys
#         self.version = 'etz & ldn'

#     def __call__(self, data):
#         threshold = -0.5
#         mask = (data['A_msk'] < 3) & (data['A_msk'] > 0).astype('uint8')
#         fakeB = data['fake_B']
#         tp = (fakeB[mask==1] > threshold).astype('uint8')
#         fp = (fakeB[mask==1] <= threshold).astype('uint8')
#         mu, std = fakeB[mask==1][tp==1].mean(), fakeB[mask==1][tp==1].std()
#         wrong = (mask==1) & (fakeB <= threshold)
#         pad = np.random.normal(mu, 0.9*std, size=tuple(fakeB[mask==1][fp==1].shape))
#         tu = fakeB[mask==1]
#         tu[fp==1] = pad
#         fakeB[mask==1] = tu

#         smooth_map = binary_dilation(wrong.astype('uint8')[0], structure=ball(3))[None, ...]
#         smooth_map[mask==0] = 0

#         data['A_msk'] = smooth_map
#         # breakpoint()

#         smt_img = fakeB.copy()
#         smt_img = filters.gaussian(smt_img, sigma=0.7)

#         fakeB[smooth_map==1] = smt_img[smooth_map==1]
#         data['fake_B'] = fakeB
#         return data


class PostprocessFakeBd(MapTransform):
    def __init__(self, keys, site) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.site = site

    def __call__(self, data):
        
        if self.site in ['ETZ', 'LDN']:
            threshold = -0.5
        else:
            threshold = -0.8  # 'UKM'

        mask = (data['A_msk'] < 3) & (data['A_msk'] > 0).astype('uint8')
        fakeB = data['fake_B']
        tp = (fakeB[mask==1] > threshold).astype('uint8')
        fp = (fakeB[mask==1] <= threshold).astype('uint8')
        mu, std = fakeB[mask==1][tp==1].mean(), fakeB[mask==1][tp==1].std()
        wrong = (mask==1) & (fakeB <= threshold)
        pad = np.random.normal(mu, std, size=tuple(fakeB[mask==1][fp==1].shape))
        tu = fakeB[mask==1]
        tu[fp==1] = pad
        fakeB[mask==1] = tu

        smooth_map = binary_dilation(wrong.astype('uint8')[0], structure=ball(3))[None, ...]
        smooth_map[mask==0] = 0

        # data['A_msk'] = smooth_map
        smt_img = fakeB.copy()
        smt_img = filters.gaussian(smt_img, sigma=0.65)

        fakeB[smooth_map==1] = smt_img[smooth_map==1]
        data['fake_B'] = fakeB
        return data



def transform(site):
    return Compose([
        LoadImaged(keys=['A_msk','fake_B']),
        AddChanneld(keys=['A_msk', 'fake_B']),
        PostprocessFakeBd(keys=['A_msk', 'fake_B'], site=site),
        SaveImaged(
            keys=['fake_B'], 
            output_dir=f'/data/crossmoda2023/query-selected-attention/checkpoints/qsSegDynEdge3D/result/post/f_image_{site}', 
            output_postfix='', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),
        ])



def postprocess_dataset(input_dir, site):
    assert site in ['ETZ', 'LDN', 'UKM']
    paths = sorted(glob(input_dir + '/*.nii.gz'))
    mask_dir = '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI'
    with tqdm(total=len(paths)) as pbar:
        for path in paths:
            mask_path = osp.join(mask_dir, osp.basename(path).replace(f'{site}_0000', 'Label'))
            data = {'A_msk': mask_path, 'fake_B': path}
            data = transform(site)(data)
            pbar.update(1)







if __name__ == "__main__":
    # fake_t2_path = '/data/crossmoda2023/query-selected-attention/checkpoints/qsSegDynEdge3D/result/f_image_ukm/crossmoda2023_etz_20_UKM_0000.nii.gz'
    # mask_path = '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_etz_20_Label.nii.gz'

    # data = {'A_msk': mask_path, 'fake_B': fake_t2_path}
    # data = transform()(data)
    
    postprocess_dataset(input_dir='/data/crossmoda2023/query-selected-attention/checkpoints/qsSegDynEdge3D/result/f_image_etz', site='ETZ')
    postprocess_dataset(input_dir='/data/crossmoda2023/query-selected-attention/checkpoints/qsSegDynEdge3D/result/f_image_ldn', site='LDN')
    postprocess_dataset(input_dir='/data/crossmoda2023/query-selected-attention/checkpoints/qsSegDynEdge3D/result/f_image_ukm', site='UKM')



    






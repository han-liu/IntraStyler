import os
import os.path as osp
import torch
import numpy as np
from glob import glob
from monai.transforms import *
import util.util as util
from tqdm import tqdm 


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


def transform_B():
    return Compose([
        LoadImaged(keys=['B']),
        AddChanneld(keys=['B']),
        NormalizeForegroundd(keys=['B']),
        ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        CastToTyped(keys=['B'], dtype=np.float32),
        ToTensord(keys=['B']),])


if __name__ == '__main__':

    # image_dir = '../data/crossmoda23_training/tgtImageROI'
    # save_dir = '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI_norm'

    image_dir = '/data/crossmoda2023/data/crossmoda23_validation/tgtImageROI'
    save_dir = '/data/crossmoda2023/data/crossmoda23_validation/tgtImageROI_norm'
    
    paths = sorted(glob(image_dir + '/*.nii.gz'))
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    with tqdm(total=len(paths)) as pbar:
        for i, path in enumerate(paths):
            data_B = transform_B()({'B': path})

            Compose([
                SqueezeDimd(keys=['B'], dim=0),
                SaveImaged(
                    keys=['B'], 
                    output_dir=save_dir, 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False)
            ])(data_B)


        
            pbar.update(1)
  
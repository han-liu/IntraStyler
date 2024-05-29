import os
import os.path as osp
import numpy as np
from glob import glob
from monai.transforms import *
from tqdm import tqdm 
import nibabel as nib



# def img_tran(pixdim):
#     return Compose([
#         LoadImaged(keys=['image']),
#         AddChanneld(keys=["image"]),
#         Orientationd(keys=["image"], axcodes="LPS"), 
#         CropForegroundd(keys=["image"], source_key='image'),
#         Spacingd(
#             keys=['image'], 
#             pixdim=pixdim, 
#             mode=("bilinear")),
#         NormalizeIntensityd(keys=["image"], nonzero=False),
#         ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         SpatialPadd(keys=["image"], spatial_size=(320, 320, -1), mode='constant', constant_values=-1),
#         CenterSpatialCropd(keys=["image"], roi_size=(320, 320, -1)),
#         SaveImaged(
#             keys=['image'], 
#             output_dir="./", 
#             output_postfix='', 
#             output_ext='.nii.gz', 
#             resample=False,
#             separate_folder=False,
#             print_log=False),
#         ])

class KeepCoch(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            data[k][data[k] < 3] = 0
        return data


if __name__ == "__main__":

    # pre_val_dir = '/data/crossmoda2023/data/crossmoda23_pre_val'
    # pre_val_dir = '/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_preprocess'
    
    # if not osp.exists(pre_val_dir):
    #     os.mkdir(pre_val_dir)

    # img_paths = sorted(glob('/data/crossmoda2023/data/crossmoda23_validation' + '/*.nii.gz'))
    img_paths = sorted(glob('/data/crossmoda2023/data/crossmoda23_training/TrainingTarget' + '/*.nii.gz'))

    img_paths = ['/data/crossmoda2023/data/crossmoda23_training/TrainingTarget/crossmoda2023_ukm_128_T2.nii.gz']

    with tqdm(total=len(img_paths)) as pbar:
        for path in img_paths:
            pseudo_label_path = path.replace('Preprocess', 'Preprocess_Pred').replace('_0000', '')
            data_dict = {'image': path, 'label': pseudo_label_path}
            img = nib.load(path)
            header = img.header
            pixdim = (0.41015625, 0.41015625, header["pixdim"][3])

            tran = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=["image", 'label']),
                Orientationd(keys=["image", 'label'], axcodes="LPS"), 
                CropForegroundd(keys=["image", 'label'], source_key='image'),
                Spacingd(
                    keys=['image', 'label'], 
                    pixdim=pixdim, 
                    mode=("bilinear", 'nearest')),
                NormalizeIntensityd(keys=["image"], nonzero=False),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
                SpatialPadd(keys=["image"], spatial_size=(320, 320, -1), mode='constant', constant_values=-1),
                SpatialPadd(keys=['label'], spatial_size=(320, 320, -1), mode='constant', constant_values=0),
                CenterSpatialCropd(keys=["image", 'label'], roi_size=(320, 320, -1)),
                # SaveImaged(
                #     keys=['image'], 
                #     output_dir=f"{pre_val_dir}", 
                #     output_postfix='', 
                #     output_ext='.nii.gz', 
                #     resample=False,
                #     separate_folder=False,
                #     print_log=False),
                ])
            
            transformed_data = tran(data_dict)

            # seg = Compose([
            #     LoadImaged(keys=['label']),
            #     AddChanneld(keys=['label'])])({'label': pseudo_label_path})['label']

            # with allow_missing_keys_mode(tran):
            #     inverted_seg = tran.inverse({'label': seg})

            # seg = infer_seg(_img, model)[0].detach().cpu()
            # seg.applied_operations = transformed_data["label"].applied_operations
            # seg_dict = {"label": seg}

            # with allow_missing_keys_mode(val_transforms):
            #     inverted_seg = val_transforms.inverse(seg_dict)

            new_path = osp.join('/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Preprocess_Pred_502', osp.basename(path))

            new = Compose([
                LoadImaged(keys=['label']),
                AddChanneld(keys=['label'])])({'label': new_path})['label']

            transformed_data['label'] = new

            xx = tran.inverse(transformed_data)

            inver = Compose([
                KeepCoch(keys=['label']),
                SaveImaged(
                    keys=['label'], 
                    output_dir="/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Pred", 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False)
                ])(xx)

            pbar.update(1)
            


    

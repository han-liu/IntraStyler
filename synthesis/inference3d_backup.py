import os
import os.path as osp
import torch
import numpy as np
from glob import glob
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from monai.inferers import sliding_window_inference
from monai.transforms import *
import util.util as util
from tqdm import tqdm 
import nibabel as nib


"""Program description 
3D inference with qs-att model w/ segmentation head, w/ dynamic instance norm
During inferece, a site argument is used to control which site style to generate. choices=['ldn', 'etz', 'ukm']
python inference.py -n qssegdyn_baseline --QS_mode=global --model qssegDyn --netG resnet_9blocks_dyn --site etz --epoch 45
"""
    
    
def load_model(opt):
    opt.num_threads = 0  
    opt.batch_size = 1  
    opt.serial_batches = True
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()
    return model


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


# class InvertT1d(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         for k in self.keys:
#             data[k] = -data[k]
#         return data


# old version
# def transform():
#     return Compose([
#         LoadImaged(keys=['A']),
#         AddChanneld(keys=['A']),
#         NormalizeForegroundd(keys=['A']),
#         ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         InvertT1d(keys=['A']),
#         # RandSpatialCropd(keys=['A', 'B', 'A_msk'], roi_size=(256, 144, 12), random_center=True, random_size=False),
#         CastToTyped(keys=['A'], dtype=np.float32),
#         ToTensord(keys=['A']),])


# class InvertT1d(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         data['A'] = -data['A']
#         data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
#         # data['Inv_A'][(data['A_msk']==1) | (data['A_msk']==2)] = -1 
#         return data


class InvertT1d(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        # data['A'] = -data['A']
        data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
        data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
        return data


def transform():
    return Compose([
        LoadImaged(keys=['A', 'A_msk']),
        AddCoded(keys=['A']),
        AddChanneld(keys=['A', 'A_msk']),
        NormalizeForegroundd(keys=['A']),
        ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-0.2, b_max=1, clip=True, relative=False),
        InvertT1d(keys=['A']),
        CastToTyped(keys=['A'], dtype=np.float32),
        ToTensord(keys=['A']),])


class AddCoded(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['code'] = torch.tensor([0,0,1.1])  ########################################
        return data


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = load_model(opt).netG

    image_dir = '../data/crossmoda23_training/srcImageROI'
    mask_dir = '../data/crossmoda23_training/srcLabelROI'
    paths = sorted(glob(image_dir + '/*.nii.gz'))
    
    save_dir = osp.join(opt.checkpoints_dir, opt.name, 'result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_img_dir = osp.join(save_dir, 'f_image_intra') ########################################
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)

    with torch.no_grad():
        with tqdm(total=len(paths)) as pbar:
            for path in paths:
                name = osp.basename(path).replace('crossmoda2023_', '').replace('_ceT1.nii.gz', '')
                mask_path = path.replace('ceT1', 'Label').replace('srcImageROI', 'srcLabelROI')
                data = {'A': path, 'A_msk': mask_path}
                data = transform()(data)

                data['A'] = sliding_window_inference(
                    inputs=data['A'].unsqueeze(0).cuda(), 
                    roi_size=(256, 144, 8), 
                    sw_batch_size=1, 
                    predictor=model,
                    overlap=0.8,
                    mode='gaussian',
                    sigma_scale=0.125,
                    padding_mode='constant',
                    cval=-1,
                    code=data['code'].unsqueeze(0).cuda())[0]

                Compose([
                    SqueezeDimd(keys=['A'], dim=0),
                    SqueezeDimd(keys=['A'], dim=0),
                    SaveImaged(
                        keys=['A'], 
                        output_dir=f"{save_img_dir}", 
                        output_postfix='520_v1', 
                        output_ext='.nii.gz', 
                        resample=False,
                        separate_folder=False,
                        print_log=False)])(data),

                pbar.update(1)
                # break

                    



# class InvertT1d(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         # data['A'] = -data['A']
#         data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
#         data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
#         return data


# def transform():
#     return Compose([
#         LoadImaged(keys=['A', 'A_msk']),
#         AddCoded(keys=['A']),
#         AddChanneld(keys=['A', 'A_msk']),
#         NormalizeForegroundd(keys=['A']),
#         ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         InvertT1d(keys=['A']),
#         CastToTyped(keys=['A'], dtype=np.float32),
#         ToTensord(keys=['A']),])


# small tumor
# oversample_list = [
#     "etz_102",
#     "ldn_10",
#     "ldn_13",
#     "ldn_34",
#     "ldn_37",
#     "ldn_39",
#     "ldn_51",
#     "ldn_74",
#     "ldn_75",
#     "ldn_77",
#     "ukm_16",
#     "ukm_33",
#     "ukm_39",
#     "ukm_42",
#     "ukm_43",
#     "etz_3",
#     "etz_5",
#     "etz_13",
#     "etz_18",
#     "etz_22",
#     "etz_23",
#     "etz_34",
#     "etz_47",
#     "etz_65",
#     "etz_71",
#     "etz_73",
#     "etz_77",
#     "etz_82",
#     "etz_87",
#     "etz_95",
#     "etz_98",
#     "ldn_4"]


# oversample_list = [
#     "etz_6",
#     "etz_17",
#     "etz_25",
#     "etz_27",
#     "etz_46",
#     "etz_69",
#     "etz_75",
#     "etz_76",
#     "etz_81",
#     "etz_88",
#     "etz_97",
#     "etz_104",
#     "etz_105",
#     "ldn_5",
#     "ldn_27",
#     "ldn_42", # good
#     "ldn_50",
#     "ldn_58", # good
#     "ldn_63", 
#     "ldn_64", 
#     "ldn_69", 
#     "ldn_70", 
#     "ldn_73", 
#     "ldn_78", 
#     "ukm_3", 
#     "ukm_15", 
#     "ukm_22", 
#     "ukm_23", # good
#     "ukm_25", # good
#     "ukm_34", 
#     "ukm_38",  # good
#     ]

# heterogeous tumors
# oversample_list = [
#     "etz_6",
#     "etz_17",
#     "etz_27",
#     "etz_46",
#     "etz_75",
#     "etz_76",
#     "etz_81",
#     "etz_105",
#     "ldn_5",
#     "ldn_42", # good
#     "ldn_64", 
#     "ldn_69", 
#     "ldn_70", 
#     "ldn_78", 
#     "ukm_3", 
#     "ukm_25",  # good
#     "ukm_34", 
#     "ukm_38",  # good
#     ]

# ldn10, ldn13, ldn51, etz18,  # ldn39, etz_71, etz_102, etz71, 

# oversample_list = [
#     "ldn_10",
#     "ldn_13",
#     "ldn_51",
#     "etz_18",
#     "ldn_37",
#     "ldn_4",
#     "ldn_39",
#     "ukm_16",
#     "etz_71",
#     "etz_102",
#     "ukm_33"
#     ]


# small dark tumors
# oversample_list = [
#     "etz_22",
#     "etz_23",
#     "etz_29",
#     "etz_30",
#     "etz_31",
#     "etz_34",
#     "etz_45",
#     "etz_47",
#     "etz_71",
#     "etz_87",
#     "etz_95",
#     "etz_102",
#     "ldn_4",
#     "ldn_10",
#     "ldn_13",
#     "ldn_23",
#     "ldn_34",
#     "ldn_75",
#     "ldn_77",
#     "ukm_13",
#     "ukm_16",
#     "ukm_37",
#     "ukm_43",
#     ]
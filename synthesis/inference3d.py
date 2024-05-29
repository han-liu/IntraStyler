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


class InvertT1d(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        # data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
        data['A'][data['A_msk']==3] = -data['A'][data['A_msk']==3]
        data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
        return data

def select_low(x):
    return x > -0.99


# def transform():
#     return Compose([
#         LoadImaged(keys=['A', 'A_msk', 'B']),
#         AddChanneld(keys=['A', 'A_msk', 'B']),
#         NormalizeForegroundd(keys=['A', 'B']),
#         ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         # InvertT1d(keys=['A']),
#         CastToTyped(keys=['A', 'B'], dtype=np.float32),
#         ToTensord(keys=['A', 'B']),])


# def transform_B():
#     return Compose([
#         LoadImaged(keys=['A']),
#         AddChanneld(keys=['A']),
#         NormalizeForegroundd(keys=['A']),
#         ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         CastToTyped(keys=['A'], dtype=np.float32),
#         ToTensord(keys=['A']),])


def transform_A():
    return Compose([
        LoadImaged(keys=['A', 'A_msk']),
        AddChanneld(keys=['A', 'A_msk']),
        NormalizeForegroundd(keys=['A']),
        ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        InvertT1d(keys=['A']),
        CastToTyped(keys=['A'], dtype=np.float32),
        ToTensord(keys=['A']),])


# def transform_B():
#     return Compose([
#         LoadImaged(keys=['B']),
#         AddChanneld(keys=['B']),

#         CropForegroundd(keys='B', source_key='B'), # , k_divisible=(8, 4, 1)
#         SpatialPadd(keys='B', spatial_size=(256, 144, 32), mode='replicate'),

#         NormalizeForegroundd(keys=['B']),
#         ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         CastToTyped(keys=['B'], dtype=np.float32),
#         ToTensord(keys=['B']),])


def transform_B():
    return Compose([
        LoadImaged(keys=['B']),
        AddChanneld(keys=['B']),
        NormalizeForegroundd(keys=['B']),
        ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        CastToTyped(keys=['B'], dtype=np.float32),
        ToTensord(keys=['B']),])


def transform_BA():
    return Compose([
        LoadImaged(keys=['A']),
        AddChanneld(keys=['A']),
        NormalizeForegroundd(keys=['A']),
        ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        CastToTyped(keys=['A'], dtype=np.float32),
        ToTensord(keys=['A']),])


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = load_model(opt).netG

    image_dir = '../data/crossmoda23_training/srcImageROI'
    mask_dir = '../data/crossmoda23_training/srcLabelROI'
    paths = sorted(glob(image_dir + '/*.nii.gz'))
    
    save_dir = osp.join(opt.checkpoints_dir, opt.name, 'result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_img_dir = osp.join(save_dir, 'outputs') ########################################
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)

    # path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_68_ceT1.nii.gz'
    # path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ldn_16_ceT1.nii.gz'
    # path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_6_ceT1.nii.gz'
    # path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_23_ceT1.nii.gz'
    # path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_8_ceT1.nii.gz'
    # path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_20_ceT1.nii.gz'
    # path_A_msk = path_A.replace('ceT1', 'Label').replace('srcImageROI', 'srcLabelROI')

    # path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_6_ceT1.nii.gz'

    # ref_paths = [
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_180_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_110_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_80_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_101_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_44_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_46_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_77_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_85_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_129_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_139_T2.nii.gz'
    # ]

    # 20
    # ref_paths = [
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_175_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_144_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_145_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_66_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_106_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_107_T2.nii.gz'
    # ]

    # # 30 contrast V5
    # ref_paths = [
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_202_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_88_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_120_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_66_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_106_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_107_T2.nii.gz'
    # ]

    # 20 contrast v8 typiclust! 5 clusters
    # ref_paths = [
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_107_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_155_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_120_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_52_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_97_T2.nii.gz',
    # ]

    # contrast v9
    # ref_paths = [
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_176_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_88_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_134_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_53_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_106_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_107_T2.nii.gz',
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_128_T2.nii.gz'
    # ]

    # contrast v10
    ref_paths = [
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_164_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_91_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_138_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_52_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_53_T2.nii.gz',
        # '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_116_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_164_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_91_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_107_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_52_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_54_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_116_T2.nii.gz'
    ]

    # hetero_list = ["etz_6", "etz_17", "etz_27", "etz_46", "etz_75", "etz_76", "etz_81", "etz_105", "ldn_5",
    #     "ldn_42", "ldn_64", "ldn_69", "ldn_70", "ldn_78", "ukm_3", "ukm_25", "ukm_34", "ukm_38"]

    # small_tumor_list = ['etz_102', 'etz_18', 'etz_22', 'etz_23', 'etz_29', 'etz_30', 'etz_31', 'etz_34', 'etz_45', 
    # 'etz_47', 'etz_71', 'etz_73', 'etz_87', 'etz_95', 'ldn_10', 'ldn_13', 'ldn_23', 'ldn_34', 'ldn_37', 'ldn_39', 
    # 'ldn_4', 'ldn_51', 'ldn_71', 'ldn_75', 'ldn_77', 'ukm_11', 'ukm_13', 'ukm_14', 'ukm_16', 'ukm_37', 'ukm_43']

    hetero_ids = ["etz_6", "etz_17", "etz_25", "etz_27", "etz_46", "etz_69", "etz_75", "etz_76", "etz_81",
            "etz_105", "ldn_42", "ldn_58", "ldn_64", "ldn_69", "ldn_70", "ldn_78", "ukm_3", "ukm_23", "ukm_25", "ukm_34", "ukm_38"]
    small_ids = ["etz_3", "etz_5", "etz_18", "etz_22", "etz_23", "etz_34", "etz_47", "etz_65", "etz_71", "etz_73", "etz_77",
            "etz_87", "etz_95", "etz_98", "etz_102", "ldn_4", "ldn_10", "ldn_13", "ldn_34", "ldn_37", "ldn_39", "ldn_51", "ldn_74",
            "ldn_75", "ldn_77", "ukm_11", "ukm_13", "ukm_14", "ukm_16", "ukm_33", "ukm_39", "ukm_42", "ukm_43"]

    # oversample_list = [
#     "etz_3",
#     "etz_5",
#     "etz_18",
#     "etz_22",
#     "etz_23",
#     "etz_34",
#     "etz_47",
#     "etz_65",
#     "etz_71",
#     "etz_73",
#     "etz_77",
#     "etz_87",
#     "etz_95",
#     "etz_98",
#     "etz_102",
#     "ldn_4",
#     "ldn_10",
#     "ldn_13",
#     "ldn_34",
#     "ldn_37",
#     "ldn_39",
#     "ldn_51",
#     "ldn_74",
#     "ldn_75",
#     "ldn_77",
#     "ukm_11",
#     "ukm_13",
#     "ukm_14",
#     "ukm_16",
#     "ukm_33",
#     "ukm_39",
#     "ukm_42",
#     "ukm_43",
#     ]

    with torch.no_grad():
        with tqdm(total=len(paths)) as pbar:
            for i, path in enumerate(paths):
                name = osp.basename(path).replace('crossmoda2023_', '').replace('_ceT1.nii.gz', '')
                mask_path = path.replace('ceT1', 'Label').replace('srcImageROI', 'srcLabelROI')
                # data_A = {'A': path, 'A_msk': mask_path}
                # data_B = {'B': ref_image_path}

                # oversampling
                # if not (osp.basename(mask_path)[14:-13] in hetero_list+small_tumor_list):
                #     pbar.update(1)
                #     continue

                for ref_path in ref_paths:
                    # data_A = {'A': path_A, 'A_msk': path_A_msk}
                    data_A = {'A': path, 'A_msk': mask_path}

                    # data_A = {'A': path_A}
                    data_B = {'B': ref_path}
                    data_A = transform_A()(data_A)
                    # data_A = transform_BA()(data_A)
                    data_B = transform_B()(data_B)
                    
                    # single style mode
                    # ref_style = model.extract_style(data_B['B'].unsqueeze(0).cuda())
                    # data_A['A'] = sliding_window_inference(
                    #     inputs=data_A['A'].unsqueeze(0).cuda(), 
                    #     roi_size=(256, 144, 8), 
                    #     sw_batch_size=1, 
                    #     predictor=model,
                    #     overlap=0.9,
                    #     mode='gaussian',
                    #     sigma_scale=0.125,
                    #     padding_mode='constant',
                    #     cval=-1,
                    #     is_train=False,
                    #     ref_style=ref_style)[0]

                    # concat mode
                    data_A['A'] = sliding_window_inference(
                        inputs=torch.cat((data_A['A'], data_B['B']), 0).unsqueeze(0).cuda(), 
                        roi_size=(256, 144, 8), 
                        sw_batch_size=1, 
                        predictor=model,
                        overlap=0.9,
                        mode='gaussian',
                        sigma_scale=0.125,
                        padding_mode='constant',
                        cval=-1,
                        is_train=False)[0]

                    # Compose([
                    #     SqueezeDimd(keys=['A'], dim=0),
                    #     SqueezeDimd(keys=['A'], dim=0),
                    #     SaveImaged(
                    #         keys=['A'], 
                    #         output_dir=f"{save_img_dir}", 
                    #         output_postfix=osp.basename(ref_path)[-17:-7], 
                    #         output_ext='.nii.gz', 
                    #         resample=False,
                    #         separate_folder=False,
                    #         print_log=False)])(data_A)

                    # oversampling
                    # Compose([
                    #     SqueezeDimd(keys=['A'], dim=0),
                    #     SqueezeDimd(keys=['A'], dim=0),
                    #     SaveImaged(
                    #         keys=['A'], 
                    #         output_dir=f"/data/crossmoda2023/query-selected-attention/checkpoints/contrast_v9/result/OS_imagesTr", 
                    #         output_postfix=osp.basename(ref_path)[-17:-7] + '_OS_0000', 
                    #         output_ext='.nii.gz', 
                    #         resample=False,
                    #         separate_folder=False,
                    #         print_log=False),

                    #     SaveImaged(
                    #         keys=['A_msk'], 
                    #         output_dir=f"/data/crossmoda2023/query-selected-attention/checkpoints/contrast_v9/result/OS_labelsTr", 
                    #         output_postfix=osp.basename(ref_path)[-17:-7] + '_OS', 
                    #         output_ext='.nii.gz', 
                    #         resample=False,
                    #         separate_folder=False,
                    #         print_log=False)])(data_A),

                    Compose([
                        SqueezeDimd(keys=['A'], dim=0),
                        SqueezeDimd(keys=['A'], dim=0),
                        SaveImaged(
                            keys=['A'], 
                            output_dir=f"/data/crossmoda2023/query-selected-attention/checkpoints/contrast_v10/result/imagesTr", 
                            output_postfix=osp.basename(ref_path)[-17:-7],# + '_0000', 
                            output_ext='.nii.gz', 
                            resample=False,
                            separate_folder=False,
                            print_log=False),

                        SaveImaged(
                            keys=['A_msk'], 
                            output_dir=f"/data/crossmoda2023/query-selected-attention/checkpoints/contrast_v10/result/labelsTr", 
                            output_postfix=osp.basename(ref_path)[-17:-7], 
                            output_ext='.nii.gz', 
                            resample=False,
                            separate_folder=False,
                            print_log=False)])(data_A),


                
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
#     "etz_69",
#     "etz_75",
#     "etz_76",
#     "etz_81",
#     "etz_105",
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


# small tumor
# oversample_list = [
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
#     "etz_102",
#     "ldn_4",
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
#     ]
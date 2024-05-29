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
PCA augmentation based on the style codes extracted from the style encoder.
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
        data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
        data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
        return data

def select_low(x):
    return x > -0.99


def transform():
    return Compose([
        LoadImaged(keys=['A', 'A_msk', 'B']),
        AddChanneld(keys=['A', 'A_msk', 'B']),
        NormalizeForegroundd(keys=['A', 'B']),
        ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        InvertT1d(keys=['A']),
        CastToTyped(keys=['A', 'B'], dtype=np.float32),
        ToTensord(keys=['A', 'B']),])


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

    ref_paths = [
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_180_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_101_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_44_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_46_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_77_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_85_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_129_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_139_T2.nii.gz'
    ]

    with torch.no_grad():
        with tqdm(total=len(paths)) as pbar:
            for i, path in enumerate(paths):
                name = osp.basename(path).replace('crossmoda2023_', '').replace('_ceT1.nii.gz', '')
                mask_path = path.replace('ceT1', 'Label').replace('srcImageROI', 'srcLabelROI')
                # data = {'A': path, 'A_msk': mask_path, 'B': ref_image_path}

                for ref_path in ref_paths:
                    data = {
                        # 'A': '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_76_ceT1.nii.gz', 
                        'A': '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_12_ceT1.nii.gz', 
                        'A_msk': '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_ukm_12_Label.nii.gz', 
                        'B': ref_path}

                    data = transform()(data)

                    data['A'] = sliding_window_inference(
                        inputs=torch.cat((data['A'], data['B']), 0).unsqueeze(0).cuda(), 
                        roi_size=(256, 144, 8), 
                        sw_batch_size=1, 
                        predictor=model,
                        overlap=0.8,
                        mode='gaussian',
                        sigma_scale=0.125,
                        padding_mode='constant',
                        cval=-1,
                        is_train=False)[0]

                    Compose([
                        SqueezeDimd(keys=['A'], dim=0),
                        SqueezeDimd(keys=['A'], dim=0),
                        SaveImaged(
                            keys=['A'], 
                            output_dir=f"{save_img_dir}", 
                            output_postfix=osp.basename(ref_path)[-17:-7], 
                            output_ext='.nii.gz', 
                            resample=False,
                            separate_folder=False,
                            print_log=False)])(data),

                
                pbar.update(1)
                break

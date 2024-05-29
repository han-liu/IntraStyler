import os
import os.path as osp
import torch
import numpy as np
from glob import glob
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from monai.transforms import *
import util.util as util
from tqdm import tqdm 
import nibabel as nib


"""Program description 
2D inference with qs-att model w/ segmentation head, w/ dynamic instance norm
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


class GetCoded(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        if 'ukm' in data['site']:
            data['code'] = torch.tensor([0,0,1])
        elif 'etz' in data['site']:
            data['code'] = torch.tensor([0,1,0])
        elif 'ldn' in data['site']:
            data['code'] = torch.tensor([1,0,0])
        return data


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = load_model(opt).netG

    crop_size = (320, 320)
    image_dir = '../data/crossmoda23_training/resampled_TrainingSourceImage'
    mask_dir = '../data/crossmoda23_training/resampled_TrainingSourceLabel'
    paths = glob(image_dir + '/*.nii.gz')
    
    save_dir = osp.join(opt.checkpoints_dir, opt.name, 'result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_img_dir = osp.join(save_dir, f'{opt.site}')
    save_msk_dir = osp.join(save_dir, 'label')

    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    if not os.path.exists(save_msk_dir):
        os.mkdir(save_msk_dir)

    normalize = Compose([
        NormalizeIntensity(nonzero=False),
        ScaleIntensityRangePercentiles(lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False)])

    transform2D = Compose([
        GetCoded(keys='site'),
        AddChanneld(keys=['A', 'A_msk']),
        SpatialPadd(keys='A', spatial_size=crop_size, mode='constant', constant_values=-1),
        SpatialPadd(keys='A_msk', spatial_size=crop_size, mode='constant', constant_values=0),
        CenterSpatialCropd(keys=['A', 'A_msk'], roi_size=crop_size),
        CastToTyped(keys='A', dtype=np.float32),
        ToTensord(keys='A')
        ])

    with torch.no_grad():
        with tqdm(total=len(paths)) as pbar:
            for path in paths:
                # print(path)
                ref_image = nib.load(path)
                data = ref_image.get_fdata()
                data = normalize(data)
                mask_path = path.replace('Image', 'Label').replace('ceT1', 'Label')
                ref_label = nib.load(mask_path)
                label = ref_label.get_fdata()
                syn, msk = [], []

                for z in range(data.shape[2]):
                    A = np.transpose(data[:, :, z])
                    A_msk = np.transpose(label[:, :, z])
                    data_dict = {'A': A, 'A_msk': A_msk, 'site': opt.site}
                    data_dict = transform2D(data_dict)

                    syn_slc = model(
                        input=data_dict['A'].unsqueeze(0).cuda(), 
                        code=data_dict['code'].unsqueeze(0).cuda())[0].squeeze(0).squeeze(0).detach().cpu().numpy()

                    syn.append(syn_slc)
                    msk.append(data_dict['A_msk'].squeeze(0))

                img = nib.Nifti1Image(np.transpose(np.array(syn), (2, 1, 0)), affine=ref_image.affine)
                # msk = nib.Nifti1Image(np.transpose(np.array(msk), (2, 1, 0)), affine=ref_label.affine)


                nib.save(img, osp.join(save_img_dir, osp.basename(path)))
                # nib.save(msk, osp.join(save_msk_dir, osp.basename(mask_path)))

                pbar.update(1)

                    
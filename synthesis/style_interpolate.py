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
        data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
        # data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
        return data


def slerp(v1, v2, t, verbose=False):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot = np.dot(v1, v2)
    # if dot < 0.0:
    #     v2 = -v2
    #     dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 < 1e-6:
        return (1.0 - t) * v1 + t * v2

    theta_t = theta_0 * t  # angle between v1 and the output
    sin_theta_t = np.sin(theta_t)  # compute sine of theta_t

    s1 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s2 = sin_theta_t / sin_theta_0
    output = s1 * v1 + s2 * v2

    if verbose:
        print(f'norm of out: {np.linalg.norm(output)}')
        print(f'angle between v1  and v2 : {np.arccos(np.dot(v1, v2)) / np.pi * 180:.2f}')
        print(f'angle between v1  and out: {np.arccos(np.dot(v1, output)) / np.pi * 180:.2f}')
        print(f'angle between out and v2 : {np.arccos(np.dot(output, v2)) / np.pi * 180:.2f}')
    return output


def transform_A():
    return Compose([
        LoadImaged(keys=['A', 'A_msk']),
        AddChanneld(keys=['A', 'A_msk']),
        NormalizeForegroundd(keys=['A']),
        ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        InvertT1d(keys=['A']),
        CastToTyped(keys=['A'], dtype=np.float32),
        ToTensord(keys=['A']),])


def transform_B():
    return Compose([
        LoadImaged(keys=['B']),
        AddChanneld(keys=['B']),
        NormalizeForegroundd(keys=['B']),
        ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        CastToTyped(keys=['B'], dtype=np.float32),
        ToTensord(keys=['B']),])


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

    path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ldn_43_ceT1.nii.gz'
    path_A_msk = path_A.replace('ceT1', 'Label').replace('srcImageROI', 'srcLabelROI')
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

    ref_paths = [
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_202_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_88_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ldn_120_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_66_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_106_T2.nii.gz',
        '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_107_T2.nii.gz'
    ]

    with torch.no_grad():
        with tqdm(total=len(paths)) as pbar:
            for i, path in enumerate(paths):
                name = osp.basename(path).replace('crossmoda2023_', '').replace('_ceT1.nii.gz', '')
                mask_path = path.replace('ceT1', 'Label').replace('srcImageROI', 'srcLabelROI')

                data_A = {'A': path_A, 'A_msk': path_A_msk}
                data_A = transform_A()(data_A)

                data_B = {'B': ref_paths[0]}
                data_B = transform_B()(data_B)
                ref_style1 = model.extract_style(data_B['B'].unsqueeze(0).cuda())

                data_B = {'B': ref_paths[1]}
                data_B = transform_B()(data_B)
                ref_style2 = model.extract_style(data_B['B'].unsqueeze(0).cuda())

                # for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                # for t in [1.1, 1.2, 1.3, 1.4, 1.5]:   # extrapolation
                for i, t in enumerate([1.000,1.000,1.000,1.000,1.000]):
                    slerp_style = slerp(ref_style1.view(-1).cpu(), ref_style2.view(-1).cpu(), t, True)

                    slerp_style = slerp_style.numpy() + np.random.normal(0, 15, size=slerp_style.numpy().shape)
                    slerp_style /= np.linalg.norm(slerp_style)
                    slerp_style = torch.tensor(slerp_style)

                    slerp_style = slerp_style.unsqueeze(0).cuda()

                    data_A['output'] = sliding_window_inference(
                        inputs=data_A['A'].unsqueeze(0).cuda(), 
                        roi_size=(256, 144, 12), 
                        sw_batch_size=1, 
                        predictor=model,
                        overlap=0.8,
                        mode='gaussian',
                        sigma_scale=0.125,
                        padding_mode='constant',
                        cval=-1,
                        is_train=False,
                        ref_style=slerp_style)[0]
                    data_A['output_meta_dict'] = data_A['A_meta_dict']

                    Compose([
                        SqueezeDimd(keys=['output'], dim=0),
                        SqueezeDimd(keys=['output'], dim=0),
                        SaveImaged(
                            keys=['output'], 
                            output_dir=f"{save_img_dir}", 
                            output_postfix=f'_{t}_{i}', 
                            output_ext='.nii.gz', 
                            resample=False,
                            separate_folder=False,
                            print_log=False)])(data_A),

            
                pbar.update(1)
                break
                    
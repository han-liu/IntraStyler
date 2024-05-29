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
import pickle
    
    
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
        # data['A'][data['A_msk']==3] = -data['A'][data['A_msk']==3]
        # data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)

        data['A'][data['A_msk']==3] *= -1
        data['A'][(data['A_msk']==1) | (data['A_msk']==2)] *= -1
        
        return data


def select_low(x):
    return x > -0.9999


def transform():
    return Compose([
        LoadImaged(keys=['B']),
        AddChanneld(keys=['B']),
        CropForegroundd(keys='B', source_key='B'), # , k_divisible=(8, 4, 1)
        
        # SpatialPadd(keys='B', spatial_size=(256, 144, -1), constant_values=0),
        NormalizeForegroundd(keys=['B']),
        ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        SpatialPadd(keys='B', spatial_size=(256, 144, 32), mode='replicate'),
        # CenterSpatialCropd(keys=['B'], roi_size=(256, 144, 12)),
        CastToTyped(keys=['B'], dtype=np.float32),
        ToTensord(keys=['B']),])


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = load_model(opt).netG

    image_dir = '../data/crossmoda23_training/tgtImageROI'
    paths = sorted(glob(image_dir + '/*.nii.gz'))
    save_dir = osp.join(opt.checkpoints_dir, opt.name, 'result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_img_dir = osp.join(save_dir, 'embeddings')
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)

    # record
    data_path = osp.join(save_img_dir, f'projector_data.tsv')
    metadata_path = osp.join(save_img_dir, f'projector_metadata.tsv')
    f_data, f_metadata = open(data_path, 'w'), open(metadata_path, 'w')
    f_metadata.write('id\tdataset\n')
    style_vectors, subject_ids, sites = [], [], []

    with torch.no_grad():
        with tqdm(total=len(paths)) as pbar:
            for i, path in enumerate(paths):
                # if i > 10:
                #     break

                data = {'B': path}
                data = transform()(data)

                # print(data['B'].shape)

                data['B'] = model.extract_style(data['B'].unsqueeze(0).cuda())
                
                # data['B'] = sliding_window_inference(
                #     inputs=data['B'].unsqueeze(0).cuda(), 
                #     roi_size=(256, 144, 8), 
                #     sw_batch_size=1, 
                #     predictor=model.style_encoder,
                #     overlap=0.8,
                #     mode='constant',
                #     padding_mode='constant',
                #     cval=-1)[0]
                
                # data['B'] = data['B'].squeeze(0).squeeze(0).mean(1)

                v = data['B'].detach().cpu().numpy().reshape(-1) 
                assert v.shape[0]==256, f'{v.shape[0]}'
                output_string = '\t'.join(map(str, v))
                f_data.write(output_string + '\n')
                subject_id = osp.basename(path)[14:-10]
                site = subject_id[:3]
                f_metadata.write(f'{subject_id}\t{site}\n')
                style_vectors.append(v)
                subject_ids.append(subject_id)
                sites.append(site)

                
                pbar.update(1)

    with open(osp.join(save_img_dir, 'style_vectors.npz'), 'wb') as f:
        pickle.dump(style_vectors, f)
    with open(osp.join(save_img_dir, 'subject_ids.npz'), 'wb') as f:
        pickle.dump(subject_ids, f)
    with open(osp.join(save_img_dir, 'sites.npz'), 'wb') as f:
        pickle.dump(sites, f)
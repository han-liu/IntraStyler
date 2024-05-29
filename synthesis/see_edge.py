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



def sobelLayer(input):
    pad = nn.ConstantPad3d((1,1,1,1,1,1),-1)
    kernel = create3DsobelFilter().float()
    act = nn.Tanh()
    paded = pad(input)
    fake_sobel = F.conv3d(paded, kernel, padding = 0, groups = 1)/4
    n,c,h,w,l = fake_sobel.size()
    fake = torch.norm(fake_sobel,2,1,True)/c*3
    fake_out = act(fake)*2-1
    return fake_out


def create3DsobelFilter():
    num_1, num_2, num_3 = np.zeros((3,3))
    num_1 = [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
    num_2 = [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    num_3 = [[-1., -2., -1.],
             [-2., -4., -2.],
             [-1., -2., -1.]]
    sobelFilter = np.zeros((3,1,3,3,3))
    sobelFilter[0,0,0,:,:] = num_1
    sobelFilter[0,0,1,:,:] = num_2
    sobelFilter[0,0,2,:,:] = num_3
    sobelFilter[1,0,:,0,:] = num_1
    sobelFilter[1,0,:,1,:] = num_2
    sobelFilter[1,0,:,2,:] = num_3
    sobelFilter[2,0,:,:,0] = num_1
    sobelFilter[2,0,:,:,1] = num_2
    sobelFilter[2,0,:,:,2] = num_3
    return torch.from_numpy(sobelFilter)


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
        data['A'] = -data['A']
        data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
        # data['Inv_A'][(data['A_msk']==1) | (data['A_msk']==2)] = -1 
        return data


def transform():
    return Compose([
        LoadImaged(keys=['A', 'A_msk', 'fake_B']),
        AddChanneld(keys=['A', 'A_msk', 'fake_B']),
        NormalizeForegroundd(keys=['A']),
        ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-0.2, b_max=1, clip=True, relative=False),
        InvertT1d(keys=['A']),
        CastToTyped(keys=['A'], dtype=np.float32),
        GetEdged(keys=['A']),
        ToTensord(keys=['A', 'fake_B', 'edge', 'A_msk']),
        GetBoundaryMaskd(keys=['A']),
        GetOutputd(keys=['A']),
        SaveImaged(
            keys=['A'], 
            output_dir='/data/crossmoda2023/query-selected-attention', 
            output_postfix='edgeA', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),
        SaveImaged(
            keys=['fake_B'], 
            output_dir='/data/crossmoda2023/query-selected-attention', 
            output_postfix='edgeB', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),
        SaveImaged(
            keys=['A_msk'], 
            output_dir='/data/crossmoda2023/query-selected-attention', 
            output_postfix='new_mask', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),
        ])


class GetEdged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        # remove cochlea
        # data['label'][data['label']==3] = 0
        # binarize label
        co_mask = data['A_msk'].copy()
        vs_mask = data['A_msk'].copy()

        co_mask[co_mask<3]=0
        co_mask[co_mask!=0]=1
        co_mask = binary_dilation(co_mask[0], structure=ball(4)).astype('uint8')
    
        vs_mask[vs_mask>2]=0
        vs_mask[vs_mask!=0]=1
        label = vs_mask[0]
        output = []
        for z in range(label.shape[2]):
            slc = label[:, :, z]
            dilate = binary_dilation(slc, structure=disk(5))
            erode = binary_erosion(slc, structure=disk(5))
            dilate[erode] = False
            output.append(dilate)
        output = np.transpose(np.array(output), (1, 2, 0))

        output[co_mask==1] = 2

        data['edge'] = output[None, ...]
        return data


class GetBoundaryMaskd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        vs_mask = data['A_msk']
        vs_mask[vs_mask>2]=0
        vs_mask[vs_mask!=0]=1
        vs_mask[vs_mask!=1]=-1
        vs_mask[vs_mask==0]=1
        vs_mask[vs_mask!=1]=0
        data['A_msk'] = vs_mask
        return data


class GetOutputd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['edge_A'] = sobelLayer(data['A'].unsqueeze(0)).squeeze(0)
        data['edge_B'] = sobelLayer(data['fake_B'].unsqueeze(0)).squeeze(0)
        edge_roi = data['edge']
        data['A'] = data['edge_A'] * edge_roi
        data['fake_B'] = data['edge_B'] * edge_roi
        return data



if __name__ == "__main__":

    # t1_path = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_8_ceT1.nii.gz'
    # fake_t2_path = '/data/crossmoda2023/query-selected-attention/checkpoints/qsSegDynEdge3D/result/f_image_etz/crossmoda2023_etz_8_ETZ_0000.nii.gz'
    # mask_path = '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_etz_8_Label.nii.gz'
    # roi_path = '/data/crossmoda2023/data/crossmoda23_training/srcEdgeROI/crossmoda2023_etz_8_Label.nii.gz'

    # data = {'A': t1_path, 'A_msk': mask_path, 'fake_B': fake_t2_path, 'edge': roi_path}
    # data = transform()(data)


    t1_path = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_8_ceT1.nii.gz'
    fake_t2_path = '/data/crossmoda2023/query-selected-attention/checkpoints/qsSegDynEdge3D/result/f_image_etz/crossmoda2023_etz_8_ETZ_0000.nii.gz'
    mask_path = '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_etz_8_Label.nii.gz'
    roi_path = '/data/crossmoda2023/data/crossmoda23_training/srcEdgeROI/crossmoda2023_etz_8_Label.nii.gz'

    data = {'A': t1_path, 'A_msk': mask_path, 'fake_B': fake_t2_path, 'edge': roi_path}
    data = transform()(data)
    

    






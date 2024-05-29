import os.path
from glob import glob
from data.base_dataset import BaseDataset
# from base_dataset import BaseDataset
import random
import numpy as np
import torch
import copy
from monai.transforms import *


class Unaligned3DDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, 'srcImageROI') 
            self.dir_A_msk = os.path.join(opt.dataroot, 'srcLabelROI') 
            self.dir_B = os.path.join(opt.dataroot, 'tgtImageROI')  
            self.A_img_paths = sorted(glob(self.dir_A + '/*.nii.gz'))
            self.A_msk_paths = [path.replace('Image', 'Label').replace('ceT1', 'Label') for path in self.A_img_paths]   
            self.A_edge_paths = [path.replace('srcLabelROI', 'srcEdgeV4ROI') for path in self.A_msk_paths]   # check version!
            self.B_img_paths = sorted(glob(self.dir_B + '/*.nii.gz'))

        if opt.phase == "test":
            pass

        self.A_size = len(self.A_img_paths)  
        self.B_size = len(self.B_img_paths)
        self.transform = modaTransform()
        self.phase = opt.phase

    def __getitem__(self, index):
        A_img_path = self.A_img_paths[index % self.A_size] 
        A_msk_path = self.A_msk_paths[index % self.A_size] 
        A_edge_path = self.A_edge_paths[index % self.A_size] 

        if self.opt.serial_batches:   
            index_B = index % self.B_size
        else:   
            index_B = random.randint(0, self.B_size - 1)
        B_img_path = self.B_img_paths[index_B]
        data_dict = {'A': A_img_path, 'B': B_img_path, 'A_msk': A_msk_path, 'A_edge': A_edge_path, 'A_paths': A_img_path, 'B_paths': B_img_path}
        if self.phase == 'train':
            # data_dict = self.transform.train(data_dict)
            output = self.transform.train(data_dict)
            neg_dict = (modaTransform().get_negatives)({'posB': output['posB']})
            negB = [x['negB'] for x in neg_dict]
            data_dict = {
                'A': output['A'], 
                'B': output['B'], 
                'A_msk': output['A_msk'], 
                'A_edge': output['A_edge'],
                'PosB': output['posB'],
                'NegB': negB}
            del output, neg_dict

        elif self.phase == 'test':
            pass  # inference code is not implemented here
        return data_dict

    def __len__(self):
        return max(self.A_size, self.B_size)


class GetCoded(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        if 'ukm' in data['B_paths']:
            data['code'] = torch.tensor([0,0,1])
        elif 'etz' in data['B_paths']:
            data['code'] = torch.tensor([0,1,0])
        elif 'ldn' in data['B_paths']:
            data['code'] = torch.tensor([1,0,0])
        return data


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


class AdjustEdged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['A_edge'][data['A_edge']==2] = 10
        data['A_edge'][data['A_edge']==1] = 2 
        return data


# class InvertT1d(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         data['A'] = -data['A']
#         data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
#         # data['Inv_A'][(data['A_msk']==1) | (data['A_msk']==2)] = -1 
#         return data



# class modaTransform(object):
#     """contrast v5: any patches within the same volume are considered positive pairs
#     therefore, any patch from this volume should have the same style code!"""
#     def __init__(self):
#         self.train = Compose([        
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             Copyd(keys='B', new_key='posB'),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandFlipd(keys='posB', prob=0.5, spatial_axis=0),

#             NormalizeForegroundd(keys=['A', 'B', 'posB']),
#             RandAdjustContrastd(keys='A', prob=0.3, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys=['B', 'posB'], prob=0.3, gamma=(0.8, 1.2)),
#             ScaleIntensityRangePercentilesd(keys=['A', 'B', 'posB'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=(256, 144, 12), random_center=True, random_size=False),
#             RandSpatialCropd(keys='posB', roi_size=(256, 144, 12), random_center=True, random_size=False),
#             AdjustEdged(keys=['A_edge']),
#             CastToTyped(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'posB', 'A_msk', 'A_edge']),])

#         self.get_negatives = Compose([
#             Copyd(keys='posB', new_key='negB'),
#             RandSpatialCropSamplesd(keys='negB', roi_size=(256, 144, 12), num_samples=6, random_center=True, random_size=False),
#             IntenTransform(keys='negB'),
#             CastToTyped(keys=['negB'], dtype=np.float32),
#             ToTensord(keys=['negB'])
#         ])


class modaTransform(object):
    """

    contrast v5: any patches within the same volume are considered positive pairs
    therefore, any patch from this volume should have the same style code!

    - no spatial transformation on posB
    - fixed hyperparameter tau:100

    contrast v6: any patches within the same volume are considered positive pairs
    therefore, any patch from this volume should have the same style code!

    - additional spatial transformation on posB
    - learnable hyperparameter tau

    contrast v7: any patches within the same volume are considered positive pairs
    therefore, any patch from this volume should have the same style code!

    - additional spatial transformation on posB
    - learnable hyperparameter tau (REMOVED)

    contrast v8: any patches within the same volume are considered positive pairs
    therefore, any patch from this volume should have the same style code!

    - additional spatial transformation on posB
    - tau = 20
    - identity loss for tgtB so that tgt domain style transfer can be better
    - no contrast augmentation for A, less contrast for B
    - additional style consistency for output B

    v9:
    similar to v5. tau=100, z=8
    """

    def __init__(self):
        self.train = Compose([        
            LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
            AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
            Copyd(keys='B', new_key='posB'),
            RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
            RandFlipd(keys='B', prob=0.5, spatial_axis=0),
            RandFlipd(keys='posB', prob=0.5, spatial_axis=0),
            # RandAffined(
            #     keys='posB', prob=1, 
            #     rotate_range=(0, 0, 10*np.pi/180),
            #     translate_range=(5, 5, 5),
            #     scale_range=(0.2, 0.2, 0.2),
            #     mode='bilinear',
            #     padding_mode='zeros'),
            # RandGridDistortiond(keys='posB', num_cells=5, prob=1.0, distort_limit=(-0.12,-0.1), mode='bilinear', padding_mode='zeros'),
            NormalizeForegroundd(keys=['A', 'B', 'posB']),
            RandAdjustContrastd(keys='A', prob=0.2, gamma=(0.8, 1.2)),
            # RandAdjustContrastd(keys=['B', 'posB'], prob=0.1, gamma=(0.8, 1.2)),
            # RandAdjustContrastd(keys='A', prob=0.4, gamma=(0.8, 1.2)),
            RandAdjustContrastd(keys=['B', 'posB'], prob=0.2, gamma=(0.8, 1.2)),
            ScaleIntensityRangePercentilesd(keys=['A', 'B', 'posB'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
            InvertT1d(keys=['A']),
            RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
            RandSpatialCropd(keys='posB', roi_size=(256, 144, 8), random_center=True, random_size=False),
            AdjustEdged(keys=['A_edge']),
            CastToTyped(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.float32, np.uint8, np.uint8)),
            ToTensord(keys=['A', 'B', 'posB', 'A_msk', 'A_edge']),])

        self.get_negatives = Compose([
            Copyd(keys='posB', new_key='negB'),
            RandSpatialCropSamplesd(keys='negB', roi_size=(256, 144, 8), num_samples=12, random_center=True, random_size=False),
            IntenTransform(keys='negB'),
            CastToTyped(keys=['negB'], dtype=np.float32),
            ToTensord(keys=['negB'])
        ])


class InvertT1d(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        # data['A'] = -data['A']
        data['A'][data['A_msk']==3] = -data['A'][data['A_msk']==3]
        # data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced

        # we add here!!!!!!!!!!!!!!!!!!!!!!!!!!!
        data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)

        # contrast v11
        # data['A'][(data['A_msk']==1) | (data['A_msk']==2)] *= -1
        return data


def IntenTransform(keys):
    seed = np.random.uniform()
    inten_list = [
        # RandAdjustContrastd(keys=keys, prob=1, gamma=(0.5, 1.5)),
        # RandBiasFieldd(keys=keys, prob=1, degree=3, coeff_range=(1e-5, 0.1)),
        # RandGaussianSmoothd(keys=keys, prob=1, sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.5,1.5),),  # v9=0.5, was 0.7
        # RandGaussianNoised(keys=keys, prob=1, mean=0.0, std=0.4),
        Compose([
            RandAdjustContrastd(keys=keys, prob=0.25, gamma=(0.5, 1.5)),
            RandBiasFieldd(keys=keys, prob=0.25, degree=3, coeff_range=(1e-5, 0.1)),
            RandGaussianSmoothd(keys=keys, prob=0.25, sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.5,1.5),),
            RandGaussianNoised(keys=keys, prob=0.25, mean=0.0, std=0.4)]),
    ]
    return random.choice(inten_list)


class Copyd(MapTransform):
    def __init__(self, keys, new_key) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.new_key = new_key

    def __call__(self, data):
        data[self.new_key] = copy.deepcopy(data[self.keys])
        return data


if __name__ == "__main__":
    
    data_dict = {
        'A': '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_12_ceT1.nii.gz', 
        'B': '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_135_T2.nii.gz', 
        'A_edge': '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_ukm_12_Label.nii.gz',
        'A_msk': '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_ukm_12_Label.nii.gz'}

    transform = modaTransform().train
    output = transform(data_dict)
    neg_dict = (modaTransform().get_negatives)({'posB': output['posB']})
    negB = [x['negB'] for x in neg_dict]
    # breakpoint()
    data_dict = {'A': output['A'], 'B': output['B'], 'PosB': output['posB'], 'NegB': negB}

    import matplotlib.pyplot as plt 
    fig = plt.figure(figsize=(20, 30))

    # Plot the image using Matplotlib
    fig.add_subplot(5, 3, 1) 
    plt.imshow(data_dict['A'][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("A") 

    fig.add_subplot(5, 3, 2) 
    plt.imshow(data_dict['B'][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("B") 

    fig.add_subplot(5, 3, 3) 
    plt.imshow(data_dict['PosB'][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Pos B") 

    fig.add_subplot(5, 3, 4) 
    plt.imshow(data_dict['NegB'][0][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #1") 

    fig.add_subplot(5, 3, 5) 
    plt.imshow(data_dict['NegB'][1][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #2") 

    fig.add_subplot(5, 3, 6) 
    plt.imshow(data_dict['NegB'][2][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #3") 

    fig.add_subplot(5, 3, 7) 
    plt.imshow(data_dict['NegB'][3][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #4") 

    fig.add_subplot(5, 3, 8) 
    plt.imshow(data_dict['NegB'][4][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #5") 

    fig.add_subplot(5, 3, 9) 
    plt.imshow(data_dict['NegB'][5][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #6") 

    fig.add_subplot(5, 3, 10) 
    plt.imshow(data_dict['NegB'][6][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #7") 

    fig.add_subplot(5, 3, 11) 
    plt.imshow(data_dict['NegB'][7][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #8") 

    fig.add_subplot(5, 3, 12) 
    plt.imshow(data_dict['NegB'][8][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #9") 

    fig.add_subplot(5, 3, 13) 
    plt.imshow(data_dict['NegB'][9][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #10") 

    fig.add_subplot(5, 3, 14) 
    plt.imshow(data_dict['NegB'][10][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #11") 

    fig.add_subplot(5, 3, 15) 
    plt.imshow(data_dict['NegB'][11][0, ..., 4].T, cmap='gray')
    plt.axis('off')
    plt.title("Neg B #12") 
    plt.show()
    

    #-------------------------------------
    # class modaTransform(object):
#     """This transform is used for style disentanglement learning w/ contrastive learning"""
#     def __init__(self):
#         self.train = Compose([        
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             CopyPosBd(keys='B'),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandFlipd(keys='posB', prob=0.5, spatial_axis=0),
#             RandAffined(keys='posB', prob=1.0, 
#                 rotate_range=(0, 0, 10*np.pi/180),
#                 translate_range=(0, 0, (-5, 5)),
#                 scale_range=(0.1, 0.1, 0.1),
#                 mode='bilinear',
#                 padding_mode='zeros'),
#             RandGridDistortiond(keys='posB', num_cells=5, prob=1.0, distort_limit=(-0.12,-0.1), mode='bilinear', padding_mode='zeros'),

#             NormalizeForegroundd(keys=['A', 'B']),
#             RandAdjustContrastd(keys='A', prob=0.3, gamma=(0.8, 1.2)),
#             ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandSpatialCropd(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
#             AdjustEdged(keys=['A_edge']),
#             CastToTyped(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'posB', 'A_msk', 'A_edge']),])

#         self.build_pairs = Compose([
#             CopyBd(keys='posB'),
#             RandSpatialCropSamplesd(keys='negB', roi_size=(256, 144, 8), num_samples=10, random_center=True, random_size=False),
#             IntenTransform(keys='negB'),
#             CastToTyped(keys=['negB'], dtype=np.float32),
#             ToTensord(keys=['negB'])
#         ])

#         # self.build_pairs = Compose([
#         #     LoadImaged(keys='posB'),
#         #     AddChanneld(keys='posB'),
#         #     NormalizeForegroundd(keys=['posB']),
#         #     ScaleIntensityRangePercentilesd(keys='posB', lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         #     RandFlipd(keys='posB', prob=0.5, spatial_axis=0),
#         #     RandSpatialCropd(keys='posB', roi_size=(256, 144, 8), random_center=True, random_size=False),
#         #     CopyBd(keys='posB'),
#         #     RandSpatialCropSamplesd(keys='negB', roi_size=(256, 144, 8), num_samples=8, random_center=True, random_size=False),
#         #     IntenTransform(keys=['B', 'posB']),
#         #     IntenTransform(keys='negB'),
#         #     CastToTyped(keys=['posB', 'negB'], dtype=(np.float32, np.float32)),
#         #     ToTensord(keys=['posB', 'negB'])
#         # ])



# class modaTransform(object):
#     """contrast v3"""
#     def __init__(self):
#         self.train = Compose([        
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             CopyPosBd(keys='B'),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandFlipd(keys='posB', prob=0.5, spatial_axis=0),
#             RandAffined(keys='posB', prob=1.0, 
#                 rotate_range=(0, 0, 10*np.pi/180),
#                 translate_range=(0, 0, (-5, 5)),
#                 scale_range=(0.1, 0.1, 0.1),
#                 mode='bilinear',
#                 padding_mode='zeros'),
#             RandGridDistortiond(keys='posB', num_cells=5, prob=1.0, distort_limit=(-0.12,-0.1), mode='bilinear', padding_mode='zeros'),

#             NormalizeForegroundd(keys=['A', 'B', 'posB']),
#             RandAdjustContrastd(keys='A', prob=0.3, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys=['B', 'posB'], prob=0.2, gamma=(0.8, 1.2)),
#             ScaleIntensityRangePercentilesd(keys=['A', 'B', 'posB'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandSpatialCropd(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
#             AdjustEdged(keys=['A_edge']),
#             CastToTyped(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'posB', 'A_msk', 'A_edge']),])

#         self.get_negatives = Compose([
#             Copyd(keys='posB', new_key='negB'),
#             RandSpatialCropSamplesd(keys='negB', roi_size=(256, 144, 8), num_samples=12, random_center=True, random_size=False),
#             IntenTransform(keys='negB'),
#             CastToTyped(keys=['negB'], dtype=np.float32),
#             ToTensord(keys=['negB'])
#         ])


# def select_any(x):
#     return x > -1

# def select_low(x):
#     return x > -0.99

# def SmartCropd():
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def is_sfov(self, img):
#         return (img.shape[0]!=256) or (img.shape[1]!=144) or (img.shape[2]!=8) 

#     def __call__(self, data):
#         if 
#         data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
#         data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
#         return data

# class modaTransform(object):
#     """Contrast v4"""
#     def __init__(self):
#         self.train = Compose([        
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AdjustEdged(keys=['A_edge']),

#             # positive pair
#             Copyd(keys='B', new_key='posB'),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandFlipd(keys='posB', prob=0.5, spatial_axis=0),
#             RandAffined(keys='posB', prob=1.0, 
#                 rotate_range=(0, 0, 10*np.pi/180),
#                 translate_range=(0, 0, (-5, 5)),
#                 scale_range=(0.1, 0.1, 0.1),
#                 mode='bilinear',
#                 padding_mode='zeros'),
#             RandGridDistortiond(keys='posB', num_cells=5, prob=1.0, distort_limit=(-0.12,-0.1), mode='bilinear', padding_mode='zeros'),

#             # same field of view. remove 
#             NormalizeForegroundd(keys=['A', 'B', 'posB']),
#             RandAdjustContrastd(keys='A', prob=0.3, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys=['B', 'posB'], prob=0.3, gamma=(0.8, 1.2)),
#             ScaleIntensityRangePercentilesd(keys=['A', 'B', 'posB'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             CropForegroundd(keys=['A', 'A_msk', 'A_edge', 'B', 'posB'], source_key='A', select_fn=select_low),
#             CropForegroundd(keys=['A', 'A_msk', 'A_edge', 'B', 'posB'], source_key='B', select_fn=select_low),
#             RandSpatialCropd(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
#             CropForegroundd(keys=['A', 'A_msk', 'A_edge', 'B', 'posB'], source_key='A', select_fn=select_any, k_divisible=(32, 16, 8)),
#             CastToTyped(keys=['A', 'B', 'posB', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'posB', 'A_msk', 'A_edge']),])

#         self.get_negatives = Compose([
#             Copyd(keys='posB', new_key='negB'),
#             RandSpatialCropSamplesd(keys='negB', roi_size=(256, 144, 8), num_samples=12, random_center=True, random_size=False),
#             IntenTransform(keys='negB'),
#             CastToTyped(keys=['negB'], dtype=np.float32),
#             ToTensord(keys=['negB'])
#         ])

#----------------------------------------------------


 # data = {'A': '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_12_ceT1.nii.gz',
    #         'A_msk': '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_ukm_12_Label.nii.gz',
    #         'B': '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_ukm_96_T2.nii.gz'}
    # transform = Compose([        
    #         LoadImaged(keys=['A', 'A_msk', 'B']),
    #         AddChanneld(keys=['A', 'A_msk', 'B']),
    #         NormalizeForegroundd(keys=['A', 'B']),
    #         RandAdjustContrastd(keys=['A', 'B'], prob=1, gamma=(0.8, 1.2)),
    #         ScaleIntensityRangePercentilesd(keys=['A', 'B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
    #         InvertT1d(keys=['A']),
    #         RandSpatialCropd(keys=['A', 'B'], roi_size=(256, 144, 8), random_center=True, random_size=False),
    #         CropForegroundd(keys=['A'], source_key='A', select_fn=select_fn),
    #         CropForegroundd(keys=['B'], source_key='B', select_fn=select_fn),
    #         SaveImaged(
    #                 keys=['A'], 
    #                 output_dir=f'/data/crossmoda2023/data/', 
    #                 output_postfix='', 
    #                 output_ext='.nii.gz', 
    #                 resample=False,
    #                 separate_folder=False,
    #                 print_log=False),
    #         SaveImaged(
    #                 keys=['B'], 
    #                 output_dir=f'/data/crossmoda2023/data/', 
    #                 output_postfix='', 
    #                 output_ext='.nii.gz', 
    #                 resample=False,
    #                 separate_folder=False,
    #                 print_log=False),
    #         ])

    # output = transform(data)




# def SpatialTransform(keys):
#     return Compose([
#         RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
#         RandZoomd(keys=keys, prob=0.3, min_zoom=0.9, max_zoom=1.1, 
#             mode='trilinear', padding_mode='constant', constant_values=-1),
#         Rand3DElasticd(keys=keys, sigma_range=(5,7), 
#             magnitude_range=(50,150), prob=1, mode='bilinear'),
#     ])


# def IntenTransform(keys):
#     return Compose([
#         RandAdjustContrastd(keys=keys, prob=0.2, gamma=(0.7, 3)),#(0.9, 1.2)),
#         RandBiasFieldd(keys=keys, degree=5, coeff_range=(0.01, 0.015), prob=0.2),
#         RandGibbsNoised(keys=keys, prob=0.2, alpha=(0.70, 0.8)),
#         RandGaussianSmoothd(keys=keys, sigma_x=(0.5,0.9), sigma_y=(0.5,0.9), sigma_z=(0.5,0.9), prob=0.2),
#         RandGaussianNoised(keys=keys, prob=0.2, mean=0.0, std=0.18),
#     ])


# def IntenTransform(keys):
#     return Compose([
#         RandAdjustContrastd(keys=keys, prob=0.3, gamma=(0.7, 2)),
#         RandBiasFieldd(keys=keys, prob=0.3, degree=5, coeff_range=(0.01, 0.1)),
#         RandGibbsNoised(keys=keys, prob=0.3, alpha=(0.7, 0.8)),
#         RandGaussianSmoothd(keys=keys, prob=0.3),
#         RandGaussianSharpend(keys=keys, prob=0.3),
#         RandGaussianNoised(keys=keys, prob=0.3, mean=0.0, std=0.1),
#     ])



        


# def IntenTransform(keys):
#     return Compose([
#         RandAdjustContrastd(keys=keys, prob=1, gamma=(0.8, 1.2)),
#         RandBiasFieldd(keys=keys, prob=0.3, coeff_range=(0.1, 0.2)),
#         RandGibbsNoised(keys=keys, prob=0.3, alpha=(0.1, 0.2)),
#         RandGaussianSmoothd(keys=keys, prob=0.3),
#         RandGaussianSharpend(keys=keys, prob=0.3),
#     ])


# class modaTransform(object):
#     """This transform is used for style disentanglement learning"""
#     def __init__(self):
#         self.train = Compose([        
#             # GetCoded(keys='B_paths'),
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             # RandAdjustContrastd(keys=['A', 'B'], prob=0.2, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys='A', prob=0.4, gamma=(0.8, 1.2)),
#             # RandAdjustContrastd(keys='B', prob=0.1, gamma=(0.8, 1.2)),
#             NormalizeForegroundd(keys=['A', 'B']),
#             # ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-0.2, b_max=1, clip=True, relative=False),
#             ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=(256, 144, 16), random_center=True, random_size=False),
#             RandSpatialCropd(keys=['A', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
#             RandSpatialCropSamplesd(keys='B', roi_size=(256, 144, 8), num_samples=8, random_center=True, random_size=False),
#             AdjustEdged(keys=['A_edge']),
#             CastToTyped(keys=['A', 'B', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'A_msk', 'A_edge']),])


# class modaTransform(object):
#     def __init__(self):
#         self.train = Compose([        
#             # GetCoded(keys='B_paths'),
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             # RandAdjustContrastd(keys=['A', 'B'], prob=0.2, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys='A', prob=0.4, gamma=(0.8, 1.2)),
#             # RandAdjustContrastd(keys='B', prob=0.1, gamma=(0.8, 1.2)),
#             NormalizeForegroundd(keys=['A', 'B']),
#             # ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-0.2, b_max=1, clip=True, relative=False),
#             ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
#             AdjustEdged(keys=['A_edge']),
#             CastToTyped(keys=['A', 'B', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'A_msk', 'A_edge']),])


# class InvertT1d(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         # data['A'] = -data['A']
#         data['A'][data['A_msk']==3] = 1  # modality where cochlea is enhanced
#         data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
#         return data



# class modaTransform(object):
#     def __init__(self):
#         self.train = Compose([        
#             GetCoded(keys='B_paths'),
#             LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge']),
#             AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge']),
#             # RandAdjustContrastd(keys=['A', 'B'], prob=0.2, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys='A', prob=0.4, gamma=(0.8, 1.2)),
#             RandAdjustContrastd(keys='B', prob=0.1, gamma=(0.8, 1.2)),
#             NormalizeForegroundd(keys=['A', 'B']),
#             ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0),
#             RandFlipd(keys='B', prob=0.5, spatial_axis=0),
#             RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=(256, 144, 8), random_center=True, random_size=False),
#             AdjustEdged(keys=['A_edge']),
#             CastToTyped(keys=['A', 'B', 'A_msk', 'A_edge'], dtype=(np.float32, np.float32, np.uint8, np.uint8)),
#             ToTensord(keys=['A', 'B', 'A_msk', 'A_edge']),])



# class tran(object):
#     def __init__(self):
#         self.train = Compose([        
#             GetCoded(keys='B_paths'),
#             LoadImaged(keys=['A', 'B', 'A_msk']),
#             AddChanneld(keys=['A', 'B', 'A_msk']),
#             RandAdjustContrastd(keys=['A', 'B'], prob=1, gamma=(0.8, 1.2)),
#             NormalizeForegroundd(keys=['A', 'B']),
#             ScaleIntensityRangePercentilesd(keys=['A', 'B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#             InvertT1d(keys=['A']),
#             RandFlipd(keys=['A', 'B'], prob=0, spatial_axis=0),
#             # RandSpatialCropd(keys=['A', 'B', 'A_msk'], roi_size=(256, 144, 12), random_center=True, random_size=False),
#             CastToTyped(keys=['A', 'B', 'A_msk'], dtype=(np.float32, np.float32, np.uint8)),
#             SaveImaged(
#                     keys=['A'], 
#                     output_dir=f'/data/crossmoda2023/data/crossmoda23_training', 
#                     output_postfix='', 
#                     output_ext='.nii.gz', 
#                     resample=False,
#                     separate_folder=False,
#                     print_log=False)])
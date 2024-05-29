import os
import os.path as osp
import numpy as np
from glob import glob
from monai.transforms import *
from tqdm import tqdm 
import nibabel as nib
import SimpleITK as sitk
import pdb
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball, disk
from skimage import filters


class merge(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['out'] = data['label1'] + data['label2']
        return data


def resample_reorient(image_dir, label_dir, output_dir, mode='src'):
    img_paths = sorted(glob(image_dir + '/*.nii.gz'))
    with tqdm(total=len(img_paths)) as pbar:
        for path in img_paths:
            if mode == 'src':
                mask_path = osp.join(label_dir, osp.basename(path)).replace('ceT1', 'Label')
            elif mode == 'tgt':
                mask_path = osp.join(label_dir, osp.basename(path)).replace('_0000', '')

            data_dict = {'image': path, 'label': mask_path}
            tran = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=["image", 'label']),
                Orientationd(keys=["image", 'label'], axcodes="LPS"), 
                Spacingd(
                    keys=['image', 'label'], 
                    pixdim=(0.41015625, 0.41015625, 1), 
                    mode=("bilinear", "nearest")),
                SaveImaged(
                    keys=['image'], 
                    output_dir=osp.join(output_dir, f'{mode}ImageRR'), 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False),

                SaveImaged(
                    keys=['label'], 
                    output_dir=osp.join(output_dir, f'{mode}LabelRR'), 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False),
                ])

            output = tran(data_dict)
            pbar.update(1)


def crop_roi(image_dir, label_dir, output_dir, mode='src'):
    img_paths = sorted(glob(image_dir + '/*.nii.gz'))
    with tqdm(total=len(img_paths)) as pbar:
        for path in img_paths:
            # print(path, 'etz_120' in path)
            
            # if 'crossmoda2023_etz_1_ceT1' not in path:
            #     continue

            # if 'etz_120' not in path:
            #     continue

            # if 'ukm_77' not in path:
            #     continue

            # if 'ukm_50' not in path:
            #     continue

            if mode == 'src':
                mask_path = osp.join(label_dir, osp.basename(path)).replace('ceT1', 'Label')
            elif mode == 'tgt':
                mask_path = osp.join(label_dir, osp.basename(path)).replace('_0000', '')

            data_dict = {'image': path, 'label': mask_path}
            msk = nib.load(mask_path).get_fdata()
            
            if mode == 'src':
                msk[msk<3] = 0
                msk[msk!=0] = 1

            x_min = np.where(msk==1)[0].min()
            x_max = np.where(msk==1)[0].max()
            y_min = np.where(msk==1)[1].min()
            y_max = np.where(msk==1)[1].max()
            z_min = np.where(msk==1)[2].min()
            z_max = np.where(msk==1)[2].max()
            dim = msk.shape

            # print(x_min, x_max, y_min, y_max, z_min, z_max)

            if x_max - x_min < 50: # only a single cochlea is detected
                print('not both cochleae were detected..')
                print(path)
                if x_max > dim[0]/2:  # cochlea on the right side
                    x_min = x_max + 80 -256
                else:  # cochlea on the left
                    x_max = x_min - 80 + 256


            

            # center = ((x_max+x_min)/2, (y_max+y_min)/2+30, (z_max+z_min)/2)
            # print('center: ', center)

            roi_size = (256, 144, 32)

            # expand the box
            y_min -= 25
            y_max = y_min + 144
            x_min -= 25
            x_max += 25
            z_min -= 10
            z_max += 10

            if x_max - x_min < 256:
                pad = 256 - (x_max - x_min)
                x_min -= int(pad /2)
                x_max += pad - int(pad /2)

            # if y_max - y_min < 144:
            #     pad = 144 - (y_max - y_min)
            #     y_min -= int(pad /3)
            #     y_max += pad - int(pad /3)

            if z_max - z_min < 32:
                pad = 32 - (z_max - z_min)
                z_min -= int(pad /2)
                z_max += pad - int(pad /2)

            # print(x_min, x_max, y_min, y_max, z_min, z_max)

            # limit the range
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            z_min = max(0, z_min)
            x_max = min(int(dim[0]), x_max)
            y_max = min(int(dim[1]), y_max)
            z_max = min(int(dim[2]), z_max)

            # print(x_min, x_max, y_min, y_max, z_min, z_max)
            # print(x_max-x_min, y_max-y_min, z_max-z_min)

            # import pdb
            # pdb.set_trace()


            tran = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=["image", 'label']),
                # SpatialCropd(keys=["image", "label"], roi_center=center, roi_size=roi_size),
                SpatialCropd(keys=["image", "label"], roi_start=(x_min, y_min, z_min), roi_end=(x_max, y_max, z_max)),
                SpatialPadd(keys=["image", "label"], spatial_size=(256, 144, 32), mode='constant', constant_values=0),
                CenterSpatialCropd(keys=["image", 'label'], roi_size=(256, 144, 32)),
                SaveImaged(
                    keys=['image'], 
                    output_dir=osp.join(output_dir, f'{mode}ImageROI'), 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False),

                SaveImaged(
                    keys=['label'], 
                    output_dir=osp.join(output_dir, f'{mode}LabelROI'), 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False),
                ])

            output = tran(data_dict)
            pbar.update(1)



def threshold(x):
    return (x==1) | (x==2)



class PseudoImage1d(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['T1'] = -data['T1']
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



class DilateMaskd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        radius = 5
        self.structuring_element = ball(radius)

    def __call__(self, data):
        # remove cochlea
        data['label'][data['label']==3] = 0
        # binarize label
        data['label'][data['label']!=0] = 1
        data['label'] = dilated_volume = binary_dilation(data['label'][0], structure=self.structuring_element)
        data['label'] = data['label'][None, ...]
        return data


class GetEdgeMap(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        radius = 5
        self.structuring_element = ball(radius)

    def __call__(self, data):
        data['T1'][None, ...] = filters.sobel(data['T1'][0])
        data['T2'][None, ...] = filters.sobel(data['T2'][0])
        return data


def crop_local(T1_path, T2_path, mask_path, output_dir='/data/crossmoda2023/tmp'):
    data_dict = {'T1': T1_path, 'T2': T2_path, 'label': mask_path}
    tran = Compose([
        LoadImaged(keys=['T1', 'T2', 'label']),
        AddChanneld(keys=['T1', 'T2', 'label']),

        NormalizeForegroundd(keys=['T1']),
        ScaleIntensityRangePercentilesd(keys='T1', lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        PseudoImage1d(keys='T1'),

        CropForegroundd(
            keys=['T1', 'T2', 'label'], 
            source_key='label', 
            select_fn=threshold, 
            channel_indices=None, 
            margin=20),

        SaveImaged(
            keys=['T1'], 
            output_dir=output_dir, 
            output_postfix='T1', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),

        SaveImaged(
            keys=['T2'], 
            output_dir=output_dir, 
            output_postfix='T2', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),

        SaveImaged(
            keys=['label'], 
            output_dir=output_dir,
            output_postfix='label', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),

        # DilateMaskd(keys=['label']),
        GetBoundaryMaskd(keys=['label']),
        SaveImaged(
            keys=['label'], 
            output_dir=output_dir,
            output_postfix='reg_mask', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),

        GetEdgeMap(keys=['T1', 'T2']),
        SaveImaged(
            keys=['T1'], 
            output_dir=output_dir, 
            output_postfix='T1_edge', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),

        SaveImaged(
            keys=['T2'], 
            output_dir=output_dir, 
            output_postfix='T2_edge', 
            output_ext='.nii.gz', 
            resample=False,
            separate_folder=False,
            print_log=False),
        ])

    output = tran(data_dict)


# class GetBoundaryMaskd(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys
#         radius = 4
#         self.structuring_element = ball(radius)
#         # self.structuring_element = disk(radius)

#     def __call__(self, data):
#         # remove cochlea
#         # data['label'][data['label']==3] = 0
#         # binarize label
#         co_mask = data['label'].copy()
#         vs_mask = data['label'].copy()

#         co_mask[co_mask<3]=0
#         co_mask[co_mask!=0]=1
#         co_mask = binary_dilation(co_mask[0], structure=self.structuring_element).astype('uint8')
    
#         vs_mask[vs_mask>2]=0
#         vs_mask[vs_mask!=0]=1
#         vs_mask = binary_dilation(vs_mask[0], structure=ball(10)).astype('uint8')

#         output = vs_mask
#         output[co_mask==1] = 2

#         # print(np.unique(output))


#         # data['label'][data['label']!=0] = 1
        
#         # label = data['label'][0]
#         # output = []
#         # for z in range(label.shape[2]):
#         #     slc = label[:, :, z]
#         #     dilate = binary_dilation(slc, structure=self.structuring_element)
#         #     erode = binary_erosion(slc, structure=self.structuring_element)
#         #     dilate[erode] = False
#         #     output.append(dilate)
#         # output = np.transpose(np.array(output), (1, 2, 0))

#         # dilate = binary_dilation(data['label'][0], structure=self.structuring_element)
#         # erode = binary_erosion(data['label'][0], structure=self.structuring_element)
#         # dilate[erode] = False
#         # output = dilate

#         data['label'] = output[None, ...]
#         return data



# V2
# class GetBoundaryMaskd(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys
#         radius = 1
#         self.structuring_element = ball(radius)
#         # self.structuring_element = disk(radius)

#     def __call__(self, data):
#         # remove cochlea
#         # data['label'][data['label']==3] = 0
#         # binarize label
#         co_mask = data['label'].copy()
#         vs_mask = data['label'].copy()

#         co_mask[co_mask<3]=0
#         co_mask[co_mask!=0]=1
#         co_mask = binary_dilation(co_mask[0], structure=ball(4)).astype('uint8')
    
#         vs_mask[vs_mask>2]=0
#         vs_mask[vs_mask!=0]=1
        
#         # dilate = binary_dilation(vs_mask[0], structure=ball(2)).astype('uint8')
#         # erode  = binary_erosion(vs_mask[0], structure=ball(1)).astype('uint8')
#         # dilate[erode==1] = 0

#         # output = dilate
#         # output[co_mask==1] = 2

#         label = vs_mask[0]
#         output = []
#         for z in range(label.shape[2]):
#             slc = label[:, :, z]
#             dilate = binary_dilation(slc, structure=disk(4))
#             erode = binary_erosion(slc, structure=disk(4))
#             dilate[erode] = False
#             output.append(dilate)
#         output = np.transpose(np.array(output), (1, 2, 0)).astype('uint8')

#         output[co_mask==1] = 2

#         data['label'] = output[None, ...]
#         return data



#  V4
class GetBoundaryMaskd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        radius = 2
        self.structuring_element = ball(radius)
        # self.structuring_element = disk(radius)

    def __call__(self, data):
        # remove cochlea
        # data['label'][data['label']==3] = 0
        # binarize label
        co_mask = data['label'].copy()
        vs_mask = data['label'].copy()

        co_mask[co_mask<3]=0
        co_mask[co_mask!=0]=1
        co_mask = binary_dilation(co_mask[0], structure=self.structuring_element).astype('uint8')
    
        vs_mask[vs_mask>2]=0
        vs_mask[vs_mask!=0]=1
        vs_mask = binary_erosion(vs_mask[0], structure=ball(2.4)).astype('uint8')

        output = vs_mask
        output[co_mask==1] = 2

        # print(np.unique(output))


        # data['label'][data['label']!=0] = 1
        
        # label = data['label'][0]
        # output = []
        # for z in range(label.shape[2]):
        #     slc = label[:, :, z]
        #     dilate = binary_dilation(slc, structure=self.structuring_element)
        #     erode = binary_erosion(slc, structure=self.structuring_element)
        #     dilate[erode] = False
        #     output.append(dilate)
        # output = np.transpose(np.array(output), (1, 2, 0))

        # dilate = binary_dilation(data['label'][0], structure=self.structuring_element)
        # erode = binary_erosion(data['label'][0], structure=self.structuring_element)
        # dilate[erode] = False
        # output = dilate

        data['label'] = output[None, ...]
        return data

if __name__ == "__main__":


    # crop_local(
        # T1_path='/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_etz_6_ceT1.nii.gz', 
        # T2_path='/data/nnUNetV2/data/nnUNet_raw/Dataset504_fakeT2/imagesTr/crossmoda2023_etz_6_0000.nii.gz', 
        # mask_path='/data/nnUNetV2/data/nnUNet_raw/Dataset504_fakeT2/labelsTr/crossmoda2023_etz_6.nii.gz')

    label_dir = '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI'
    paths = sorted(glob(label_dir + '/*.nii.gz'))

    for path in paths:
        print(path)
        data_dict = {'label': path}
        tran = Compose([
            LoadImaged(keys=['label']),
            AddChanneld(keys=['label']),
            GetBoundaryMaskd(keys=['label']),
            SaveImaged(
                keys=['label'], 
                output_dir='/data/crossmoda2023/data/crossmoda23_training/srcEdgeV4ROI',
                output_postfix='', 
                output_ext='.nii.gz', 
                resample=False,
                separate_folder=False,
                print_log=False),
            ])

        output = tran(data_dict)


    # validation set

    # resample_reorient(
    #     '/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
    #     '/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred',
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation',
    #     mode='tgt')

    # crop_roi(
    #     '/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
    #     '/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR',
    #     '/data/crossmoda2023/data/crossmoda23_validation',
    #     mode='tgt')

    # crop_roi(
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtImageRR', 
    #     '/data/crossmoda2023/data/crossmoda23_training/tgtLabelRR',
    #     mode='tgt')

    # crop_roi(
    #     '/data/crossmoda2023/data/crossmoda23_training/srcImageRR', 
    #     '/data/crossmoda2023/data/crossmoda23_training/srcLabelRR',
    #     mode='src')

    # resample_reorient(
    #     '/data/crossmoda2023/data/crossmoda23_training/TrainingSourceImage', 
    #     '/data/crossmoda2023/data/crossmoda23_training/TrainingSourceLabel',
    #     mode='src')
    
    # resample_reorient(
    #     '/data/crossmoda2023/data/crossmoda23_training/TrainingTarget', 
    #     '/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Pred',
    #     mode='tgt')

    # paths = glob('/data/nnUNetV2/data/nnUNet_raw/Dataset503_LocT2/labelsTr' + '/*.nii.gz')

    # paths = ['/data/crossmoda2023/data/crossmoda23_training/srcLabelRR/crossmoda2023_etz_1_Label.nii.gz']

    # for path in paths:
    #     label = sitk.ReadImage(path)
    #     img = sitk.GetArrayFromImage(label)
    #     # img[img != 0] = 1

    #     # pdb.set_trace()
    #     img = sitk.GetImageFromArray(img)
    #     img.CopyInformation(label)
    #     sitk.WriteImage(img, path) 

    # label1 = sitk.ReadImage('/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Pred/crossmoda2023_ukm_79_T2_1.nii.gz') 
    # label2 = sitk.ReadImage('/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Pred/crossmoda2023_ukm_79_T2_2.nii.gz')

    # label1 = sitk.GetArrayFromImage(label1)
    # label2 = sitk.GetArrayFromImage(label2)
    # label3 = label1 + label2

    # label3 = sitk.GetImageFromArray(label3)
    # label3.CopyInformation(sitk.ReadImage('/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Pred/crossmoda2023_ukm_79_T2_1.nii.gz') )
    # sitk.WriteImage(label3, '/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Pred/crossmoda2023_ukm_79_T2_merge.nii.gz')
    
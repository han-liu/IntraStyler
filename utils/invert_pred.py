import os
import os.path as osp
import numpy as np
from glob import glob
from monai.transforms import *
from tqdm import tqdm 
import nibabel as nib
import SimpleITK as sitk


def inv_resample_reorient(image_dir, label_dir, pred_dir, output_dir):
    img_paths = sorted(glob(image_dir + '/*.nii.gz'))
    with tqdm(total=len(img_paths)) as pbar:
        for path in img_paths:
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
                ])

            output = tran(data_dict)

            # get the predition from nnU-Net
            pred_path = osp.join(pred_dir, osp.basename(mask_path))  # need modification
            pred = Compose([LoadImaged(keys=['label']), AddChanneld(keys=['label'])])({'label': pred_path})['label']
            output['label'] = pred
            inverted_data = tran.inverse(output)
            inver = Compose([
                SaveImaged(
                    keys=['label'], 
                    output_dir=output_dir, 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False)
                ])(inverted_data)

            pbar.update(1)


def inv_crop_roi(image_dir, label_dir, pred_dir, output_dir):
    """
    image_dir: full images T2
    label_dir: pseudo labels of cochlea T2 (used for localization)
    pred_dir: ROI predition from segmentation network
    output_dir: the output predition in original space 
    """
    img_paths = sorted(glob(image_dir + '/*.nii.gz'))
    with tqdm(total=len(img_paths)) as pbar:
        for path in img_paths:
            
            mask_path = osp.join(label_dir, osp.basename(path)).replace('_0000', '')
            data_dict = {'image': path, 'label': mask_path}
            msk = nib.load(mask_path).get_fdata()
            
            x_min = np.where(msk==1)[0].min()
            x_max = np.where(msk==1)[0].max()
            y_min = np.where(msk==1)[1].min()
            y_max = np.where(msk==1)[1].max()
            z_min = np.where(msk==1)[2].min()
            z_max = np.where(msk==1)[2].max()
            dim = msk.shape

            if x_max - x_min < 50: # only a single cochlea is detected
                print('not both cochleae were detected..')
                print(path)
                if x_max > dim[0]/2:  # cochlea on the right side
                    x_min = x_max + 80 -256
                else:  # cochlea on the left
                    x_max = x_min - 80 + 256

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

            if z_max - z_min < 32:
                pad = 32 - (z_max - z_min)
                z_min -= int(pad /2)
                z_max += pad - int(pad /2)

            # limit the range
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            z_min = max(0, z_min)
            x_max = min(int(dim[0]), x_max)
            y_max = min(int(dim[1]), y_max)
            z_max = min(int(dim[2]), z_max)

            tran = Compose([
                LoadImaged(keys=['image', 'label']),
                AddChanneld(keys=["image", 'label']),
                SpatialCropd(keys=["image", "label"], roi_start=(x_min, y_min, z_min), roi_end=(x_max, y_max, z_max)),
                SpatialPadd(keys=["image", "label"], spatial_size=(256, 144, 32), mode='constant', constant_values=0),
                CenterSpatialCropd(keys=["image", 'label'], roi_size=(256, 144, 32)),
                ])

            output = tran(data_dict)

            # get the predition from nnU-Net
            pred_path = osp.join(pred_dir, osp.basename(mask_path))  # need modification

            pred = Compose([
                LoadImaged(keys=['label']),
                AddChanneld(keys=['label'])])({'label': pred_path})['label']

            output['label'] = pred
            inverted_data = tran.inverse(output)
            inver = Compose([
                SaveImaged(
                    keys=['label'], 
                    output_dir=output_dir, 
                    output_postfix='', 
                    output_ext='.nii.gz', 
                    resample=False,
                    separate_folder=False,
                    print_log=False)
                ])(inverted_data)

            pbar.update(1)


if __name__ == "__main__":
    
    # inv_crop_roi(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_basic/basic',   #  need modification 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_basic/itm')

    # inv_resample_reorient(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_basic/itm', 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_basic/final')

    # inv_crop_roi(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_slerp/slerp',   #  need modification 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_slerp/itm')

    # inv_resample_reorient(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_slerp/itm', 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/own_slerp/final')


    # ########################
    # inv_crop_roi(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/NoAdapt/predit',   #  need modification 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/NoAdapt/itm')

    # inv_resample_reorient(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/NoAdapt/itm', 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/NoAdapt/final')

    # ########################
    inv_crop_roi(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
        pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/predict_v2',   #  need modification 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/itm_v2')

    inv_resample_reorient(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
        pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/itm_v2', 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/final_v2')

    inv_crop_roi(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
        pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/predict_v3',   #  need modification 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/itm_v3')

    inv_resample_reorient(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
        pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/itm_v3', 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Sep/final_v3')

    ########################
    # inv_crop_roi(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Dyn/predit',   #  need modification 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Dyn/itm')

    # inv_resample_reorient(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Dyn/itm', 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/Dyn/final')

    

    inv_crop_roi(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
        pred_dir='/data/crossmoda2023/segmentation/checkpoints/basic_no_over/results_current_model',   #  need modification 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/basic_no_over/itm')

    inv_resample_reorient(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
        pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/basic_no_over/itm', 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/basic_no_over/final')

    inv_crop_roi(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
        pred_dir='/data/crossmoda2023/segmentation/checkpoints/slerp_no_over/results_current_model',   #  need modification 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/slerp_no_over/itm')

    inv_resample_reorient(
        image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
        label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
        pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/slerp_no_over/itm', 
        output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/slerp_no_over/final')


    # inv_crop_roi(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImageRR', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtLabelRR', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/0706/ensemble',   #  need modification 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/0706/itm')

    # inv_resample_reorient(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImage', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_validation/tgtImagePred', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/0706/itm', 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_validation/submission/0706/final')
    

    # inv_crop_roi(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_training/tgtImageRR', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_training/tgtLabelRR', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_training/PseudoLabel515',   #  need modification 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_training/itm')

    # inv_resample_reorient(
    #     image_dir='/data/crossmoda2023/data/crossmoda23_training/TrainingTarget', 
    #     label_dir='/data/crossmoda2023/data/crossmoda23_training/TrainingTarget_Pred', 
    #     pred_dir='/data/crossmoda2023/data/crossmoda23_training/itm', 
    #     output_dir='/data/crossmoda2023/data/crossmoda23_training/final')
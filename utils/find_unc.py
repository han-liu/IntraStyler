import numpy as np
import nibabel as nib
import os.path as osp
from glob import glob
from cc3d import connected_components
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_closing, binary_opening


def entropy_3d_volume(vol_input):
    vol_input = vol_input.astype(dtype='float32')
    dims = vol_input.shape
    reps = dims[0]
    entropy = np.zeros(dims[2:], dtype='float32')
    threshold = 0.00005
    vol_input[vol_input <= 0] = threshold

    if len(dims) == 5:
        for channel in range(dims[1]):
            t_vol = np.squeeze(vol_input[:, channel, :, :, :])
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            t_entropy = -np.multiply(t_avg, t_log)
            entropy = entropy + t_entropy
    else:
        t_vol = np.squeeze(vol_input)
        t_sum = np.sum(t_vol, axis=0)
        t_avg = np.divide(t_sum, reps)
        t_log = np.log(t_avg)
        t_entropy = -np.multiply(t_avg, t_log)
        entropy = entropy + t_entropy
    return entropy


def variance_3d_volume(vol_input):
    vol_input = vol_input.astype(dtype='float32')
    dims = vol_input.shape
    threshold = 0.0005
    vol_input[vol_input<=0] = threshold
    breakpoint()
    vari = np.nanvar(vol_input, axis=0)
    variance = np.sum(vari, axis=0)
    # variance = np.expand_dims(variance, axis=0)
    # variance = np.expand_dims(variance, axis=0)
    return vari#variance


def generate_unc():
    # paths = glob('/data/crossmoda2023/data/crossmoda23_training/tgtImageROI_thin_pred_511/*.npz')
    paths = sorted(glob('/data/crossmoda2023/data/crossmoda23_training/PseudoLabel511/*.nii.gz'))

    for path in paths:
        data = dict(np.load(path, allow_pickle=True))['probabilities']
        entropy = 0

        threshold = 0.00005
        data[data <= 0] = threshold
        for channel in range(4):
            # if channel not in [0, 1]:
            #     continue
            t_vol = data[channel, ...]
            t_log = np.log(t_vol)
            t_entropy = -np.multiply(t_vol, t_log)
            entropy = entropy + t_entropy

        unc = entropy
        # unc = data[1, ...]
        unc = np.moveaxis(unc, (0, 1, 2), (2, 1, 0))

        # breakpoint()
        ref_img = nib.load(path.replace('tgtImageROI_thin_pred_511', 'tgtImageROI_thin').replace('.npz', '_0000.nii.gz'))
        new_img = nib.Nifti1Image(unc, ref_img.affine, ref_img.header)

        # Save the new image
        nib.save(new_img, osp.join('/data/crossmoda2023/data/crossmoda23_training/tgtImageROI_thin_pred_511_unc', osp.basename(path).replace('.npz', '.nii.gz')))


def rank_unc():
    # paths = glob('/data/crossmoda2023/data/crossmoda23_training/tgtImageROI_thin_pred_511/*.npz')
    paths = sorted(glob('/data/crossmoda2023/data/crossmoda23_training/PseudoLabel511/*.npz'))
    mask_paths = sorted(glob('/data/crossmoda2023/data/crossmoda23_training/PseudoLabel511/*.nii.gz'))
    scores = []

    for path in paths:
        # if path != paths[-1]:
        #     continue
        mask_path = path.replace('.npz', '.nii.gz')
        mask = nib.load(mask_path).get_fdata()
        vol = (mask==1).sum() + (mask==2).sum()

        data = dict(np.load(path, allow_pickle=True))['probabilities']
        entropy = 0
        threshold = 0.00005
        data[data <= 0] = threshold

        for channel in range(4):
            # if channel not in [0, 1]:
            #     continue
            t_vol = data[channel, ...]
            t_log = np.log(t_vol)
            t_entropy = -np.multiply(t_vol, t_log)
            entropy = entropy + t_entropy

        score = entropy.mean()/vol
        scores.append(score)

    paths = [osp.basename(path)[:-7] for path in paths]
    pairs = sorted(zip(scores, paths))
    sorted_paths = [b for a, b in pairs]
    breakpoint()




def check_label():
    paths = sorted(glob('/data/crossmoda2023/data/crossmoda23_training/PseudoLabel511/*.nii.gz'))

    structure = np.ones((3, 3, 3)) 

    for path in paths:
        # if path == '/data/crossmoda2023/data/crossmoda23_training/PseudoLabel511/crossmoda2023_ldn_84_T2.nii.gz':

        #     breakpoint()
        mask = nib.load(path).get_fdata()
        closed_mask = (mask)
        _, num1 = label(binary_opening(binary_closing(mask==1).astype('uint8'), structure=structure))
        _, num2 = label(binary_opening(binary_closing(mask==2).astype('uint8'), structure=structure))
        _, num3 = label(binary_opening(binary_closing(mask==3).astype('uint8'), structure=structure))

        if num1>1 or num2>1 or num3<2 or (num1==0 and num2==0):
            print(path, num1, num2, num3)
        # breakpoint()


def filter_tumor_label():
    paths = sorted(glob('/data/crossmoda2023/data/crossmoda23_training/final_filtered/*.nii.gz'))
    for path in paths:
        img = nib.load(path)
        new_data = img.get_fdata()
        new_data[new_data != 3] = 0
        new_data[new_data != 0] = 1
        new_img = nib.Nifti1Image(new_data, img.affine, img.header)
        nib.save(new_img, path)
        


if __name__ == "__main__":
    # generate_unc()

    # check_label()
    # rank_unc()

    filter_tumor_label()
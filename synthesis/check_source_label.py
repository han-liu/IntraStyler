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
import matplotlib.pyplot as plt


if __name__ == "__main__":

    source_label_dir = '/data/crossmoda2023/data/crossmoda23_training/srcLabelROI'
    paths = sorted(glob(source_label_dir + '/*.nii.gz'))

    filenames, label1, label2, label3, label4, stds = [], [], [], [], [], []
    
    with tqdm(total=len(paths)) as pbar:
        for path in paths:
            mask = LoadImage(image_only=True)(path)
            lab1 = mask[mask==1].sum()
            lab2 = mask[mask==2].sum()
            lab3 = mask[mask==3].sum()
            lab4 = lab1 + lab2
            image = LoadImage(image_only=True)(path.replace('srcLabelROI', 'srcImageROI').replace('Label', 'ceT1'))
            std = np.std(image[(mask==1) | (mask==2)]) * lab4

            filenames.append(osp.basename(path)[14:-13])
            label1.append(lab1)
            label2.append(lab2)
            label3.append(lab3)
            label4.append(lab4)
            stds.append(std)
            pbar.update(1)

    small_inside = [x for _, x in sorted(zip(label1, filenames))]
    large_inside = small_inside[::-1]

    small_outside = [x for _, x in sorted(zip(label2, filenames))]
    large_outside = small_outside[::-1]

    small_coch = [x for _, x in sorted(zip(label3, filenames))]
    large_coch = small_coch[::-1]

    small_tumor = [x for _, x in sorted(zip(label4, filenames))]
    small_std = [x for _, x in sorted(zip(stds, filenames))]

    cutoff = np.percentile(sorted(label1), 10)
    temp = np.array(sorted(label1))
    list1 = small_inside[:len(temp[temp <= cutoff])]

    cutoff = np.percentile(sorted(label1), 90)
    temp = np.array(sorted(label1))
    list2 = small_inside[-len(temp[temp >= cutoff]):]

    cutoff = np.percentile(sorted(label4), 90)
    temp = np.array(sorted(label4))
    list3 = small_tumor[-len(temp[temp >= cutoff]):]
    breakpoint()

    cutoff = np.percentile(sorted(stds), 90)
    temp = np.array(sorted(stds))
    list4 = small_std[-len(temp[temp >= cutoff]):]
    breakpoint()

    # plt.hist(label1, bins=226)
    # plt.show()
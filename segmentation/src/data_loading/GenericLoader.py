#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 12/11/2023


from glob import glob
import numpy as np
import os.path as osp
from utils.util import update_log, get_time
from monai.data import Dataset, DataLoader


OVERSAMPLE_RATIO = 4


class GenericLoader(object):
    def __init__(self, tr, opt, phase:str='train'):
        self.tr = tr
        self.phase = phase
        self.expr_dir = opt.expr_dir
        self.run_log = osp.join(self.expr_dir, 'run.log')
        self.batch_size = opt.batch_size
        self.num_workers = opt.num_workers
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        train_paths, val_paths, test_paths = [], [], []

        path_dict = np.load('paths.npz', allow_pickle=True)
        src_paths = path_dict['source'].tolist()
        lab_paths = path_dict['labels'].tolist()
        tgt_paths = path_dict['target'].tolist()

        # Add oversampling cases: 54 cases
        # hetero_ids = ["etz_6", "etz_17", "etz_25", "etz_27", "etz_46", "etz_69", "etz_75", "etz_76", "etz_81",
        #     "etz_105", "ldn_42", "ldn_58", "ldn_64", "ldn_69", "ldn_70", "ldn_78", "ukm_3", "ukm_23", "ukm_25", "ukm_34", "ukm_38"]
        # small_ids = ["etz_3", "etz_5", "etz_18", "etz_22", "etz_23", "etz_34", "etz_47", "etz_65", "etz_71", "etz_73", "etz_77",
        #     "etz_87", "etz_95", "etz_98", "etz_102", "ldn_4", "ldn_10", "ldn_13", "ldn_34", "ldn_37", "ldn_39", "ldn_51", "ldn_74",
        #     "ldn_75", "ldn_77", "ukm_11", "ukm_13", "ukm_14", "ukm_16", "ukm_33", "ukm_39", "ukm_42", "ukm_43"]
        # all_ids = (hetero_ids + small_ids) * OVERSAMPLE_RATIO

        # os_src_paths = [f'/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_{os_id}_ceT1.nii.gz' for os_id in all_ids]
        # os_lab_paths = [f'/data/crossmoda2023/data/crossmoda23_training/srcLabelROI/crossmoda2023_{os_id}_Label.nii.gz' for os_id in all_ids]

        # src_paths += os_src_paths
        # lab_paths += os_lab_paths

        # src_paths = src_paths * 2
        # lab_paths = lab_paths * 2

        if self.phase == 'train':
            train_ds = [{'image': src_paths[i], 'label': lab_paths[i], 'ref_paths': tgt_paths} for i in range(len(src_paths))]
            val_ds = train_ds[-1]  # just a place holder
            train_ds = Dataset(data=train_ds, transform=self.tr.translate)
            val_ds = Dataset(data=val_ds, transform=self.tr.infer)

            self.train_ds = DataLoader(
                train_ds, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers)

            self.val_ds = DataLoader(
                val_ds, 
                batch_size=1, 
                shuffle=False, 
                num_workers=self.num_workers)
        else:
            tgt_paths = sorted(glob(opt.test_dir + '/*.nii.gz'))
            test_ds = [{'image': path} for path in tgt_paths]
            print(f'Number of testing images: {len(test_ds)}')
            test_ds = Dataset(data=test_ds, transform=self.tr.infer)
            
            self.test_ds = DataLoader(
                test_ds, 
                batch_size=1, 
                shuffle=False, 
                num_workers=self.num_workers)

    def get_data(self):
        if self.phase == 'train':
            return self.train_ds, self.val_ds
        else:
            return self.test_ds

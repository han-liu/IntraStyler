#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 09/02/2023


""" ***** Program description *****
This class is used to make inference on independent testing sets. 
Evaluation metrics include: (1) Dice Score (2) HD95 and (3) ASSD
"""


import os.path as osp
import torch
import numpy as np
from numpy import inf
from time import time
from tqdm import tqdm
from monai.metrics import *
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from utils.util import update_log, get_time, mkdir
import SimpleITK as sitk


class GenericPredictor(object):
    def __init__(self, data, model, opt):
        self.opt = opt
        self.infer_log = osp.join(self.opt.expr_dir, 'inference.log')
        self.test_ds = data.get_data()
        self.model = model.cuda() if torch.cuda.is_available() else model

        if self.opt.multi_gpu and torch.cuda.device_count() > 1:
            print(f"multiple GPUs are used: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model).cuda().module
            
        self.include_background = False
        self.ckpt_path = osp.join(self.opt.checkpoints_dir, self.opt.name, self.opt.epoch + '.pth')
        self.model.load_state_dict(torch.load(self.ckpt_path)['state_dict'])
        update_log(f"model and optimizer are initialized from {self.ckpt_path}", self.infer_log)

        if self.opt.save_output:
            self.result_dir = osp.join(self.opt.expr_dir, f'results_{self.opt.epoch}')
            mkdir(self.result_dir)

    def post_pred(self, pred):
        pred = decollate_batch(pred)[0]
        pred = AsDiscrete(argmax=True, to_onehot=self.opt.num_classes)(pred)
        pred = pred.unsqueeze(0)
        return pred

    def post_label(self, target):
        return one_hot(target, self.opt.num_classes, dim=1)

    def compute_metrics(self, pred, target, include_background):
        dice = np.array(compute_meandice(pred, target)[0].cpu())
        if not include_background:
            dice = dice[1:]
        hd = np.array(compute_hausdorff_distance(pred, target, include_background, percentile=95)[0].cpu())
        assd = np.array(compute_average_surface_distance(pred, target, include_background)[0].cpu())
        assd[np.isinf(assd)] = np.nan
        return dice, hd, assd

    def predict(self, data):
        return sliding_window_inference(
            inputs=data, 
            roi_size=self.opt.crop_size, 
            sw_batch_size=self.opt.sw_batch_size, 
            predictor=self.model,
            overlap=self.opt.overlap,
            mode=self.opt.blend_mode,
            sigma_scale=self.opt.blend_sigma,
            padding_mode=self.opt.padding_mode,
            cval=self.opt.padding_val)

    def run(self):
        self.model.eval()
        self.test_dice, self.test_hd, self.test_assd = [], [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_ds)) as pbar:
                for i, data in enumerate(self.test_ds):
                    start = time()
                    path = data["subject"][0]
                    prefix = osp.basename(path)[:-7] 
                    pred = self.predict(data['image'].cuda())
                    pred = self.post_pred(pred)

                    # evaluation metrics
                    if self.opt.cal_metric:
                        target = self.post_label(data['label'].cuda())
                        dice, hd, assd = self.compute_metrics(pred, target, self.include_background)
                        self.test_dice.append(dice)
                        self.test_hd.append(hd)
                        self.test_assd.append(assd)

                    if self.opt.display_per_iter:
                        update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                            f"epoch={self.opt.epoch}, id={i+1}/{len(self.test_ds.dataset)}, "
                            f"subject={prefix}, time={time()-start:.4f}, "
                            f"dsc={list(map('{:.4f}'.format, dice))}, "
                            f"hd={list(map('{:.2f}'.format, hd))}, "
                            f"assd={list(map('{:.2f}'.format, assd))}"), self.infer_log)

                    if self.opt.save_output:
                        ref_image = sitk.ReadImage(path)
                        pred = torch.argmax(pred.squeeze(0), dim=0).detach().cpu().numpy().astype('uint8')
                        pred = np.transpose(pred, (2, 1, 0))
                        sitk_image = sitk.GetImageFromArray(pred)
                        sitk_image.CopyInformation(ref_image)
                        sitk.WriteImage(sitk_image, osp.join(self.result_dir, f'{prefix}_pred.nii.gz'))

                    pbar.update(1)

        if self.opt.cal_metric:
            np.savez(osp.join(self.opt.expr_dir, 'results.npz'),
                dice=self.test_dice,
                hd=self.test_hd,
                assd=self.test_assd)

            self.mean_dice = np.nanmean(self.test_dice, axis=0)  # report mean dice without nans
            self.mean_hd = np.nanmean(self.test_hd, axis=0)  
            self.mean_assd = np.nanmean(self.test_assd, axis=0)  
            self.std_dice = np.nanstd(self.test_dice, axis=0)  # report std dice without nans
            self.std_hd = np.nanstd(self.test_hd, axis=0)  
            self.std_assd = np.nanstd(self.test_assd, axis=0)  

            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                "Test foreground metrics: mean: (Dice|HD|ASSD): "
                f"{list(map('{:.4f}'.format, self.mean_dice))}, "
                f"{list(map('{:.2f}'.format, self.mean_hd))}, "
                f"{list(map('{:.2f}'.format, self.mean_assd))}"), self.infer_log)

            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                "Test foreground metrics: std: (Dice|HD|ASSD): "
                f"{list(map('{:.4f}'.format, self.std_dice))}, "
                f"{list(map('{:.2f}'.format, self.std_hd))}, "
                f"{list(map('{:.2f}'.format, self.std_assd))}"), self.infer_log)

            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                "Test mean foreground metrics(Dice|HD|ASSD): "
                f"{np.nanmean(self.mean_dice):.4f}, "
                f"{np.nanmean(self.mean_hd):.2f}, "
                f"{np.nanmean(self.mean_assd):.2f}"), self.infer_log)

            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                "Test std foreground metrics(Dice|HD|ASSD): "
                f"{np.nanmean(self.std_dice):.4f}, "
                f"{np.nanmean(self.std_hd):.2f}, "
                f"{np.nanmean(self.std_assd):.2f}"), self.infer_log)


import numpy as np
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from .net_utils import *


class Translator(nn.Module): 
    def __init__(
        self, 
        input_nc=1, 
        output_nc=1, 
        ngf=64, 
        use_dropout=False, 
        n_blocks=9, 
        style_dim=256,
        n_downsampling=2,
        padding_type='reflect', 
        no_antialias_up=False,
        use_bias=True):

        assert(n_blocks >= 0)
        super(Translator, self).__init__()

        # no learnable affine parameters for IN
        norm_layer = functools.partial(
            nn.InstanceNorm3d, 
            affine=False, 
            track_running_stats=False)

        # initial convolution
        encoder = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]

        # downsampling
        for i in range(n_downsampling): 
            mult = 2 ** i
            encoder += [
                nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
                Downsample(ngf * mult * 2)]
        mult = 2 ** n_downsampling

        # residual blocks x 9
        for i in range(n_blocks):
            encoder += [
                ResnetBlock(
                    ngf * mult, 
                    padding_type=padding_type, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout, 
                    use_bias=use_bias)
                ]

        self.encoder = nn.Sequential(*encoder)

        # decoders for synthesis and segmentation
        self.syn_head = DynSynHead(ngf, n_downsampling, output_nc)
        self.seg_head = SegHeadL(ngf, n_downsampling, output_nc, norm_layer)
        
        # style encoder trained by self-supervised contrastive learning
        self.style_encoder = StyleEncoder(
            n_downsample=2, 
            input_dim=input_nc, 
            dim=ngf, 
            style_dim=style_dim, 
            norm='inst', 
            activ='relu', 
            vae=False)

    def extract_style(self, input):
        return self.style_encoder(input)

    def forward(self, input, ref_style=None):
        if ref_style is None:
            """In this case, we assume the reference style is derived from a reference image,
            which is concatenated with the input image"""
            src_image = input[:, 0:1, ...]
            ref_image = input[:, 1:2, ...]
            ref_style = self.extract_style(ref_image) 
            bottleneck = self.encoder(src_image)
            output = self.syn_head(bottleneck, ref_style)
            return output
        else:
            """In this case, a reference style vector is explicitly provided"""
            bottleneck = self.encoder(input)
            output = self.syn_head(bottleneck, ref_style)
            return output





# #-------------

# def transform_A():
#     return Compose([
#         LoadImaged(keys=['A', 'A_msk']),
#         AddChanneld(keys=['A', 'A_msk']),
#         NormalizeForegroundd(keys=['A']),
#         ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         InvertT1d(keys=['A']),
#         CastToTyped(keys=['A'], dtype=np.float32),
#         ToTensord(keys=['A']),])


# def transform_B():
#     return Compose([
#         LoadImaged(keys=['B']),
#         AddChanneld(keys=['B']),
#         NormalizeForegroundd(keys=['B']),
#         ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
#         CastToTyped(keys=['B'], dtype=np.float32),
#         ToTensord(keys=['B']),])


# class NormalizeForegroundd(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         for k in self.keys:
#             img = data[k]
#             _mean = img[img!=0].mean()
#             _std = img[img!=0].mean()
#             data[k] = (data[k] - _mean)/_std
#         return data


# class InvertT1d(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         data['A'][data['A_msk']==3] = 1 
#         return data


# class BasicTransform(object):
#     def __init__(self):
#         self.translate = Compose([
#             LoadImaged(keys=['A', 'A_msk', 'B']), 
#             AddChanneld(keys=['A', 'A_msk', 'B']),
#             NormalizeForegroundd(keys=['A', 'B']),
#             ScaleIntensityRangePercentilesd(
#                 keys=['A', 'B'], 
#                 lower=0, 
#                 upper=99.9, 
#                 b_min=-1, 
#                 b_max=1, 
#                 clip=True, 
#                 relative=False),
#             InvertT1d(keys=['A']),
#             CastToTyped(
#                 keys=['A', 'A_msk', 'B'], 
#                 dtype=[np.float32, np.uint8, np.float32]),
#             ToTensord(keys=['A', 'A_msk', 'B'])
#         ])


# if __name__ == "__main__":

#     path = '/data/crossmoda2023/segmentation/src/models/pretrained.pth'
#     model = Translator().cuda()
#     model.load_state_dict(torch.load(path))

#     path_A = '/data/crossmoda2023/data/crossmoda23_training/srcImageROI/crossmoda2023_ukm_20_ceT1.nii.gz'
#     path_A_msk = path_A.replace('ceT1', 'Label').replace('srcImageROI', 'srcLabelROI')
#     ref_path = '/data/crossmoda2023/data/crossmoda23_training/tgtImageROI/crossmoda2023_etz_107_T2.nii.gz'
    
#     # data_A = {'A': path_A, 'A_msk': path_A_msk}
#     # data_B = {'B': ref_path}

#     # data_A = transform_A()(data_A)
#     # data_B = transform_B()(data_B)

#     data = {'A': path_A, 'A_msk': path_A_msk, 'B': ref_path}
#     data_A = BasicTransform().translate(data)
#     with torch.no_grad():
#         data_A['A'] = sliding_window_inference(
#             inputs=torch.cat((data_A['A'], data_A['B']), 0).unsqueeze(0).cuda(), 
#             roi_size=(256, 144, 12), 
#             sw_batch_size=1, 
#             predictor=model,
#             overlap=0.9,
#             mode='gaussian',
#             sigma_scale=0.125,
#             padding_mode='constant',
#             cval=-1,
#             )[0]
#         # data_A['A'] = model(torch.cat((data_A['A'], data_A['B']), 0).unsqueeze(0).cuda())
#     # breakpoint()
#     Compose([
#         # SqueezeDimd(keys=['A'], dim=0),
#         SqueezeDimd(keys=['A'], dim=0),
#         SaveImaged(
#             keys=['A'], 
#             output_dir=f"/data/crossmoda2023/segmentation/checkpoints/temp/web/images", 
#             output_postfix=osp.basename(ref_path)[-17:-7], 
#             output_ext='.nii.gz', 
#             resample=False,
#             separate_folder=False,
#             print_log=False)])(data_A)

#     # data = torch.ones((1, 2, 256, 144, 12)).cuda()
#     # output = model(data)
#     # breakpoint()


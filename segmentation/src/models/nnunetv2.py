from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op


def load_nnunetv2():
    conv_kernel_sizes = [[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    conv_or_blocks_per_stage = {
        'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
        'n_conv_per_stage_decoder': [2, 2, 2, 2, 2]}
    conv_op = convert_dim_to_conv_op(3)

    kwargs = {
        'conv_bias': True,
        'norm_op': get_matching_instancenorm(conv_op),
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True}}

    model = PlainConvUNet(
        input_channels=1,
        n_stages=len(conv_kernel_sizes),
        features_per_stage=[min(32 * 2 ** i, 320) for i in range(len(conv_kernel_sizes))],
        conv_op=conv_op,
        kernel_sizes=conv_kernel_sizes,
        strides=pool_op_kernel_sizes,
        num_classes=4,
        deep_supervision=False,
        **conv_or_blocks_per_stage,
        **kwargs
    )

    return model



import numpy as np
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt[:, None] * filt[None, :] 
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad3d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad3d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad3d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride, ::self.stride]
        else:
            return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose3d(
            self.pad(inp), 
            self.filt, 
            stride=self.stride, 
            padding=1 + self.pad_size, 
            groups=inp.shape[1])[:, :, 1:, 1:, 1:]

        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1, :-1]


class DynSynHead(nn.Module):
    def __init__(self, ngf, n_downsampling, output_nc):
        super(DynSynHead, self).__init__()
        syn_head = []
        for i in range(n_downsampling): 
            mult = 2 ** (n_downsampling - i)
            syn_head += [
                Upsample(ngf * mult),
                nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                        kernel_size=3, stride=1, padding=1, bias=True),
                DynInsNorm(int(ngf * mult / 2), code_length=256),
                nn.ReLU(True)]

        syn_head += [nn.ReflectionPad3d(3)]
        syn_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        syn_head += [nn.Tanh()]
        self.syn_head = nn.Sequential(*syn_head)

    def forward(self, feat, code):
        for layer_id, layer in enumerate(self.syn_head):
            if layer_id in [2, 6]: # yes, we hard-code here
                feat = layer(feat, code)
            else:
                feat = layer(feat)
        return feat


# class DynInsNorm(nn.Module):
#     def __init__(self, num_features, code_length=3):
#         super(DynInsNorm, self).__init__()
#         self.num_features = num_features
#         self.controller = nn.Conv3d(in_channels=code_length, out_channels=num_features*2, 
#             kernel_size=1, stride=1, padding=0, bias=True)  

#     def generate_dyn_param(self, code):
#         params = self.controller(code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())  
#         params.squeeze_(-1).squeeze_(-1).squeeze_(-1)  
#         return params[:, :self.num_features], params[:, self.num_features:]

#     def forward(self, x, code):
#         weight, bias = self.generate_dyn_param(code)
#         weight = weight.reshape(-1)
#         bias = bias.reshape(-1)
#         output = F.instance_norm(input=x, weight=weight, bias=bias)
#         return output


class DynInsNorm(nn.Module):
    def __init__(self, num_features, code_length=3):
        super(DynInsNorm, self).__init__()
        self.num_features = num_features
        self.controller = nn.Conv3d(in_channels=code_length, out_channels=num_features*2, 
            kernel_size=1, stride=1, padding=0, bias=True)  

    def generate_dyn_param(self, code):
        params = self.controller(code.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float())  
        params.squeeze_(-1).squeeze_(-1).squeeze_(-1)  
        return params[:, :self.num_features], params[:, self.num_features:]

    def forward(self, x, code):
        weight, bias = self.generate_dyn_param(code)
        outputs = []
        for i in range(x.shape[0]):
            outputs.append(F.instance_norm(input=x[i:i+1, ...], weight=weight[i], bias=bias[i]))
        output = torch.cat(outputs, dim=0)
        return output


class SegHeadL(nn.Module):
    def __init__(self, ngf, n_downsampling, output_nc, norm_layer):
        super(SegHeadL, self).__init__()
        seg_head = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            seg_head += [
                Upsample(ngf * mult), 
                nn.Conv3d(
                    ngf * mult, int(ngf * mult / 2),
                    kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        seg_head += [nn.Conv3d(ngf, 4, kernel_size=3, padding=1)]
        self.seg_head = nn.Sequential(*seg_head) 

    def forward(self, x):
        return self.seg_head(x)


class Conv3dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv3dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad3d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad3d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm3d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm3d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [
            Conv3dBlock(input_dim=input_dim, output_dim=dim, kernel_size=7, stride=1, 
                padding=3, norm=norm, activation=activ, pad_type='reflect')]

        for i in range(2):
            self.model += [
                Conv3dBlock(input_dim=dim, output_dim=2 * dim, kernel_size=4, stride=2, 
                    padding=1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2

        for i in range(n_downsample - 2):
            self.model += [
                Conv3dBlock(input_dim=dim, output_dim=dim, kernel_size=4, stride=2, 
                    padding=1, norm=norm, activation=activ, pad_type='reflect')]

        self.model += [nn.AdaptiveAvgPool3d(1)]  # global average pooling

        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)  
            self.fc_var = nn.Linear(dim, style_dim)
        else:
            self.model += [nn.Conv3d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            x = self.model(x).view(x.size(0), -1)
            x = F.normalize(x, p=2, dim=1)
            # x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # patch-based inference
            return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import random


###############################################################################
# Helper Functions
###############################################################################


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
    filt = filt[:, None] * filt[None, :]  # 2D -> 3D
    filt = filt / torch.sum(filt)
    return filt


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
        # self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1))) # 2D -> 3D
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride, ::self.stride]
        else:
            return F.conv3d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


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
        # self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.register_buffer('filt', filt[None, None, :, :, :].repeat((self.channels, 1, 1, 1, 1))) # 2D -> 3D

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose3d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1, :-1]


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


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_9blocks_dyn':
        net = ResnetGeneratorDyn(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_9blocks_post':
        net = ResnetGeneratorPost(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_9blocks_style':
        net = ResnetGeneratorStyle(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_9blocks_bsl':
        net = ResnetGeneratorBSL(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'sid':
        net = SIDGenerator(input_nc, output_nc, ngf, style_dim=256)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'resnet_cat':
        n_blocks = 8
        net = G_Resnet(input_nc, output_nc, opt.nz, num_downs=2, n_res=n_blocks - 4, ngf=ngf, norm='inst', nl_layer='relu')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias,)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == "patch":
        net = PatchDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool3d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.model(x))


class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool3d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)


class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        # self.conv1 = nn.Conv3d(256, 128, 3, stride=2)
        # self.conv2 = nn.Conv3d(128, 64, 3, stride=1)
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv3d(C, max(C // 2, 64), 3, stride=2))
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv3d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def forward(self, x, use_instance_norm=False):
        C, H = x.shape[1], x.shape[2]
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = F.instance_norm(x)
        return self.l2_norm(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, attn_mats=None):
        return_ids = []
        return_feats = []
        return_mats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, C, H, W, D = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4]
            feat_reshape = feat.permute(0, 2, 3, 4, 1).flatten(1, 3) # B*HWD*C
            if num_patches > 0:
                if feat_id < 3:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)  # random id in [0, HWD]
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                    x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
                    attn_qs = torch.zeros(1).to(feat.device)
                else:
                    if attn_mats is not None:
                        attn_qs = attn_mats[feat_id]
                    else:
                        feat_q = feat_reshape
                        feat_k = feat_reshape.permute(0, 2, 1)
                        dots = torch.bmm(feat_q, feat_k)  # (B, HWD, HWD)
                        attn = dots.softmax(dim=2)
                        prob = -torch.log(attn)
                        prob = torch.where(torch.isinf(prob), torch.full_like(prob, 0), prob)
                        entropy = torch.sum(torch.mul(attn, prob), dim=2)
                        _, index = torch.sort(entropy)
                        patch_id = index[:, :num_patches]
                        attn_qs = attn[torch.arange(B)[:, None], patch_id, :]
                    feat_reshape = torch.bmm(attn_qs, feat_reshape)  # (B, n_p, C)
                    x_sample = feat_reshape.flatten(0, 1)
                    patch_id = []
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            return_mats.append(attn_qs)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W, D])
            return_feats.append(x_sample)
        return return_feats, return_ids, return_mats

class G_Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
                 norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def forward(self, image, style=None, nce_layers=[], encode_only=False):
        content, feats = self.enc_content(image, nce_layers=nce_layers, encode_only=encode_only)
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon


##################################################################################
# Encoder and Decoders
##################################################################################


class E_adaIN(nn.Module):
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
                 norm=None, nl_layer=None, vae=False):
        # style encoder
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)

    def forward(self, image):
        style = self.enc_style(image)
        return style


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        self.model += [Conv3dBlock(
            input_dim=input_dim, 
            output_dim=dim, 
            kernel_size=7, 
            stride=1, 
            padding=3, 
            norm=norm, 
            activation=activ, 
            pad_type='reflect')]

        for i in range(2):
            self.model += [Conv3dBlock(
                input_dim=dim, 
                output_dim=2 * dim, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                norm=norm, 
                activation=activ, 
                pad_type='reflect')]
            dim *= 2

        for i in range(n_downsample - 2):
            self.model += [Conv3dBlock(
                input_dim=dim, 
                output_dim=dim, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                norm=norm, 
                activation=activ, 
                pad_type='reflect')]

        self.model += [nn.AdaptiveAvgPool3d(1)]  # global average pooling

        if self.vae:
            self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
            self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
        else:
            self.model += [nn.Conv3d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        # print(x.shape)
        if self.vae:
            output = self.model(x)
            output = output.view(x.size(0), -1)
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            # return self.model(x).view(x.size(0), -1)
            x = self.model(x).view(x.size(0), -1)
            x = F.normalize(x, p=2, dim=1)
            # print(x[:, :5])
            # x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # patch-based inference
            # print(x.shape)
            return x



# class StyleEncoder(nn.Module):
#     def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=True):
#         super(StyleEncoder, self).__init__()
#         self.vae = vae
#         self.model = []
#         self.model += [Conv3dBlock(
#             input_dim=input_dim, 
#             output_dim=dim, 
#             kernel_size=7, 
#             stride=1, 
#             padding=3, 
#             norm=norm, 
#             activation=activ, 
#             pad_type='reflect')]

#         for i in range(2):
#             self.model += [Conv3dBlock(
#                 input_dim=dim, 
#                 output_dim=2 * dim, 
#                 kernel_size=4, 
#                 stride=2, 
#                 padding=1, 
#                 norm=norm, 
#                 activation=activ, 
#                 pad_type='reflect')]
#             dim *= 2

#         for i in range(n_downsample - 2):
#             self.model += [Conv3dBlock(
#                 input_dim=dim, 
#                 output_dim=dim, 
#                 kernel_size=4, 
#                 stride=2, 
#                 padding=1, 
#                 norm=norm, 
#                 activation=activ, 
#                 pad_type='reflect')]

#         self.model += [nn.AdaptiveAvgPool3d(1)]  
#         self.model = nn.Sequential(*self.model)
#         self.fc_mean = nn.Linear(dim, style_dim) 
#         self.fc_var = nn.Linear(dim, style_dim)   
#         self.reparameterize = self._reparameterize
#         self.output_dim = dim

#     def _reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         z = mu + eps*std
#         return z

#     def forward(self, x):
#         output = self.model(x)
#         output = output.view(x.size(0), -1)
#         mu = self.fc_mean(output)
#         logvar = self.fc_var(output)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv3dBlock(
            input_dim=input_dim, 
            output_dim=dim, 
            kernel_size=7, 
            stride=1, 
            padding=3, 
            norm=norm, 
            activation=activ, 
            pad_type='reflect')]

        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv3dBlock(
                input_dim=dim, 
                output_dim=2 * dim, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                norm=norm, 
                activation=activ, 
                pad_type='reflect')]
            dim *= 2
        
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

    # def forward(self, x, nce_layers=[], encode_only=False):
    #     if len(nce_layers) > 0:
    #         feat = x
    #         feats = []
    #         for layer_id, layer in enumerate(self.model):
    #             feat = layer(feat)
    #             if layer_id in nce_layers:
    #                 feats.append(feat)
    #             if layer_id == nce_layers[-1] and encode_only:
    #                 return None, feats
    #         return feat, feats
    #     else:
    #         return self.model(x), None



### Single image disentanglement ### ----------------------------------------------------------------------------------------------------
class SIDGenerator(nn.Module):
    def __init__(
        self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        style_dim=256):

        super(SIDGenerator, self).__init__()
        self.style_encoder = StyleEncoder(
            n_downsample=2, input_dim=input_nc, dim=ngf, style_dim=style_dim, norm='none', activ='relu', vae=False)
        """no normalization layer used in style encoder to preserve style"""
        self.content_encoder = ContentEncoder(
            n_downsample=2, n_res=9, input_dim=input_nc, dim=ngf, norm='inst', activ='relu')
        self.decoder = CSDD(ngf, n_downsampling=2, output_nc=output_nc)

    def forward(self, input):
        style = self.style_encoder(input)
        content = self.content_encoder(input)
        output = self.decoder(content, style)
        return output


class CSDD(nn.Module):
    """ Content style disentanglement decoder"""
    def __init__(self, ngf, n_downsampling, output_nc):
        super(CSDD, self).__init__()
        syn_head = []
        for i in range(n_downsampling): 
            mult = 2 ** (n_downsampling - i)
            if i == n_downsampling - 1:
                syn_head += [
                    Upsample(ngf * mult),
                    nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=3, stride=1, padding=1, bias=True),
                    DynInsNorm(int(ngf * mult / 2), code_length=256),
                    nn.ReLU(True)]
            else:
                syn_head += [
                    Upsample(ngf * mult),
                    nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=3, stride=1, padding=1, bias=True),
                    nn.InstanceNorm3d(int(ngf * mult / 2)),
                    nn.ReLU(True)]

        syn_head += [nn.ReflectionPad3d(3)]
        syn_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        syn_head += [nn.Tanh()]
        self.syn_head = nn.Sequential(*syn_head)
        self.dyn_layer = self.syn_head[6]

    def forward(self, content, code):
        for layer_id, layer in enumerate(self.syn_head):
            if layer_id in [6]:
                feat = layer(feat, code)
            else:
                feat = layer(feat)
        return feat


class Decoder_all(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        # AdaIN residual blocks
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # upsampling blocks
        for i in range(n_upsample):
            block = [Upsample2(scale_factor=2), Conv3dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2
        # use reflection padding in the last conv layer
        setattr(self, 'block_{:d}'.format(self.n_blocks), Conv3dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1

    def forward(self, x, y=None):
        if y is not None:
            output = self.resnet_block(cat_feature(x, y))
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n))
                if n > 0:
                    output = block(cat_feature(output, y))
                else:
                    output = block(output)
            return output


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            if i == 0:
                input_dim = dim + nz
            else:
                input_dim = dim
            self.model += [Upsample2(scale_factor=2), Conv3dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv3dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x, y=None):
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)

##################################################################################
# Sequential Models
##################################################################################


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
def cat_feature(x, y):
    y_expand = y.view(y.size(0), y.size(1), 1, 1).expand(
        y.size(0), y.size(1), x.size(2), x.size(3))
    x_cat = torch.cat([x, y_expand], 1)
    return x_cat


class ResBlock(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv3dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv3dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv3dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv3dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad3d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad3d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
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

        # initialize activation
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

        # initialize convolution
        self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
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

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        encoder = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]
            else:
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        # bottleneck
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder = nn.Sequential(*encoder)

        # create synthetic head
        syn_head = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                syn_head += [
                    nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                        kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
            else:
                syn_head += [
                    Upsample(ngf * mult),
                    nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        syn_head += [nn.ReflectionPad3d(3)]
        syn_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        syn_head += [nn.Tanh()]
        self.syn_head = nn.Sequential(*syn_head)
        
        # Add a segmentation head. The same decoder structure is used as for synthesis.
        # seg_head = []
        # for i in range(n_downsampling):
        #     mult = 2 ** (n_downsampling - i)
        #     seg_head += [
        #         Upsample(ngf * mult), 
        #         nn.Conv3d(
        #             ngf * mult, int(ngf * mult / 2),
        #             kernel_size=3, stride=1, padding=1, bias=use_bias),
        #         norm_layer(int(ngf * mult / 2)),
        #         nn.ReLU(True)]
        # seg_head += [nn.ReflectionPad3d(3)]
        # seg_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        # seg_head += [nn.Sigmoid()]
        # self.seg_head = nn.Sequential(*seg_head) 

        self.seg_head = SegHeadL(ngf, n_downsampling, output_nc, norm_layer)

    def forward(self, input, layers=[], encode_only=False):

        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.encoder):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features

        else:
            """Standard forward"""
            bottleneck = self.encoder(input)
            syn = self.syn_head(bottleneck)
            seg = self.seg_head(bottleneck)
            return syn #, seg


class ResnetGeneratorDyn(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        assert(n_blocks >= 0)
        super(ResnetGeneratorDyn, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        encoder = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]
            else:
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        # bottleneck
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder = nn.Sequential(*encoder)
        self.syn_head = DynSynHead(ngf, n_downsampling, output_nc)
        self.seg_head = SegHeadL(ngf, n_downsampling, output_nc, norm_layer)

    def forward(self, input, layers=[], encode_only=False, code=None):
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.encoder):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features

        else:
            """Standard forward"""
            bottleneck = self.encoder(input)
            syn = self.syn_head(bottleneck, code)
            seg = self.seg_head(bottleneck)
            return syn, seg


class ResnetGeneratorStyle(nn.Module): # --------------------------------------- STYLE QS-ATT-----------------------------------
    def __init__(
        self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        norm_layer=nn.BatchNorm3d, 
        use_dropout=False, 
        n_blocks=6, 
        padding_type='reflect', 
        no_antialias=False, 
        no_antialias_up=False, 
        opt=None):

        assert(n_blocks >= 0)
        super(ResnetGeneratorStyle, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        style_dim = 256

        encoder = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling): 
            mult = 2 ** i
            if(no_antialias):
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]
            else:
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        # bottleneck
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder = nn.Sequential(*encoder)
        self.syn_head = DynSynHead(ngf, n_downsampling, output_nc)
        self.seg_head = SegHeadL(ngf, n_downsampling, output_nc, norm_layer)
        self.style_encoder = StyleEncoder(n_downsample=2, input_dim=input_nc, dim=ngf, style_dim=style_dim, norm='inst', activ='relu', vae=False)

    def extract_style(self, input):
        return self.style_encoder(input)

    def forward(self, input, layers=[], encode_only=False, condition=None, is_train=True, ref_style=None):
        if is_train:
            if len(layers) > 0:
                feat = input
                feats = []
                for layer_id, layer in enumerate(self.encoder):
                    feat = layer(feat)
                    if layer_id in layers:
                        feats.append(feat)
                    else:
                        pass
                    if layer_id == layers[-1] and encode_only:
                        return feats
                return feat, feats 
            else:
                ref_style = self.extract_style(condition)
                bottleneck = self.encoder(input)
                syn = self.syn_head(bottleneck, ref_style)
                seg = self.seg_head(bottleneck)
                return syn, seg
        else:
            if len(layers) > 0:
                feat = input
                feats = []
                for layer_id, layer in enumerate(self.encoder):
                    # print(layer_id, layer)
                    feat = layer(feat)
                    if layer_id in layers:
                        # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                        feats.append(feat)
                    else:
                        # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                        pass
                    if layer_id == layers[-1] and encode_only:
                        # print('encoder only return features')
                        return feats  # return intermediate features alone; stop in the last layers

                return feat, feats  # return both output and intermediate features

            else:
                if ref_style is None:
                    """Standard forward"""
                    ref_style = self.extract_style(input[:, 1, ...].unsqueeze(0)) 
                    # print(ref_style.view(-1)[:6])
                    bottleneck = self.encoder(input[:, 0, ...].unsqueeze(0))
                    syn = self.syn_head(bottleneck, ref_style)
                    seg = self.seg_head(bottleneck)
                    return syn, seg
                else:
                    # generate novel styles here
                    # print(ref_style.view(-1)[:6])
                    bottleneck = self.encoder(input)
                    syn = self.syn_head(bottleneck, ref_style)
                    seg = self.seg_head(bottleneck)
                    return syn, seg



#------------------baseline ------------------------
class ResnetGeneratorBSL(nn.Module): 
    def __init__(
        self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        norm_layer=nn.BatchNorm3d, 
        use_dropout=False, 
        n_blocks=6, 
        padding_type='reflect', 
        no_antialias=False, 
        no_antialias_up=False, 
        opt=None):

        assert(n_blocks >= 0)
        super(ResnetGeneratorBSL, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        style_dim = 256

        encoder = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling): 
            mult = 2 ** i
            if(no_antialias):
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]
            else:
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        # bottleneck
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder = nn.Sequential(*encoder)
        self.syn_head = SynHead(ngf, n_downsampling, output_nc, norm_layer)
        self.seg_head = SegHeadL(ngf, n_downsampling, output_nc, norm_layer)

    def forward(self, input, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.encoder):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features

        else:
            """Standard forward"""
            bottleneck = self.encoder(input)
            syn = self.syn_head(bottleneck)
            seg = self.seg_head(bottleneck)
            return syn, seg


class ResnetGeneratorPost(nn.Module):

    def __init__(
        self, 
        input_nc, 
        output_nc, 
        ngf=64, 
        norm_layer=nn.BatchNorm3d, 
        use_dropout=False, 
        n_blocks=6, 
        padding_type='reflect', 
        no_antialias=False, 
        no_antialias_up=False, 
        opt=None):

        assert(n_blocks >= 0)
        super(ResnetGeneratorPost, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        encoder = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]
            else:
                encoder += [
                    nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling

        # bottleneck
        for i in range(n_blocks):
            encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.encoder = nn.Sequential(*encoder)
        self.syn_head = DynSynHead(ngf, n_downsampling, output_nc)
        self.seg_head = SegHeadL(ngf, n_downsampling, output_nc, norm_layer)

    # def forward(self, input, layers=[], encode_only=False, condition=None):
    #     if len(layers) > 0:
    #         feat = input
    #         feats = []
    #         for layer_id, layer in enumerate(self.encoder):
    #             # print(layer_id, layer)
    #             feat = layer(feat)
    #             if layer_id in layers:
    #                 # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
    #                 feats.append(feat)
    #             else:
    #                 # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
    #                 pass
    #             if layer_id == layers[-1] and encode_only:
    #                 # print('encoder only return features')
    #                 return feats  # return intermediate features alone; stop in the last layers

    #         return feat, feats  # return both output and intermediate features

    #     else:
    #         """Standard forward"""
    #         ref_style = self.encoder(condition)  # 1 x 256 x 64 x 36 x 2
    #         bottleneck = self.encoder(input)
    #         ref_style = nn.AdaptiveAvgPool3d(1)(ref_style)
    #         ref_style = ref_style.view(-1, ref_style.size(1))  # 1 x 256
    #         syn = self.syn_head(bottleneck, ref_style)
    #         seg = self.seg_head(bottleneck)
    #         return syn, seg


    # inference time
    def forward(self, input, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.encoder):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features

        else:
            """Standard forward"""
            # breakpoint()
            ref_style = self.encoder(input[:, 1, ...].unsqueeze(0))  # 1 x 256 x 64 x 36 x 2
            bottleneck = self.encoder(input[:, 0, ...].unsqueeze(0))
            ref_style = nn.AdaptiveAvgPool3d(1)(ref_style)
            ref_style = ref_style.view(-1, ref_style.size(1))  # 1 x 256
            syn = self.syn_head(bottleneck, ref_style)
            seg = self.seg_head(bottleneck)
            return syn, seg


######################################################


class ResnetDecoder(nn.Module):
    """Resnet-based decoder that consists of a few Resnet blocks + a few upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based decoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if(no_antialias):
                model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        model += [nn.ReflectionPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetEncoder(nn.Module):
    """Resnet-based encoder that consists of a few downsampling + several Resnet blocks
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
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
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = (4, 4, 3)  # update by Han, was 4. We update this because we have smaller z-axis
        padw = 1

        if(no_antialias):
            sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),  # H = H-2
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class PatchDiscriminator(NLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().forward(input)


class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)


class SegHead(nn.Module):
    def __init__(self, ngf, n_downsampling, output_nc, norm_layer):
        super(SegHead, self).__init__()
        seg_head = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            seg_head += [
                Upsample(ngf * mult), 
                nn.Conv3d(
                    ngf * mult, int(ngf * mult / 2),
                    kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]
        seg_head += [nn.ReflectionPad3d(3)]
        seg_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        seg_head += [nn.Sigmoid()]
        self.seg_head = nn.Sequential(*seg_head) 

    def forward(self, x):
        return self.seg_head(x)


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

        # two more blocks for style transfer
        # for i in range(2):
        #     seg_head += [
        #         nn.Conv3d(int(ngf/(2**i)), int(ngf/(2**(i+1))), kernel_size=3, stride=1, padding=1, bias=True),
        #         norm_layer(ngf),
        #         nn.ReLU(True)]

        # output channels = 4: background, VS1, VS2, cochlea
        seg_head += [nn.Conv3d(ngf, 4, kernel_size=3, padding=1)]
        self.seg_head = nn.Sequential(*seg_head) 

    def forward(self, x):
        return self.seg_head(x)


# 3D implementation
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
        weight = weight.reshape(-1)
        bias = bias.reshape(-1)
        output = F.instance_norm(input=x, weight=weight, bias=bias)
        # return output
        return output


class DynSynHead(nn.Module):
    """ two dynamic instance normalziation layers """
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
            # print(layer_id, layer)
            if layer_id in [2, 6]:
                feat = layer(feat, code)
            else:
                feat = layer(feat)
        # breakpoint()
        return feat


class SynHead(nn.Module):
    """ two dynamic instance normalziation layers """
    def __init__(self, ngf, n_downsampling, output_nc, norm_layer):
        super(SynHead, self).__init__()
        syn_head = []
        for i in range(n_downsampling): 
            mult = 2 ** (n_downsampling - i)
            syn_head += [
                Upsample(ngf * mult),
                nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                        kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)]

        syn_head += [nn.ReflectionPad3d(3)]
        syn_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        syn_head += [nn.Tanh()]
        self.syn_head = nn.Sequential(*syn_head)

    def forward(self, feat):
        return self.syn_head(feat)



# class DynSynHead(nn.Module):
#     def __init__(self, ngf, n_downsampling, output_nc):
#         super(DynSynHead, self).__init__()
#         syn_head = []
#         for i in range(n_downsampling): 
#             mult = 2 ** (n_downsampling - i)
#             if i == n_downsampling - 1:
#                 syn_head += [
#                     Upsample(ngf * mult),
#                     nn.Conv3d(ngf * mult, int(ngf * mult / 2),
#                             kernel_size=3, stride=1, padding=1, bias=True),
#                     DynInsNorm(int(ngf * mult / 2), code_length=256),
#                     nn.ReLU(True)]
#             else:
#                 syn_head += [
#                     Upsample(ngf * mult),
#                     nn.Conv3d(ngf * mult, int(ngf * mult / 2),
#                             kernel_size=3, stride=1, padding=1, bias=True),
#                     nn.InstanceNorm3d(int(ngf * mult / 2)),
#                     nn.ReLU(True)]

#         syn_head += [nn.ReflectionPad3d(3)]
#         syn_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
#         syn_head += [nn.Tanh()]
#         self.syn_head = nn.Sequential(*syn_head)
#         # self.dyn_layer = self.syn_head[6]

#     def forward(self, feat, code):
#         for layer_id, layer in enumerate(self.syn_head):
#             if layer_id in [6]:
#                 feat = layer(feat, code)
#             else:
#                 feat = layer(feat)
#         return feat


class CLModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.01)))

    def forward(self, feat, pos, neg):
        def _shuffle(feats, labels):
            combined = list(zip(feats, labels))
            random.shuffle(combined)
            feats_shuffled, labels_shuffled = zip(*combined)
            return torch.stack(feats_shuffled), torch.tensor(labels_shuffled)

        pairs = torch.cat([pos, neg], 0)
        labels = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))], 0)
        pairs, labels = _shuffle(pairs, labels)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(feat, pairs.T)
        label = torch.where(labels == 1)[0].cuda()
        return logits, label


def sobelLayer(input):
    pad = nn.ConstantPad3d((1,1,1,1,1,1),-1)
    kernel = create3DsobelFilter()
    act = nn.Tanh()
    paded = pad(input)
    fake_sobel = F.conv3d(paded, kernel, padding = 0, groups = 1)/4
    n,c,h,w,l = fake_sobel.size()
    fake = torch.norm(fake_sobel,2,1,True)/c*3
    fake_out = act(fake)*2-1
    return fake_out


def create3DsobelFilter():
    num_1, num_2, num_3 = np.zeros((3,3))
    num_1 = [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
    num_2 = [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    num_3 = [[-1., -2., -1.],
             [-2., -4., -2.],
             [-1., -2., -1.]]
    sobelFilter = np.zeros((3,1,3,3,3))
    sobelFilter[0,0,0,:,:] = num_1
    sobelFilter[0,0,1,:,:] = num_2
    sobelFilter[0,0,2,:,:] = num_3
    sobelFilter[1,0,:,0,:] = num_1
    sobelFilter[1,0,:,1,:] = num_2
    sobelFilter[1,0,:,2,:] = num_3
    sobelFilter[2,0,:,:,0] = num_1
    sobelFilter[2,0,:,:,1] = num_2
    sobelFilter[2,0,:,:,2] = num_3
    return torch.from_numpy(sobelFilter).type(torch.cuda.FloatTensor)



# class DynSynHead(nn.Module):
#     def __init__(self, ngf, n_downsampling, output_nc):
#         super(DynSynHead, self).__init__()
#         syn_head = []
#         for i in range(n_downsampling): 
#             mult = 2 ** (n_downsampling - i)
#             syn_head += [
#                 Upsample(ngf * mult),
#                 nn.Conv3d(ngf * mult, int(ngf * mult / 2),
#                         kernel_size=3, stride=1, padding=1, bias=True),
#                 DynInsNorm(int(ngf * mult / 2)),
#                 nn.ReLU(True)]

#         # two more blocks for style transfer
#         for i in range(2):
#             syn_head += [
#                 nn.Conv3d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=True),
#                 DynInsNorm(int(ngf * mult / 2)),
#                 nn.ReLU(True)]

#         syn_head += [nn.ReflectionPad3d(3)]
#         syn_head += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
#         syn_head += [nn.Tanh()]
#         self.syn_head = nn.Sequential(*syn_head)

#     def forward(self, feat, code):
#         for layer_id, layer in enumerate(self.syn_head):
#             if layer_id in [2, 6, 9, 12]:
#                 feat = layer(feat, code)
#             else:
#                 feat = layer(feat)
#         return feat


# class DynInsNorm(nn.Module):
#     def __init__(self, num_features, code_length=3):
#         super(DynInsNorm, self).__init__()
#         self.num_features = num_features
#         self.controller = nn.Conv3d(in_channels=code_length, out_channels=num_features*2, 
#             kernel_size=1, stride=1, padding=0, bias=True)  

#     def generate_dyn_param(self, code):
#         params = self.controller(code.unsqueeze(-1).unsqueeze(-1).float())  
#         params.squeeze_(-1).squeeze_(-1).squeeze_(0)  
#         return params[:self.num_features], params[self.num_features:]

#     def forward(self, x, code):
#         weight, bias = self.generate_dyn_param(code)
#         output = F.instance_norm(input=x, weight=weight, bias=bias)
#         return output


# code = torch.ones((1,3)).cuda()
# # x = torch.rand((2, 256, 80, 80))
# x = torch.rand((2, 1, 256, 128, 32)).cuda()
# # my = DynSynHead(64, 2, 1)

# my = ResnetGeneratorDyn(
#     input_nc=1, 
#     output_nc=1, 
#     ngf=64, 
#     norm_layer=nn.InstanceNorm3d, 
#     use_dropout=False, 
#     n_blocks=9, 
#     padding_type='reflect', 
#     no_antialias=False, 
#     no_antialias_up=False, 
#     opt=None).cuda()

# out = my(x, code=code)

# import pdb
# pdb.set_trace()



# weight = torch.tensor((1,2,3,4,5)).float()
# bias = torch.ones((5)) * 0
# output1 = torch.nn.functional.instance_norm(input=x, weight=weight, bias=bias, use_input_stats=True, momentum=0.1, eps=1e-05)

# my_layer = torch.nn.InstanceNorm3d(num_features=24, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None)

# output2 = my_layer(x)

# print(output1[0,0,...])
# print(output2[0,0,...])

# import pdb
# pdb.set_trace()

# print(torch.allclose(output1, output2))

# network = ResnetGenerator(
#     input_nc=1, 
#     output_nc=1, 
#     ngf=64, 
#     norm_layer=nn.InstanceNorm3d, 
#     use_dropout=False, 
#     n_blocks=9, 
#     padding_type='reflect', 
#     no_antialias=False, 
#     no_antialias_up=False, 
#     opt=None).cuda()

# data = torch.ones((1, 1, 320, 320)).cuda()

# # output = network(data, [0,4,8,12,16], True)
# syn, seg = network(data)
# print(syn.shape, seg.shape)
# print(torch.max(syn), torch.min(syn), torch.max(seg), torch.min(seg))

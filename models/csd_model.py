import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks_global_seg_3D
import util.util as util


class CSDModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_recon', type=float, default=1.0, help='weight for reconstruction loss')
        parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
        parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        assert opt.netG in ['sid']
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        networks = networks_global_seg_3D
        self.loss_names = ['Recon', 'style', 'content']
        self.visual_names = ['recon']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else: 
            self.model_names = ['G']
        
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self):
        bs_per_gpu = self.real_A.size(0) // len(self.opt.gpu_ids)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                  
        if self.opt.isTrain:
            self.backward_D()           
            self.backward_G()  

    def optimize_parameters(self):
        # forward
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def set_input(self, input):
        self.images = input['B'].to(self.device)

    def forward(self):
        bs = self.images.size()[0]
        content_images = self.images[:(bs//2), ...]  # same styles
        style_images = self.images[(bs//2):, ...]    # same contents
        self.recon = self.netG(self.images)
        self.style_feats = self.netG.style_encoder(content_images)   # should be consistent
        self.content_feats = self.netG.content_encoder(style_images) # should be consistent

    def backward_D(self):
        if self.opt.lambda_GAN > 0.0:
            fake = self.recon.detach()
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            pred_real = self.netD(self.images)
            loss_D_real_unweighted = self.criterionGAN(pred_real, True)
            self.loss_D_real = loss_D_real_unweighted.mean()
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        else:
            self.loss_D_real, self.loss_D_fake, self.loss_D = 0.0, 0.0, 0.0

    def backward_G(self):
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(self.recon)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_recon = self.calculate_recon_loss(self.images, self.recon) * self.opt.lambda_recon
        self.loss_style = self.calculate_style_loss(self.style_feats) * self.opt.lambda_style
        self.loss_content = self.calculate_content_loss(self.content_feats) * self.opt.lambda_content
        self.loss_G = self.loss_G_GAN + self.loss_recon + self.loss_style + self.loss_content
        self.loss_G.backward()

    def calculate_recon_loss(self, image, recon):
        return self.criterionL1(recon, image)

    def calculate_style_loss(self, style_feats):
        return self.criterionL2(fake_gamma, real_gamma) + self.criterionL2(fake_beta, real_beta)

    def calculate_content_loss(self, content_feats):
        return self.criterionL2(fake_gamma, real_gamma) + self.criterionL2(fake_beta, real_beta)

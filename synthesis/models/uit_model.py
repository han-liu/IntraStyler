import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks_global_seg_3D, networks_local, networks_local_global
from .patchnce import PatchNCELoss
import util.util as util
from monai.networks.utils import one_hot
from .generic_UNet import Generic_UNet
from monai.losses import DiceCELoss


class UITModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--QS_mode', type=str, default="global", choices='(global, local, local_global)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_segA', type=float, default=0.5, help='weight for DiceCE loss for A segmentation')
        parser.add_argument('--lambda_segB', type=float, default=0.5, help='weight for DiceCE loss for B segmentation')
        parser.add_argument('--lambda_style', type=float, default=1e3, help='weight for style loss')
        parser.add_argument('--lambda_idt', type=float, default=0.0, help='weight for identity reconstruction loss')
        parser.add_argument('--lambda_sobel', type=float, default=1.0, help='weight for edge loss')
        parser.add_argument('--nce_idt', type=util.str2bool, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        assert opt.netG in ['resnet_9blocks_post']
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'DiceA', 'DiceB', 'sobelL1', 'style']#, 'idt']#, 'tumor']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_A_mask', 'seg_A', 'seg_B', 'fake_sobel', 'real_sobel', 'edge_mask']

        if self.opt.segB:
            self.loss_names.append('DiceB')
            self.visual_names.append('seg_B')

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.dce_loss = DiceCELoss(to_onehot_y=False, softmax=True)

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'S']
        else:  # during test time, only load G
            self.model_names = ['G']

        if self.opt.QS_mode == 'global':
            networks = networks_global_seg_3D
        elif self.opt.QS_mode == 'local':
            networks = networks_local
        else:
            networks = networks_local_global

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netS = Generic_UNet(input_channels=1, base_num_features=16, num_classes=4, num_pool=5, pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 1, 1]], conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], deep_supervision=False).cuda()

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters())+list(self.netS.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = self.real_A.size(0) // len(self.opt.gpu_ids)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()                  # calculate gradients for D
            self.backward_G()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

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
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A_mask = input['A_msk' if AtoB else 'B_msk'].to(self.device)
        self.edge_mask = input['A_edge' if AtoB else 'B_edge'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # self.code = input['code'].to(self.device)  # dynamic

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        # self.fake, self.seg = self.netG(self.real)
        # self.fake, self.seg = self.netG(self.real, condition=self.real_B)
        self.fake, self.seg = self.netG(self.real, condition=self.real_B)
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.seg_A = self.seg[:self.real_A.size(0)]
        self.seg_B = self.netS(self.fake_B)

        # style_loss ##########################################################################################
        self.fake_style = nn.AdaptiveAvgPool3d(1)(self.netG.encoder(self.fake_B))
        self.real_style = nn.AdaptiveAvgPool3d(1)(self.netG.encoder(self.real_B))
        self.fake_style = self.fake_style.view(-1, self.fake_style.size(1))
        self.real_style = self.real_style.view(-1, self.real_style.size(1))
        # obtain affine parameters
        self.fake_gamma, self.fake_beta = self.netG.syn_head.dyn_layer.generate_dyn_param(self.fake_style)
        self.real_gamma, self.real_beta = self.netG.syn_head.dyn_layer.generate_dyn_param(self.real_style)

        # calculate edge
        self.fake_sobel = networks_global_seg_3D.sobelLayer(self.fake_B) * self.edge_mask
        self.real_sobel = networks_global_seg_3D.sobelLayer(self.real_A).detach() * self.edge_mask

        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        self.feat_k = self.netG(self.real_A, self.nce_layers, encode_only=True)

    def backward_D(self):
        if self.opt.lambda_GAN > 0.0:
            """Calculate GAN loss for the discriminator"""
            fake = self.fake_B.detach()
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake)
            # print(pred_fake.shape)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            pred_real = self.netD(self.real_B)
            loss_D_real_unweighted = self.criterionGAN(pred_real, True)
            self.loss_D_real = loss_D_real_unweighted.mean()

            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        else:
            self.loss_D_real, self.loss_D_fake, self.loss_D = 0.0, 0.0, 0.0

    def backward_G(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.lambda_idt > 0.0:
            self.loss_idt = self.calculate_idt_loss(self.real_B, self.idt_B) * self.opt.lambda_idt

        # style loss ###############################################################################################
        # self.loss_style = self.calculate_style_loss(self.fake_style, self.real_style) * self.opt.lambda_style
        self.loss_style = self.calculate_style_loss(self.fake_gamma, self.fake_beta, self.real_gamma, self.real_beta) * self.opt.lambda_style

        self.loss_DiceA = self.calculate_dice_loss(self.seg_A, self.real_A_mask) * self.opt.lambda_segA
        self.loss_DiceB = self.calculate_dice_loss(self.seg_B, self.real_A_mask) * self.opt.lambda_segB 
        self.loss_sobelL1 = self.calculate_edge_loss(self.fake_sobel, self.real_sobel) * self.opt.lambda_sobel  # edge loss
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_DiceA + self.loss_DiceB + self.loss_sobelL1 + self.loss_style # + self.loss_idt#+ self.loss_tumor
        self.loss_G.backward()

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netF(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netF(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_dice_loss(self, src, tgt):
        tgt = one_hot(tgt, 4, dim=1)
        return self.dce_loss(src, tgt)
    
    def calculate_edge_loss(self, fake_sobel, real_sobel, weight_map=None):
        # we use weighting map to emphasize the edge of target structures: VS tumor and cochlea
        # for other regions, we only need to make sure they look like realT2 but they do not need to have the exact same anatomy
        if weight_map is None:
            edge_loss = self.criterionL1(fake_sobel, real_sobel)
        else:
            edge_loss = self.criterionL1(fake_sobel * weight_map, real_sobel* weight_map)
        return edge_loss

    # def calculate_style_loss(self, fake_style, real_style):
    #     return self.criterionL2(fake_style, real_style)

    def calculate_style_loss(self, fake_gamma, fake_beta, real_gamma, real_beta):
        return self.criterionL2(fake_gamma, real_gamma) + self.criterionL2(fake_beta, real_beta)

    def calculate_idt_loss(self, real_B, idt_B):
        return self.criterionL1(real_B, idt_B)

    def calculate_tumor_loss(self, fake_B, real_A_mask):
        threshold = -0.8
        tumor_mask = torch.clone(real_A_mask).detach()
        tumor_mask[tumor_mask==3] = 0
        tumor_mask[tumor_mask!=0] = 1
        wrong_region = ((fake_B < threshold) * tumor_mask).bool()
        true_region = ((fake_B >= threshold) * tumor_mask).bool()
        if torch.sum(wrong_region) == 0:
            return 0.0
        # breakpoint()
        mu, std = float(fake_B[true_region].mean()), float(fake_B[true_region].std())
        correct = torch.normal(mean=mu, std=std, size=tuple(fake_B.size())).cuda()
        loss = self.criterionL1(fake_B[wrong_region], correct[wrong_region])
        return loss

    # v3
    # def update_segB_lambda(self, epochNum):
    #     if epochNum <= 50:
    #         self.opt.lambda_segB = 0
    #     elif epochNum > 50 and epochNum < 100:
    #         self.opt.lambda_segB = 0.5
    #     elif epochNum > 100 and epochNum < 200:
    #         self.opt.lambda_segB = 1
    #     elif epochNum > 200:
    #         self.opt.lambda_segB = 0.5
    #     print('update segB lambda: %f' % (self.opt.lambda_segB))


    def update_segB_lambda(self, epochNum):
        if epochNum <= 200:
            self.opt.lambda_segB = 0.2
        elif epochNum > 200 and epochNum < 400:
            self.opt.lambda_segB = 0.35
        elif epochNum > 400 and epochNum < 600:
            self.opt.lambda_segB = 0.5
        elif epochNum > 600:
            self.opt.lambda_segB = 1.2

        print('update segB lambda: %f' % (self.opt.lambda_segB))
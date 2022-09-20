import ntpath
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data import create_eval_dataloader
from metric import get_fid, get_mIoU
from metric.inception import InceptionV3
from metric.mIoU_score import DRNSeg
from utils import util
from . import networks
from .base_model import BaseModel
from .modules.loss import GANLoss


class SketchGAN(BaseModel):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """
    def __init__(self, opt):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        BaseModel.__init__(self, opt)
        
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids \
            else torch.device('cpu')
        if opt.isTrain:
            self.save_dir = os.path.join(opt.log_dir, 'checkpoints')
            os.makedirs(self.save_dir, exist_ok=True)
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
       
        self.image_paths = []
        self.metric = 0
        self.loss_names = [
            'G_gan', 'G_recon', 'D_real', 'D_fake', 'G_comp_cost'
        ]
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D']
        self.netG = networks.define_G(opt.input_nc,
                                      opt.output_nc,
                                      opt.ngf,
                                      opt.netG,
                                      opt.norm,
                                      opt.dropout_rate,
                                      opt.init_type,
                                      opt.init_gain,
                                      self.gpu_ids,
                                      opt=opt)

        self.netD = networks.define_D(opt.input_nc + opt.output_nc,
                                      opt.ndf,
                                      opt.netD,
                                      opt.n_layers_D,
                                      opt.norm,
                                      opt.init_type,
                                      opt.init_gain,
                                      self.gpu_ids,
                                      opt=opt)

        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        if opt.recon_loss_type == 'l1':
            self.criterionRecon = torch.nn.L1Loss()
        elif opt.recon_loss_type == 'l2':
            self.criterionRecon = torch.nn.MSELoss()
        elif opt.recon_loss_type == 'smooth_l1':
            self.criterionRecon = torch.nn.SmoothL1Loss()
        else:
            raise NotImplementedError(
                'Unknown reconstruction loss type [%s]!' % opt.loss_type)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader = create_eval_dataloader(self.opt)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.inception_model.to(self.device)
        self.inception_model.eval()

        self.best_fid = 1e9
        self.best_mIoU = -1e9
        self.fids, self.mIoUs = [], []
        self.is_best = False
        self.Tacts, self.Sacts = {}, {}
        self.npz = np.load(opt.real_stat_path)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
         parser = super(SketchGAN, SketchGAN).modify_commandline_options(
            parser, is_train)
        parser.add_argument('--restore_G_path',
                            type=str,
                            default=None,
                            help='the path to restore the generator')
        parser.add_argument('--restore_D_path',
                            type=str,
                            default=None,
                            help='the path to restore the discriminator')
        parser.add_argument('--recon_loss_type',
                            type=str,
                            default='l1',
                            choices=['l1', 'l2', 'smooth_l1'],
                            help='the type of the reconstruction loss')
        parser.add_argument('--lambda_recon',
                            type=float,
                            default=100,
                            help='weight for reconstruction loss')
        parser.add_argument('--lambda_gan',
                            type=float,
                            default=1,
                            help='weight for gan loss')
        parser.add_argument(
            '--real_stat_path',
            type=str,
            required=True,
            help=
            'the path to load the groud-truth images information to compute FID.'
        )
        parser.add_argument('--lambda_comp_cost',
                            type=float,
                            default=0,
                            help='weight for comp cost loss')
        parser.add_argument('--comp_cost',
                            type=str,
                            default='l1',
                            choices=['l1'],
                            help='the type of the comp cost loss')
        parser.add_argument('--l1_renorm',
                            action='store_true',
                            help='renorm for l1')
        parser = networks.modify_commandline_options(parser, is_train)
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1).detach()
        real_AB = torch.cat((self.real_A, self.real_B), 1).detach()
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake,
                                             False,
                                             for_discriminator=True)

        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real,
                                             True,
                                             for_discriminator=True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_gan = self.criterionGAN(
            pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        self.loss_G_recon = self.criterionRecon(
            self.fake_B, self.real_B) * self.opt.lambda_recon

        if getattr(self.opt, 'lambda_comp_cost', 0) > 0:
            p = int(self.opt.comp_cost[-1])
            self.loss_G_comp_cost = getattr(
                getattr(self.netG, 'module', self.netG), 'get_comp_cost',
                lambda p, renorm: 0)(p=p, renorm=self.opt.l1_renorm)
            self.loss_G_comp_cost *= self.opt.lambda_comp_cost

        self.loss_G = self.loss_G_gan + self.loss_G_recon
        if getattr(self.opt, 'lambda_comp_cost', 0) > 0:
            self.loss_G += self.loss_G_comp_cost
        self.loss_G.backward()
    

    @abstractmethod
    def optimize_parameters(self, steps):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def setup(self, opt, verbose=True):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, opt)
                for optimizer in self.optimizers
            ]
        self.load_networks(verbose=verbose,
                           teacher_only=getattr(opt, 'prune_continue', False))
        if verbose:
            self.print_networks()

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
                if hasattr(self.opt, 'log_dir'):
                    with open(
                            os.path.join(self.opt.log_dir,
                                         'net' + name + '.txt'), 'w') as f:
                        f.write(str(net) + '\n')
                        f.write(
                            '[Network %s] Total number of parameters : %.3f M\n'
                            % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self, logger=None):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.opt.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        if logger is not None:
            logger.print_info('learning rate = %.7f\n' % lr)
        else:
            print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_set = OrderedDict()
        for name in self.loss_names:
            if not hasattr(self, 'loss_' + name):
                continue
            key = name

            def has_number(s):
                for i in range(10):
                    if str(i) in s:
                        return True
                return False

            if has_number(key):
                key = 'Specific_loss/' + key
            elif key.startswith('D_'):
                key = 'D_loss/' + key
            elif key.startswith('G_'):
                key = 'G_loss/' + key
            else:
                assert False
            errors_set[key] = float(getattr(self, 'loss_' + name))
        return errors_set

    def load_networks(self,
                      verbose=True,
                      teacher_only=False,
                      restore_pretrain=True):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.opt, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path, verbose)

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if isinstance(net, DataParallel):
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def evaluate_model(self, step):
        self.is_best = False

        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netG.eval()

        fakes, names = [], []
        cnt = 0
        for i, data_i in enumerate(tqdm(self.eval_dataloader)):
            self.set_input(data_i)
            self.test()
            fakes.append(self.fake_B.cpu())
            for j in range(len(self.image_paths)):
                short_path = ntpath.basename(self.image_paths[j])
                name = os.path.splitext(short_path)[0]
                names.append(name)
                if cnt < 10 or save_image:
                    input_im = util.tensor2im(self.real_A[j])
                    real_im = util.tensor2im(self.real_B[j])
                    fake_im = util.tensor2im(self.fake_B[j])
                    util.save_image(input_im,
                                    os.path.join(save_dir, 'input',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(real_im,
                                    os.path.join(save_dir, 'real',
                                                 '%s.png' % name),
                                    create_dir=True)
                    util.save_image(fake_im,
                                    os.path.join(save_dir, 'fake',
                                                 '%s.png' % name),
                                    create_dir=True)
                cnt += 1

        fid = get_fid(fakes,
                      self.inception_model,
                      self.npz,
                      device=self.device,
                      batch_size=self.opt.eval_batch_size)
        if fid < self.best_fid:
            self.is_best = True
            self.best_fid = fid
        self.fids.append(fid)
        if len(self.fids) > 3:
            self.fids.pop(0)

        ret = {
            'metric/fid': fid,
            'metric/fid-mean': sum(self.fids) / len(self.fids),
            'metric/fid-best': self.best_fid
        }

        self.netG.train()
        return ret

    # def profile(self):
    #     raise NotImplementedError
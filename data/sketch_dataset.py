"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os.path
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
from PIL import Image

from utils import util
from .base_dataset import BaseDataset, get_params, get_transform
from .image_folder import make_dataset


class SketchDataset(BaseDataset):
    def __init__(self, opt, transforms, csv_file):
        super(SketchDataset, self).__init__(opt)
        self.initialize(opt, transforms, csv_file)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument(
            '--no_pairing_check',
            action='store_true',
            help=
            'If specified, skip sanity check of correct label-image file pairing'
        )
        parser.add_argument(
            '--no_instance',
            action='store_true',
            help='if specified, do *not* add instance map as input')
        parser.add_argument(
            '--contain_dontcare_label',
            action='store_true',
            help='if the label map contains dontcare label (dontcare=255)')
        parser.set_defaults(preprocess='scale_width',
                            aspect_ratio=2,
                            load_size=256,
                            crop_size=256,
                            direction='BtoA',
                            display_winsize=256,
                            input_nc=3,
                            num_threads=0)
        return parser



    def __getitem__(self, idx):
        # Label Image
        cimg_path = os.path.join(self.dir,self.img_names.iloc[idx,1])
        print(cimg_path)
        condition = Image.open(cimg_path).convert('RGB')
        T= torchvision.transforms.Resize((256,256))
        condition = T(condition)

        # input image (real images)
        rimg_path = os.path.join(self.dir,self.img_names.iloc[idx,0])
        print(rimg_path)
        real = Image.open(rimg_path).convert('RGB')
        T = torchvision.transforms.Resize((128,128))
        real = T(real)
    
        if self.transforms:
            print("Successful transforms")
            real = self.transforms(real)
            print(real.shape)
            condition = self.transforms(condition)
            print(condition.shape)

        instance_tensor = torch.zeros((3,256, 256))

        input_dict = {
            'label': condition,
            'instance': instance_tensor,
            'image': real,
            'path': rimg_path,
        }
        return input_dict
        

    def initialize(self, opt, transforms, csv_file):
        self.opt = opt
        self.dir = opt.dataroot
        phase = opt.phase
        self.img_names = pd.read_csv(csv_file)
        self.transforms = transforms

        size = len(self.img_names)
        self.dataset_size = size
        self.label_cache = {}
        self.image_cache = {}
        self.instance_cache = {}


    def __len__(self):
        if self.opt.max_dataset_size == -1:
            return self.dataset_size
        else:
            return self.opt.max_dataset_size
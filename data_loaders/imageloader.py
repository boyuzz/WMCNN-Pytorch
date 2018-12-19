# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
# @Time    : 11/09/2018 11:33 PM
# @Author  : Boyu Zhang
# @Site    : 
# @File    : imageloader.py
# @Software: PyCharm
"""

import torch.utils.data as data
from bases.data_loader_base import DataLoaderBase
from torch.utils.data import DataLoader
from utils.imresize import imresize
# import cv2
import h5py
import torch
import random

import os
from os.path import join
from skimage import io, transform, color
from utils.imresize import imresize
import numpy as np
import torchvision.transforms as tfs
from PIL import Image
from os import listdir
from utils.utils import is_image_file
from scipy import misc
import scipy.io as sio


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class SingleImageLoader(data.Dataset):
    def __init__(self, image_dirs, is_gray=False, crop_size=64, random_scale=True, rotate=True, fliplr=True,
                 fliptb=True, preload=True):
        super(SingleImageLoader, self).__init__()

        self.image_filenames = []
        self.preload = preload

        all_files = os.walk(image_dirs)
        for path, dir_list, file_list in all_files:
            self.image_filenames.extend(join(path, x) for x in file_list if is_image_file(x))
        if self.preload:
            self.image_list = []
            for file in self.image_filenames:
                img = Image.open(file).convert('RGB')
                self.image_list.append(img)

        self.is_gray = is_gray
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.random_scale = random_scale

    def __getitem__(self, index):
        # load image
        # img = io.imread(self.image_filenames[index])
        if self.preload:
            img = self.image_list[index]
        else:
            img = Image.open(self.image_filenames[index]).convert('RGB')
        # original HR image size
        # (hr_img_h, hr_img_w, channel) = img.shape
        hr_img_w = img.size[0]
        hr_img_h = img.size[1]

        # random scaling between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            # ratio = random.uniform(0.5, 1)
            ratio = random.randint(5, 10)*0.1
            if hr_img_w * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)
            # tfs resize set size as (h, w)
            # transform = tfs.Resize((scale_h, scale_w), interpolation=Image.BICUBIC)
            img = np.asarray(img)
            img = imresize(img, output_shape=(scale_h, scale_w))
            img = Image.fromarray(img.squeeze())
            # img = transform(img)

        # random crop
        if self.crop_size:
            transform = tfs.RandomCrop(self.crop_size)
            img = transform(img)

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(0, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = tfs.RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # REMEMBER: change to gray image in the last, otherwise the color range will be out of [16, 235]
        if self.is_gray:
            # only Y-channel is super-resolved
            # skimage can convert RGB to YCbCr in [16, 235] while PIL cannot.
            img = np.asarray(img)
            img = color.rgb2ycbcr(img)/255
            channel = len(img.shape)
            img, _, _ = np.split(img, indices_or_sections=channel, axis=-1)
            # img = img.convert('YCbCr')
            # precision degrade from float64 to float32
            img = Image.fromarray(img.squeeze())
            # img, _, _ = img.split()

        # hr_img HR image
        hr_transform = tfs.ToTensor()
        img = hr_transform(img)

        # easy = img.data.cpu().numpy()
        # # easy = np.asarray(img)
        # if easy.min() < 0.0627 or easy.max() > 0.922:
        #     print('ERROR IN LOADING Y CHANNEL!')

        # Bicubic interpolated image
        # bc_transform = tfs.Compose([tfs.ToPILImage(), tfs.Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        # bc_img = bc_transform(lr_img)

        return img

    def __len__(self):
        return len(self.image_filenames)


class BenchmarkLoader(data.Dataset):
    def __init__(self, image_dir, is_gray=False, scale_factor=[2, 3, 4]):
        super(BenchmarkLoader, self).__init__()

        self.image_filenames = []
        all_files = os.walk(image_dir)
        for path, dir_list, file_list in all_files:
            self.image_filenames.extend(join(path, x) for x in file_list if is_image_file(x))
        # self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.is_gray = is_gray
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        # img = io.imread(self.image_filenames[index])
        img = Image.open(self.image_filenames[index]).convert('RGB')
        # original HR image size
        # (hr_img_h, hr_img_w, channel) = img.shape
        hr_img_w = img.size[0]
        hr_img_h = img.size[1]

        if self.is_gray:
            # only Y-channel is super-resolved
            # skimage can convert RGB to YCbCr in [16, 235] while PIL cannot.
            img = np.asarray(img)
            img = color.rgb2ycbcr(img) / 255
            channel = len(img.shape)
            img, _, _ = np.split(img, indices_or_sections=channel, axis=-1)
            # img = img.convert('YCbCr')
            # precision degrade from float64 to float32
            img = Image.fromarray(img.squeeze())
            # img, _, _ = img.split()

        # img = Image.fromarray(img.squeeze())

        # hr_img HR image
        hr_transform = tfs.ToTensor()
        hr_img = hr_transform(img)

        # determine lr_img LR image size
        lr_img_list = []
        for sf in self.scale_factor:
            lr_img_w = hr_img_w // sf
            lr_img_h = hr_img_h // sf

            # lr_img LR image
            # tfs resize set size as (h, w)
            img = np.asarray(img)
            img = imresize(img, output_shape=(lr_img_h, lr_img_w))
            img = Image.fromarray(img.squeeze())
            lr_transform = tfs.ToTensor()

            lr_img = lr_transform(img)
            lr_img_list.append(lr_img)

        # Bicubic interpolated image
        # bc_transform = tfs.Compose([tfs.ToPILImage(), tfs.Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), tfs.ToTensor()])
        # bc_img = bc_transform(lr_img)
        return lr_img_list, hr_img

    def __len__(self):
        return len(self.image_filenames)


class SampleLoader(data.Dataset):
    def __init__(self, image_dir, is_gray=False, preload=False):
        super(SampleLoader, self).__init__()

        self.image_filenames = []
        self.preload = preload

        all_files = os.walk(image_dir)
        for path, dir_list, file_list in all_files:
            self.image_filenames.extend(join(path, x) for x in file_list if is_image_file(x))
        if self.preload:
            self.image_list = []
            for file in self.image_filenames:
                img = Image.open(file).convert('RGB')
                self.image_list.append(img)

        self.is_gray = is_gray

    def __getitem__(self, index):
        # load image
        # img = io.imread(self.image_filenames[index])
        if self.preload:
            img = self.image_list[index]
        else:
            img = Image.open(self.image_filenames[index]).convert('RGB')

        # hr_img HR image
        hr_transform = tfs.ToTensor()
        img = hr_transform(img)

        return img


class ImageLoader(DataLoaderBase):
    def __init__(self, config=None):
        super(ImageLoader, self).__init__(config)


    def get_sample_train(self):
        data_path = self.config['data_path'] + self.config['train_path']
        train_set = SingleImageLoader(data_path, is_gray=self.config['is_gray'],
                                      crop_size=False, random_scale=False,
                                      rotate=False, fliplr=False,
                                      fliptb=False, preload=self.config['preload'])

        train_loader = DataLoader(dataset=train_set, num_workers=self.config['threads'],
                                  batch_size=self.config['batch_size'], shuffle=True)
        return train_loader

    def get_hdf5_sample_data(self):
        data_path = self.config['data_path'] + self.config['train_path']
        train_set = DatasetFromHdf5(data_path)
        train_loader = DataLoader(dataset=train_set, num_workers=self.config['threads'],
                                    batch_size=self.config['batch_size'], shuffle=True)
        return train_loader

    def get_hdf5_data(self):
        data_path = self.config['data_path'] + self.config['train_path']
        train_set = DatasetStackFromHdf5(data_path)
        train_loader = DataLoader(dataset=train_set, num_workers=self.config['threads'],
                                    batch_size=self.config['batch_size'], shuffle=True)
        return train_loader

    def get_wmcnn_hdf5_data(self):
        data_path = self.config['data_path'] + self.config['train_path']
        train_set = DatasetWMCNNFromHdf5(data_path)

        # if self.config['distributed']:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        # else:
        train_sampler = None

        train_loader = DataLoader(dataset=train_set, num_workers=self.config['threads'],
                                    batch_size=self.config['batch_size'], shuffle=(train_sampler is None), sampler=train_sampler)
        return train_loader

    def get_train_data(self):
        data_path = self.config['data_path'] + self.config['train_path']
        train_set = SingleImageLoader(data_path, is_gray=self.config['is_gray'],
                                      crop_size=self.config['crop_size'], random_scale=self.config['random_scale'],
                                      rotate=self.config['is_rotate'], fliplr=self.config['is_fliplr'],
                                      fliptb=self.config['is_fliptb'], preload=self.config['preload'])

        train_loader = DataLoader(dataset=train_set, num_workers=self.config['threads'],
                                  batch_size=self.config['batch_size'], shuffle=True)
        return train_loader

    def get_val_data(self):
        data_path = self.config['data_path'] + self.config['train_path']
        val_set = BenchmarkLoader(data_path, is_gray=self.config['is_gray'])

        val_loader = DataLoader(dataset=val_set, num_workers=self.config['threads'], batch_size=1, shuffle=False, pin_memory=True)
        return val_loader

    def get_test_data(self):
        data_path = self.config['data_path'] + self.config['test_path']
        test_set = BenchmarkLoader(data_path, is_gray=self.config['is_gray'])

        test_loader = DataLoader(dataset=test_set, num_workers=self.config['threads'], batch_size=1, shuffle=True)
        return test_loader


class TestImageLoader:
    def __init__(self, config=None):
        self.config = config

    def is_mat_file(self, file, upscale):
        file_list = any(file.endswith(extension) for extension in [".mat"])
        return file_list and ('x{}'.format(upscale) in file)

    def get_test_mat(self):
        data_path = self.config['data_path'] + self.config['test_path']
        files_list = [join(data_path, x) for x in sorted(listdir(data_path)) if self.is_mat_file(x, self.config['upscale'])]
        img_list = []
        # TODO: use scipy.misc when tuning, finally use imresize for matlab-like bicubic performance
        for file in files_list:
            img_combo = sio.loadmat(file)
            # np.transpose(param_array, axes=[-1, *range(len(param_array.shape) - 1)])
            (rows, cols, channel) = img_combo['im_l_ycbcr'].shape
            img_y, img_cb, img_cr = np.split(img_combo['im_l_ycbcr'], indices_or_sections=channel, axis=2)

            # io.imshow(img_cr.squeeze())
            # io.show()

            img_bundle = {'name': os.path.basename(file), 'origin': img_combo['im_l']/255, 'x': img_combo['im_l_y']/255, 'y': img_combo['im_gt_y']/255, 'cb': img_cb.squeeze(), 'cr': img_cr.squeeze(),
                          'size':img_combo['im_gt_y'].shape}
            # img_bundle = {'name': os.path.basename(file), 'x': img_y_lr, 'y': img_y, 'cb': img_cb.squeeze(),
            #               'cr': img_cr.squeeze(),
            #               'size': size_lr}
            img_list.append(img_bundle)
        return img_list

    def get_test_data(self):
        data_path = self.config['data_path'] + self.config['test_path']
        files_list = [join(data_path, x) for x in sorted(listdir(data_path)) if is_image_file(x)]
        img_list = []
        # TODO: use scipy.misc when tuning, finally use imresize for matlab-like bicubic performance
        for file in files_list:
            img = io.imread(file)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)
            img_ycbcr = color.rgb2ycbcr(img)/255
            (rows, cols, channel) = img_ycbcr.shape
            img_y, img_cb, img_cr = np.split(img_ycbcr, indices_or_sections=channel, axis=2)
            size_lr = (int(rows // self.config['upscale']), int(cols // self.config['upscale']))

            # img_y_lr = cv2.resize(img_y.squeeze(), dsize=(int(cols // self.config['upscale']), int(rows // self.config['upscale'])),
            #                      interpolation=cv2.INTER_CUBIC)
            # img_y_lr = misc.imresize(img_y.squeeze(), size=size_lr, interp='bicubic', mode='F')
            img_y_lr = imresize(img_y.squeeze(), output_shape=size_lr)
            img_cb = imresize(img_cb.squeeze(), output_shape=size_lr)
            img_cr = imresize(img_cr.squeeze(), output_shape=size_lr)
            img = imresize(img, output_shape=size_lr)

            # import matplotlib.pyplot as plt
            # plt.imshow(img_y_lr, cmap ='gray')
            # plt.show()

            img_bundle = {'name': os.path.basename(file), 'origin': img, 'x': img_y_lr, 'y': img_ycbcr, 'cb': img_cb.squeeze(), 'cr': img_cr.squeeze(),
                          'size':img_ycbcr.shape[0:-1]}
            # img_bundle = {'name': os.path.basename(file), 'x': img_y_lr, 'y': img_y, 'cb': img_cb.squeeze(),
            #               'cr': img_cr.squeeze(),
            #               'size': size_lr}
            img_list.append(img_bundle)

        return img_list


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path, 'r')
        self.data = hf.get("data")
        self.target = hf.get("label")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]


class DatasetWMCNNFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetWMCNNFromHdf5, self).__init__()
        hf = h5py.File(file_path, 'r')
        self.data = hf.get("data")
        self.CA = hf.get("CA")
        self.CH = hf.get("CH")
        self.CV = hf.get("CV")
        self.CD = hf.get("CD")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.CA[index,:,:,:]).float(),\
               torch.from_numpy(self.CH[index,:,:,:]).float(), torch.from_numpy(self.CV[index,:,:,:]).float(),\
               torch.from_numpy(self.CD[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]


class DatasetStackFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetStackFromHdf5, self).__init__()
        hf = h5py.File(file_path, 'r')
        self.data = hf.get("data")
        # self.nearest = hf.get("nearest")
        # self.bilinear = hf.get("bilinear")
        # self.bicubic = hf.get("bicubic")
        self.vdsr = hf.get("vdsr")
        self.lapsrn = hf.get("lapsrn")
        self.target = hf.get("label")

    def __getitem__(self, index):
        # torch.from_numpy(self.nearest[index]).float(),
        # torch.from_numpy(self.bilinear[index]).float(),
        # torch.from_numpy(self.bicubic[index]).float(),
        data = ([torch.from_numpy(self.data[index]).float(),
                torch.from_numpy(self.vdsr[index]).float(),
                torch.from_numpy(self.lapsrn[index]).float()],
                torch.from_numpy(self.target[index]).float())
        return data

    def __len__(self):
        return self.target.shape[0]

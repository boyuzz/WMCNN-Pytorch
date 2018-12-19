#!/usr/bin/env python
#  -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""
import os
import shutil
import datetime
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from scipy import misc
from utils import imresize


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[INFO] Folder "%s" exist, deleting folder.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[INFO] Folder "%s" does not exist, building folder.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def timestamp_2_readable(time_stamp):
    """
    时间戳转换为可读时间
    :param time_stamp: 时间戳，当前时间：time.time()
    :return: 可读时间字符串
    """
    return datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H_%M_%S')


def traverse_dir_files(root_dir, ext=None):
    """
    列出文件夹中的文件, 深度遍历
    :param root_dir: 根目录
    :param ext: 后缀名
    :return: [文件路径列表, 文件名称列表]
    """
    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if ext:  # 根据后缀名搜索
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    paths_list, names_list = sort_two_list(paths_list, names_list)
    return paths_list, names_list


def sort_two_list(list1, list2):
    """
    排序两个列表
    :param list1: 列表1
    :param list2: 列表2
    :return: 排序后的两个列表
    """
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def safe_div(x, y):
    """
    安全除法
    :param x: 被除数
    :param y: 除数
    :return: 结果
    """
    x = float(x)
    y = float(y)
    if y == 0.0:
        return 0.0
    else:
        return x / y


def data_preprocess(config):
    # if os.path.exists("dataset_train.csv"):
    #     return
    mkdir_if_not_exist(config.save_path)

    df_train = pd.read_csv(config.input_train_path)
    df_train["word_seg"] = df_train.word_seg.str.replace("n", " ")
    idx = np.arange(df_train.shape[0])
    np.random.seed(config.seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * config.val_ratio)

    df_train.iloc[idx[val_size:], :].to_csv(config.save_path+config.save_train_path, index=False)
    df_train.iloc[idx[:val_size], :].to_csv(config.save_path+config.save_val_path, index=False)

    df_test = pd.read_csv(config.input_test_path)
    df_test["word_seg"] = df_test.word_seg.str.replace("n", " ")
    df_test.to_csv(config.save_path+config.save_test_path, index=False)


def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_kaming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def sum_param(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, torch.nn.ReLU):
            if m.inplace:
                continue

        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def resize_tensor(tensor, size=None, scale=None):
    # TODO: Use multiple interpolation methods to reduce bias to bicubic
    # TODO: tackle 3-channels
    # TODO: Change all the bicubic operators as matlab bicubic
    if scale is not None:
        hr_w = tensor.shape[2]
        hr_h = tensor.shape[3]
        lr_w = int(np.ceil(hr_w * scale))
        lr_h = int(np.ceil(hr_h * scale))
        size = (lr_w, lr_h)

    # print(tensor.shape)
    y_numpy = tensor.data.cpu().numpy().squeeze(axis=1)
    # print(y_numpy.shape)
    newt = [imresize.imresize(y_sub, output_shape=size) for y_sub in y_numpy]
    # tensor = [misc.imresize(y_sub, size=size, interp='bicubic', mode='F') for y_sub in y_numpy]
    newt = np.array(newt, dtype=np.float32)
    newt = np.expand_dims(newt, axis=1)
    newt = torch.from_numpy(newt)

    return newt


def preview(*args):
    # dataiter = iter(training_data_loader)
    # images = dataiter.next()

    def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # show images
    # plt.subplot(1, 2, 1)
    elements = len(args)
    for i in range(elements):
        plt.subplot(elements, 1, i+1)
        imshow(torchvision.utils.make_grid(args[i]))

    plt.show()


def colorize(y, cb, cr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.float64)
    img[:, :, 0] = y*255
    img[:, :, 1] = cb*255
    img[:, :, 2] = cr*255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    img = np.asarray(img)
    # from skimage import color
    # img = color.ycbcr2rgb(img)
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".jpeg", ".bmp", ".JPG"])


def modcrop(imgs, scale):
    if len(imgs.shape) == 2:
        width, height = imgs.shape
        width = width - np.mod(width, scale)
        height = height - np.mod(height, scale)
        imgs = imgs[:width, :height]
    else:
        width, height, channel = imgs.shape
        width = width - np.mod(width, scale)
        height = height - np.mod(height, scale)
        imgs = imgs[:width, :height, :]

    return imgs


def shave(imgs, border_size=0):
    size = list(imgs.shape)
    if len(size) == 4:
        shave_imgs = torch.FloatTensor(size[0], size[1], size[2]-border_size*2, size[3]-border_size*2)
        for i, img in enumerate(imgs):
            shave_imgs[i, :, :, :] = img[:, border_size:-border_size, border_size:-border_size]
        return shave_imgs
    elif len(size) == 3:
        return imgs[border_size:-border_size, border_size:-border_size, :]
    else:
        return imgs[border_size:-border_size, border_size:-border_size]



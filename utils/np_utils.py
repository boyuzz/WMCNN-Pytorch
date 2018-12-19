#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20

参考: https://juejin.im/post/5acfef976fb9a028db5918b5
"""

import numpy as np
import math


def prp_2_oh_array(arr):
    """
    概率矩阵转换为OH矩阵
    arr = np.array([[0.1, 0.5, 0.4], [0.2, 0.1, 0.6]])
    :param arr: 概率矩阵
    :return: OH矩阵
    """
    arr_size = arr.shape[1]  # 类别数
    arr_max = np.argmax(arr, axis=1)  # 最大值位置
    oh_arr = np.eye(arr_size)[arr_max]  # OH矩阵
    return oh_arr


def interval_size(lr_size, hr_size, max_gradual_scale=None, n_step=None):
    inter_sizes = []
    ratio = hr_size[0]/lr_size[0]

    if n_step is None:
        if max_gradual_scale is None:
            raise RuntimeError("You need to set at least one parameter of max_gradual_scale or n_step!")
        n_step = max(int(np.ceil(math.log(ratio, max_gradual_scale))), 1)

    gradual_lambda = math.pow(ratio, 1/n_step)
    for i in range(n_step-1):
        if i == 0:
            inter_sizes.append((round(lr_size[0] * gradual_lambda), round(lr_size[1] * gradual_lambda)))
        else:
            inter_sizes.append((round(inter_sizes[-1][0] * gradual_lambda), round(inter_sizes[-1][1] * gradual_lambda)))
    return inter_sizes

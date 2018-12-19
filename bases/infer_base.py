#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""


class InferBase(object):
    """
    推断基类
    """

    def __init__(self, config):
        self.config = config  # 配置

    def load_model(self, name):
        """
        加载模型
        """
        raise NotImplementedError

    def predict(self, data, **kwargs):
        """
        预测结果
        """
        raise NotImplementedError

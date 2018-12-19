# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""
import torch.nn as nn
from utils.utils import weights_init_kaming, weights_init_normal


class ModelBase(nn.Module):
    """
    模型基类
    """

    def __init__(self, config):
        super(ModelBase, self).__init__()
        self.config = config  # 配置

    # def save(self, checkpoint_path):
    #     """
    #     存储checkpoint, 路径定义于配置文件中
    #     """
    #     if self.model is None:
    #         raise Exception("[Exception] You have to build the model first.")
    #
    #     print("[INFO] Saving model...")
    #     self.model.save_weights(checkpoint_path)
    #     print("[INFO] Model saved")
    #
    # def load(self, checkpoint_path):
    #     """
    #     加载checkpoint, 路径定义于配置文件中
    #     """
    #     if self.model is None:
    #         raise Exception("[Exception] You have to build the model first.")
    #
    #     print("[INFO] Loading model checkpoint {} ...\n".format(checkpoint_path))
    #     self.model.load_weights(checkpoint_path)
    #     print("[INFO] Model loaded")

    def forward(self, x):
        """
        构建模型
        """
        raise NotImplementedError

    def weight_init(self):
        '''
        initialize model parameters
        :return: None
        '''
        raise NotImplementedError

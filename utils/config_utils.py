# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""
import argparse
import json
import time
from utils.utils import timestamp_2_readable

import os

from utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    """
    with open(json_file, 'r') as config_file:
        config = json.load(config_file)  # 配置字典

    return config


def process_config(json_file):
    """
    解析Json文件
    :param json_file: 配置文件
    :return: 配置类
    """
    config = get_config_from_json(json_file)
    config["trainer"]['tb_dir'] = os.path.join("experiments", config['exp_name'], "logs/")  # 日志
    config["trainer"]['cp_dir'] = os.path.join("experiments", config['exp_name'], "checkpoints/")  # 模型
    config["trainer"]['img_dir'] = os.path.join("experiments", config['exp_name'], "images/")  # 网络
    config["trainer"]['preds_dir'] = os.path.join("experiments", config['exp_name'], "preds/")  # 预测输出
    ticks = time.time()
    config["trainer"]['time'] = timestamp_2_readable(ticks) # the time when starting

    mkdir_if_not_exist(config["trainer"]['tb_dir'])  # 创建文件夹
    mkdir_if_not_exist(config["trainer"]['cp_dir'])  # 创建文件夹
    mkdir_if_not_exist(config["trainer"]['img_dir'])  # 创建文件夹
    mkdir_if_not_exist(config["trainer"]['preds_dir'])  # 创建文件夹
    return config


def get_train_args():
    """
    添加训练参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='path',
        default='configs/rrgun.json',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser


def get_process_args():
    """
    添加数据处理参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='path',
        default='configs/data_preprocess.json',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser


def get_test_args():
    """
    添加测试路径
    :return: 参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='C',
        default='configs/rrgun.json',
        help='add a configuration file')
    parser.add_argument(
        '-m', '--mod',
        dest='model',
        metavar='',
        default='None',
        help='add a model file')
    args = parser.parse_args()
    return args, parser

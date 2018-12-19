#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20

现象：
同样放大两倍，一步到位比progressive2次到位好一点点，看看如何训练，参考IRGUN
"""

from data_loaders.imageloader import ImageLoader
import importlib

# from trainers.sr_trainer import SRTrainer
# from trainers.preload_trainer import SRTrainer
from trainers.wmcnn_trainer import SRTrainer
from utils.config_utils import process_config, get_train_args
from utils.utils import print_network
import shutil
import os
import json
import torch


def train_main():
    """
    训练模型

    :return:
    """
    print('[INFO] Retrieving configuration...')
    # import torch
    # print(torch.__version__)
    parser = None
    args = None
    config = None
    # TODO: modify the path of best checkpoint after training
    try:
        args, parser = get_train_args()
        # args.config = 'experiments/stacksr lr=1e-3 28init 3x/stacksr.json'
        # args.config = 'configs/lapsrn.json'
        config = process_config(args.config)
        shutil.copy2(args.config, os.path.join("experiments", config['exp_name']))
    except Exception as e:
        print('[Exception] Configuration is invalid, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] Refer to: python main_train.py -c configs/wmcnn.json')
        exit(0)
    # config = process_config('configs/train_textcnn.json')
    # np.random.seed(config.seed)  # 固定随机数

    print('[INFO] Loading data...')
    dl = ImageLoader(config=config['train_data_loader'])

    print('[INFO] Building graph...')
    try:
        Net = importlib.import_module('models.{}'.format(config['trainer']['net'])).Net
        model = Net(config=config['model'])
        print_network(model)
    except ModuleNotFoundError:
        raise RuntimeWarning("The model name is incorrect or does not exist! Please check!")

    # if config['distributed']:
    #     os.environ['MASTER_ADDR'] = '127.0.0.1'
    #     os.environ['MASTER_PORT'] = '29500'
    #     torch.distributed.init_process_group(backend='nccl', world_size=4, rank=2)

    print('[INFO] Training the graph...')
    # trainer = SRTrainer(
    #     model=model,
    #     data={'train': dl.get_train_data(), 'test': dl.get_test_data()},
    #     config=config['trainer'])
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    trainer = SRTrainer(
        model=model,
        data={'train': dl.get_wmcnn_hdf5_data(), 'test': dl.get_test_data()},
        # data={'train': dl.get_hdf5_data(), 'test': dl.get_test_data()},
        config=config['trainer'])

    highest_score, best_model = trainer.train()
    with open(os.path.join("experiments", config['exp_name'], 'performance.txt'), 'w') as f:
        f.writelines(str(highest_score))

    json_file = os.path.join("./experiments", config['exp_name'], os.path.basename(args.config))
    with open(json_file, 'w') as file_out:
        config['trainer']['checkpoint'] = best_model
        json.dump(config, file_out, indent=2)

    print('[INFO] Training is completed.')


if __name__ == '__main__':
    train_main()

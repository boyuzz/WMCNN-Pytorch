#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""

from data_loaders.imageloader import TestImageLoader
from infers.sr_infer import SRInfer
from utils.config_utils import process_config, get_test_args
from utils.utils import print_network
import importlib


def test_main():
    print('[INFO] Retrieving configuration...')
    parser = None
    config = None

    try:
        args, parser = get_test_args()
        # args.config = 'experiments/wmcnn/wmcnn.json'
        # args.config = 'configs/lapsrn.json'
        config = process_config(args.config)
    except Exception as e:
        print('[Exception] Configuration is invalid, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] Refer to: python main_train.py -c experiments/wmcnn/wmcnn.json')
        exit(0)

    print('[INFO] Building graph...')
    try:
        Net = importlib.import_module('models.{}'.format(config['trainer']['net'])).Net
        model = Net(config=config['model'])
        print_network(model)
    except ModuleNotFoundError:
        raise RuntimeWarning("The model name is incorrect or does not exist! Please check!")

    print('[INFO] Loading data...')
    dl = TestImageLoader(config=config['test_data_loader'])

    print('[INFO] Predicting...')
    infer = SRInfer(model, config['trainer'])
    infer.predict(dl.get_test_data(), testset=config['test_data_loader']['test_path'],
                  upscale=config['test_data_loader']['upscale'])


if __name__ == '__main__':
    test_main()

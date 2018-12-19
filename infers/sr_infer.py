# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by B. Y. Zhang on 2018/8/20
"""
import os

from bases.infer_base import InferBase
from skimage import io, measure

import torch
import numpy as np
from scipy import misc
import math
import pywt
# import matlab
# import matlab.engine as meng
# from utils.utils import print_network
# import torch.nn.functional as F
# from torchvision import transforms as tfs
# from PIL import Image
# import tqdm

from utils import np_utils, imresize
from utils.utils import mkdir_if_not_exist, colorize, shave, modcrop


class SRInfer(InferBase):
    def __init__(self, model, config=None):
        super(SRInfer, self).__init__(config)
        if self.config['cuda'] and not torch.cuda.is_available():
            print("GPU is not available on this device! Running in CPU!")
            self.config['cuda'] = False

        model_path = os.path.join(self.config['cp_dir'], self.config['checkpoint'])
        # model_path = self.config['checkpoint']

        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)

        self.model = model
        # self.model.load_from_mat([os.path.join(self.config['cp_dir'], path) for path in model_path])
        # self.load_model_from_matlab(model_path)
        if self.config['cuda']:
            self.model = self.model.cuda()

    def load_model_from_matlab(self, path):
        import scipy.io as sio

        mat_model = sio.loadmat(path)
        mat_params = mat_model['net']['params']

        for i, (name, param) in enumerate(self.model.named_parameters()):
            # param_array = np.transpose(mat_params[0, 0]['value'][0, i])
            param_array = mat_params[0, 0]['value'][0, i]

            # print(name, param.data.shape, mat_params[0, 0]['name'][0, i][0], param_array.shape)

            if 'conv' in mat_params[0, 0]['name'][0, i][0]:
                param_array = np.transpose(param_array, axes=[-1, *range(len(param_array.shape)-1)])
                # print(mat_params[0, 0]['name'][0, i][0], param_array.shape)
                if len(param_array.shape) == 4:
                    param_array = np.transpose(param_array, axes=[0, 3, 1, 2])
                    # print(mat_params[0, 0]['name'][0, i][0], param_array.shape)

            if 'b' in mat_params[0, 0]['name'][0, i][0]:
                param_array = param_array.squeeze(-1)
                # print(mat_params[0, 0]['name'][0, i][0], param_array.shape)
            if mat_params[0, 0]['name'][0, i][0] == 'img_up_f':
                param_array = np.expand_dims(np.expand_dims(param_array, 0), 0)
                # print(mat_params[0, 0]['name'][0, i][0], param_array.shape)
            if mat_params[0, 0]['name'][0, i][0] == 'residual_conv_f':
                param_array = np.expand_dims(param_array, 0)
                # print(mat_params[0, 0]['name'][0, i][0], param_array.shape)

            if param.data.shape == param_array.shape:
                param.data = torch.from_numpy(param_array)
            else:
                print('layer {} assigned wrong!, torch shape {}, mat shape {}'.format(name,
                                                                                      param.data.shape,
                                                                                      param_array.shape))

    def predict(self, data, **kwargs):
        # eng = meng.start_matlab()
        # for name, param in self.model.named_parameters():
        #     a = param.clone().cpu().data.numpy()
        #     print(name, a.max(), a.min())
        # print('\n')
        save_dir = os.path.join(self.config['preds_dir'], kwargs['testset'])
        mkdir_if_not_exist(save_dir)
        self.model.eval()
        psnr_list = []
        ssim_list = []
        b_psnr_list = []
        b_ssim_list = []
        with torch.no_grad():
            for img_bundle in data:
                # print(img_bundle['name'])
                if "color" in self.config.keys() and self.config["color"]:
                    x = img_bundle['origin']
                    y = img_bundle['y']
                    multichannel = True
                else:
                    x = img_bundle['x']
                    y = img_bundle['y']
                    (rows, cols, channel) = y.shape
                    y, _, _ = np.split(y, indices_or_sections=channel, axis=2)
                    multichannel = False

                x = torch.from_numpy(x).float().view(1, -1, x.shape[0], x.shape[1])
                if self.config['cuda']:
                    x = x.cuda()
                # print(x[:5])
                lr_size = (x.shape[2], x.shape[3])
                hr_size = img_bundle['size']
                if self.config['progressive']:
                    inter_sizes = np_utils.interval_size(lr_size, hr_size, self.config['max_gradual_scale'])
                else:
                    inter_sizes = []
                inter_sizes.append(hr_size)

                
                if self.config['net'] == 'wmcnn':
                    preds = self.model(x)
                    preds = [p.data.cpu().numpy() for p in preds]
                    # preds = [matlab.double(p.data.cpu().numpy().squeeze().tolist()) for p in preds]
                    # preds = eng.idwt2(*preds, 'bior1.1')
                    preds = pywt.idwt2((preds[0], (preds[1:])), 'bior1.1')
                else:
                    preds = self.model(x)

                if isinstance(preds, list):
					# Y-channel's pixels are within [16, 235]
                    preds = np.clip(preds[-1].data.cpu().numpy(), 16/255, 235/255).astype(np.float64)
                    # preds = np.clip(preds[-1].data.cpu().numpy(), 0, 1).astype(np.float64)
                else:
                    try:
                        preds = preds.data.cpu().numpy()
                    except AttributeError:
                        preds = preds
                    # preds = preds.mul(255).clamp(0, 255).round().div(255)
                    preds = np.clip(preds, 16/255, 235/255).astype(np.float64)
                    # preds = np.clip(preds, 0, 1).astype(np.float64)

                preds = preds.squeeze()
                if len(preds.shape) == 3:
                    preds = preds.transpose([1, 2, 0])
                preds = modcrop(preds.squeeze(), kwargs['upscale'])
                preds_bd = shave(preds.squeeze(), kwargs['upscale'])
                y = modcrop(y.squeeze(), kwargs['upscale'])
                y_bd = shave(y.squeeze(), kwargs['upscale'])

                # print(preds_bd.shape, y_bd.shape)
                x = x.data.cpu().numpy().squeeze()
                bic = imresize.imresize(x, scalar_scale=kwargs['upscale'])
                bic = np.clip(bic, 16 / 255, 235 / 255).astype(np.float64)
                bic = shave(bic.squeeze(), kwargs['upscale'])
                b_psnr = measure.compare_psnr(bic, y_bd)
                b_ssim = measure.compare_ssim(bic, y_bd)
                b_psnr_list.append(b_psnr)
                b_ssim_list.append(b_ssim)

                m_psnr = measure.compare_psnr(preds_bd, y_bd)
                m_ssim = measure.compare_ssim(preds_bd, y_bd, multichannel=multichannel)
                print('PSNR of image {} is {}'.format(img_bundle['name'], m_psnr))
                print('SSIM of image {} is {}'.format(img_bundle['name'], m_ssim))
                psnr_list.append(m_psnr)
                ssim_list.append(m_ssim)
                self.save_preds(save_dir, preds, img_bundle, True)

        print('Averaged PSNR is {}'.format(np.mean(np.array(psnr_list))))
        print('Averaged SSIM is {}'.format(np.mean(np.array(ssim_list))))
        print('Averaged BIC PSNR is {}'.format(np.mean(np.array(b_psnr_list))))
        print('Averaged BIC SSIM is {}'.format(np.mean(np.array(b_ssim_list))))

    def save_preds(self, save_dir, preds, img_bundle, is_color=False):
        if isinstance(preds, list):
            # TODO: not in use
            for i, cascade in enumerate(preds):
                cascade = cascade.data.cpu().numpy()
                cascade = cascade.clip(16 / 255, 235 / 255)     # color range of Y channel is [16, 235]
                io.imsave(os.path.join(save_dir,  '{}_{}_gt.png'.format(img_bundle['name'], i)), cascade)
        else:
            # preds = preds.data.cpu().numpy()
            # preds = preds.clip(16 / 255, 235 / 255)
            preds = preds.squeeze()
            if is_color:
                img_shape = preds.shape[:2]
                cb = misc.imresize(img_bundle['cb'], size=img_shape, mode='F')
                cr = misc.imresize(img_bundle['cr'], size=img_shape, mode='F')

                preds = colorize(preds, cb, cr)
            else:
                preds = preds*255
            io.imsave(os.path.join(save_dir, '{}_gt.png'.format(img_bundle['name'].split('.')[0])), preds.astype(np.uint8))
